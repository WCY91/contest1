# -*- coding: utf-8 -*-
"""
SwinUNETR + Teacher Distillation (修正版)
✅ 完全對齊 train_strong.py 的 Dice / Validation 計算方式
✅ 含 Student & Teacher (SwinViT) 模式
✅ AMP + Warmup + Cosine + EarlyStop
"""

import os, json, torch, numpy as np
from datetime import datetime
from typing import List
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandAffined,
    RandCropByPosNegLabeld, EnsureTyped, ToTensord, AsDiscrete
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


# ---------------- Utils ----------------
def set_seed(seed: int = 1027):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kd_loss_kl_from_logits(s_logits, t_logits, T=2.0):
    s = F.log_softmax(s_logits / T, dim=1)
    t = F.softmax(t_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def kd_loss_kl_from_vec(s_vec, t_vec, T=2.0):
    s = F.log_softmax(s_vec / T, dim=1)
    t = F.softmax(t_vec / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def make_dataloaders(data_json: str, cfg: dict):
    with open(data_json, "r") as f:
        d = json.load(f)
    train_files, val_files = d["train"], d["val"]

    train_tfms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=cfg["roi_size"], pos=1, neg=1, num_samples=2),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0,1,2]),
        RandAffined(keys=["image", "label"], prob=0.15,
                    rotate_range=(0.05,0.05,0.05), scale_range=(0.1,0.1,0.1),
                    mode=("bilinear","nearest")),
        EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"]),
    ])

    val_tfms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"], b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, train_tfms, cache_rate=0.0, num_workers=4)
    val_ds = CacheDataset(val_files, val_tfms, cache_rate=0.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False, collate_fn=list_data_collate)
    return train_loader, val_loader


# ---------------- Main ----------------
def main():
    cfg = {
        "data_json": r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\data_dicts\chgh\AICUP_training.json",
        "run_dir": r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\runs_kd_strong",
        "a_min": -102, "a_max": 423, "spacing": (0.7, 0.7, 0.8),
        "roi_size": (128, 128, 96), "batch_size": 1, "val_batch_size": 1,
        "num_classes": 4, "feature_size": 48,
        "use_checkpoint": True, "use_amp": True,
        "lr": 5e-4, "weight_decay": 5e-4,
        "max_epochs": 80, "warmup_epochs": 3, "val_every": 1,
        "max_early_stop_count": 8,
        "teacher_mode": "swinvit",
        "kd_lambda": 0.2, "kd_T": 2.0,
        "teacher_ckpt": r"C:\Users\aclab_public\Downloads\model_swinvit.pt",
        "student_ckpt": r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\exps\unetrpp\chgh\tune_results\AICUP_training\best_model.pth",
    }

    os.makedirs(cfg["run_dir"], exist_ok=True)
    log_file = os.path.join(cfg["run_dir"], f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
    best_path = os.path.join(cfg["run_dir"], "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1027)
    print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ---------------- Data ----------------
    print(f"📂 載入資料描述檔: {cfg['data_json']}")
    train_loader, val_loader = make_dataloaders(cfg["data_json"], cfg)
    print(f"📦 訓練批數: {len(train_loader)} | 驗證批數: {len(val_loader)}")

    # ---------------- Models ----------------
    print("⚙️ 初始化 Student / Teacher 模型...")
    student = SwinUNETR(
        img_size=cfg["roi_size"],
        in_channels=1, out_channels=cfg["num_classes"],
        feature_size=cfg["feature_size"],
        use_checkpoint=cfg["use_checkpoint"]
    ).to(device)
    if os.path.exists(cfg["student_ckpt"]):
        student.load_state_dict(torch.load(cfg["student_ckpt"], map_location="cpu"), strict=False)
        print(f"✅ Student 預設權重載入成功: {cfg['student_ckpt']}")

    teacher = None
    if cfg["teacher_mode"] == "swinvit" and os.path.exists(cfg["teacher_ckpt"]):
        teacher = SwinViT(
            in_chans=1, embed_dim=cfg["feature_size"],
            window_size=(7,7,7), patch_size=(2,2,2),
            depths=[2,2,2,2], num_heads=[3,6,12,24],
            spatial_dims=3
        ).to(device)
        t_state = torch.load(cfg["teacher_ckpt"], map_location="cpu")
        t_state = t_state.get("state_dict", t_state)
        teacher.load_state_dict(t_state, strict=False)
        teacher.eval()
        print(f"🎓 Teacher(SwinViT) 權重載入成功: {cfg['teacher_ckpt']}")

    # ---------------- Opt / Loss ----------------
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler(enabled=cfg["use_amp"])
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    def lr_lambda(epoch): return min(1.0, float(epoch+1)/cfg["warmup_epochs"])
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"]-cfg["warmup_epochs"], eta_min=1e-6)

    best_score = -1
    no_improve = 0

    print(f"🚀 開始訓練 SwinUNETR + KD | epochs={cfg['max_epochs']} | KD λ={cfg['kd_lambda']}")
    print("="*90)

    for epoch in range(cfg["max_epochs"]):
        student.train()
        running_loss = 0.0
        phase = "Warmup" if epoch < cfg["warmup_epochs"] else "Train"
        print(f"\n🌀 Epoch {epoch+1}/{cfg['max_epochs']} [{phase}]")

        # -------- Train --------
        for i, batch in enumerate(train_loader):
            img, lab = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg["use_amp"]):
                s_logits = student(img)
                seg_loss = loss_fn(s_logits, lab)
                loss = seg_loss

                if teacher is not None:
                    with torch.no_grad():
                        t_feats = teacher(img)
                        t_feat = t_feats[-1]
                    s_feat = student.swinViT(img)[-1]
                    t_vec = t_feat.mean(dim=(2,3,4))
                    s_vec = s_feat.mean(dim=(2,3,4))
                    kd_loss = kd_loss_kl_from_vec(s_vec, t_vec, T=cfg["kd_T"])
                    loss = seg_loss + cfg["kd_lambda"] * kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if (i+1) % 5 == 0:
                print(f"   🔹 Batch {i+1}/{len(train_loader)} | Loss={loss.item():.4f}")

        if epoch < cfg["warmup_epochs"]: warmup_scheduler.step()
        else: cosine_scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"📉 平均訓練 Loss={avg_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

        # -------- Validation --------
        if (epoch + 1) % cfg["val_every"] == 0:
            print("🧪 驗證中...")
            student.eval()
            dice_metric.reset()
            per_class_iou = []

            post_pred = AsDiscrete(argmax=True, to_onehot=cfg["num_classes"])
            post_label = AsDiscrete(to_onehot=cfg["num_classes"])

            with torch.no_grad():
                for v in val_loader:
                    img, lab = v["image"].to(device), v["label"].to(device)
                    with autocast(enabled=cfg["use_amp"]):
                        logits = sliding_window_inference(img, cfg["roi_size"], 2, student, mode="gaussian")
                        pred_soft = torch.softmax(logits, dim=1)

                    val_outputs = [post_pred(i) for i in decollate_batch(pred_soft)]
                    val_labels  = [post_label(i) for i in decollate_batch(lab)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                    # IoU（基於 argmax）
                    pred_arg = pred_soft.argmax(dim=1)
                    ious = []
                    for c in range(cfg["num_classes"]):
                        inter = ((pred_arg == c) & (lab.squeeze(1) == c)).sum().item()
                        union = ((pred_arg == c) | (lab.squeeze(1) == c)).sum().item()
                        if union > 0:
                            ious.append(inter / union)
                    if ious:
                        per_class_iou.append(np.mean(ious))

            dice_vals = dice_metric.aggregate().cpu()
            if dice_vals.numel() < cfg["num_classes"]:
                pad = torch.zeros(cfg["num_classes"] - dice_vals.numel())
                dice_vals = torch.cat([dice_vals, pad])

            mean_dice = dice_vals.mean().item()
            mean_iou = float(np.mean(per_class_iou)) if per_class_iou else 0.0
            score = (mean_dice + mean_iou) / 2

            msg = f"Epoch [{epoch+1}/{cfg['max_epochs']}], Loss={avg_loss:.4f}, Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, Score={score:.4f}"
            print("📊", msg)
            dice_str = " | ".join([f"class{i}={v:.4f}" for i, v in enumerate(dice_vals.tolist())])
            print(f"🎯 每類 Dice: {dice_str} | 平均 Dice={mean_dice:.4f}")

            with open(log_file, "a") as f: f.write(msg + "\n")

            if score > best_score:
                best_score = score
                no_improve = 0
                torch.save(student.state_dict(), best_path)
                print(f"💾 儲存最佳模型 (Dice={mean_dice:.4f}, IoU={mean_iou:.4f}) -> {best_path}")
            else:
                no_improve += 1
                print(f"⚠️ 未改善次數: {no_improve}/{cfg['max_early_stop_count']}")
                if no_improve >= cfg["max_early_stop_count"]:
                    print("🛑 Early Stopping 觸發，結束訓練。")
                    break

    print("="*80)
    print(f"✅ 訓練結束 | 最佳分數: {best_score:.4f} | 模型已保存至: {best_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
