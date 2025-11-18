# -*- coding: utf-8 -*-
import os, json, random, math, numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandAffined,
    RandCropByPosNegLabeld, EnsureTyped, ToTensord, AsDiscrete,
    RandGaussianNoised, RandBiasFieldd, RandAdjustContrastd
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def param_groups_weight_decay(model, wd, skip_list=('bias', 'norm', 'bn', 'ln', 'gn', 'embedding')):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(k in name.lower() for k in skip_list):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


class ModelEMA:
    """Exponential Moving Average of model weights for more stable eval."""
    def __init__(self, model, decay=0.999):
        self.ema = type(model)(
            img_size=model.img_size, in_channels=1,
            out_channels=model.out_channels, feature_size=model.feature_size,
            use_checkpoint=getattr(model, "use_checkpoint", False)
        ).to(next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict())
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))


# 輕量 TTA（驗證用）
TTA_FLIPS_VAL = [(), (2,), (3,), (4,)]  # 訓練過程用4組，省時間

def infer_with_tta(images, roi_size, sw_bs, model, flips):
    logits_sum = None
    for dims in flips:
        x = images
        if dims: x = torch.flip(x, dims=dims)
        logits = sliding_window_inference(
            x, roi_size, sw_bs, model, mode="gaussian", overlap=0.65
        )
        if dims: logits = torch.flip(logits, dims=dims)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(len(flips))


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ---------------- CONFIG ----------------
    workspace_dir = os.getcwd()
    data_name = "chgh"
    exp_name = "AICUP_training"

    data_json = os.path.join(workspace_dir, "exps", "data_dicts", data_name, f"{exp_name}.json")
    run_dir = os.path.join(workspace_dir, "exps", "exps", "unetrpp", data_name, "tune_results", exp_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg = {
        "a_min": -102, "a_max": 423,
        "spacing": (0.7, 0.7, 0.8),
        "roi_size": (128, 128, 96),
        "batch_size": 1, "val_batch_size": 1,
        "num_classes": 4, "feature_size": 48,
        "use_checkpoint": False, "use_amp": True,
        "lr": 5e-4, "weight_decay": 2e-4,
        "max_epochs": 110,             # 拉長尾巴
        "val_every": 1,
        "max_early_stop_count": 15,    # 給降LR後反彈空間
        "ema_decay": 0.999
    }

    # ---------------- LOAD DATA ----------------
    print(f"📂 載入資料描述檔: {data_json}")
    with open(data_json, "r") as f:
        d = json.load(f)
    train_files, val_files = d["train"], d["val"]
    print(f"📦 訓練樣本數: {len(train_files)} | 驗證樣本數: {len(val_files)}")

    # 針對 class3 稀有樣本再加權（你的清單）
    rare_cls_ids = ["0001","0012","0013","0018","0032","0033","0036","0037","0047","0048"]
    rare_cls_files = [f for f in train_files if any(rid in f["label"] for rid in rare_cls_ids)]
    train_files = rare_cls_files * 3 + train_files
    print(f"⚖️ 強化 class3 樣本數: {len(rare_cls_files)} 筆 * 3")

    # --- Transform ---
    train_tfms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # ★更穩定的前景抽樣
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=cfg["roi_size"], pos=1, neg=1, num_samples=2),
        # 幾何與強化對比/噪聲（小心別太大）
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0,1,2]),
        RandAffined(keys=["image", "label"], prob=0.15,
                    rotate_range=(0.05,0.05,0.05), scale_range=(0.1,0.1,0.1),
                    mode=("bilinear","nearest")),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
        RandBiasFieldd(keys=["image"], prob=0.1, coeff_range=(0.0, 0.3)),
        RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.7,1.5)),
        EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"]),
    ])

    val_tfms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"]),
    ])

    print("🧠 建立 Dataset ...")
    train_ds = CacheDataset(train_files, train_tfms, cache_rate=0.0, num_workers=4)
    val_ds = CacheDataset(val_files, val_tfms, cache_rate=0.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False, collate_fn=list_data_collate)

    # ---------------- MODEL ----------------
    print("⚙️ 初始化 SwinUNETR 模型中...")
    model = SwinUNETR(
        img_size=cfg["roi_size"],
        in_channels=1, out_channels=cfg["num_classes"],
        feature_size=cfg["feature_size"], use_checkpoint=cfg["use_checkpoint"]
    ).to(device)
    print("✅ 模型初始化完成!")

    # ---------------- LOSS / OPT / EMA ----------------
    # 更重疊的權重略高，幫助 Dice/IoU
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.5)

    optimizer = torch.optim.AdamW(
        param_groups_weight_decay(model, wd=cfg["weight_decay"]),
        lr=cfg["lr"]
    )
    scaler = GradScaler(enabled=cfg["use_amp"])

    ema = ModelEMA(model, decay=cfg["ema_decay"])

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=True, reduction="mean")

    def lr_lambda(epoch): return min(1.0, float(epoch + 1) / 3)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"] - 3, eta_min=1e-6)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True, min_lr=1e-6)

    best_score = -1.0
    no_improve = 0
    log_file = os.path.join(run_dir, f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt")

    print(f"🚀 開始訓練 | epochs={cfg['max_epochs']} | base_lr={cfg['lr']} | wd={cfg['weight_decay']}")
    print("="*80)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(cfg["max_epochs"]):
        model.train()
        running_loss = 0.0
        phase = "Warmup" if epoch < 3 else "Train"
        print(f"\n🌀 Epoch {epoch+1}/{cfg['max_epochs']} [{phase}]")

        for i, batch in enumerate(train_loader):
            img, lab = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg["use_amp"]):
                out = model(img)
                loss = loss_fn(out, lab)

            scaler.scale(loss).backward()
            # 梯度裁剪 + AMP 正確順序
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # EMA 跟隨
            ema.update(model)

            if (i+1) % 5 == 0:
                print(f"   🔹 Batch {i+1}/{len(train_loader)} | Loss={loss.item():.4f}")

        if epoch < 3: warmup_scheduler.step()
        else: cosine_scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"📉 平均訓練 Loss={avg_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

        # ---------------- VALIDATION ----------------
        if (epoch + 1) % cfg["val_every"] == 0:
            print("🧪 驗證中（EMA權重 + 4xTTA）...")
            model.eval()
            dice_metric.reset()
            per_class_iou = []

            # 用 EMA 權重做驗證
            bak = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.ema.state_dict(), strict=True)

            post_pred = AsDiscrete(argmax=True, to_onehot=cfg["num_classes"])
            post_label = AsDiscrete(to_onehot=cfg["num_classes"])

            with torch.no_grad():
                for val_data in val_loader:
                    img, lab = val_data["image"].to(device), val_data["label"].to(device)
                    with autocast(enabled=cfg["use_amp"]):
                        logits = infer_with_tta(img, cfg["roi_size"], 2, model, TTA_FLIPS_VAL)
                        pred_soft = torch.softmax(logits, dim=1)

                    val_outputs = [post_pred(i) for i in decollate_batch(pred_soft)]
                    val_labels  = [post_label(i) for i in decollate_batch(lab)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

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
            mean_dice = dice_vals.mean().item()
            mean_iou = float(np.mean(per_class_iou)) if per_class_iou else 0.0
            score = (mean_dice + mean_iou) / 2

            # 還原原權重，繼續訓練
            model.load_state_dict(bak, strict=True)

            msg = f"Epoch [{epoch+1}/{cfg['max_epochs']}], Loss={avg_loss:.4f}, Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, Score={score:.4f}"
            print("📊", msg)
            dice_str = " | ".join([f"class{i}={v:.4f}" for i, v in enumerate(dice_vals.tolist())])
            print(f"🎯 每類 Dice: {dice_str} | 平均 Dice={mean_dice:.4f}")

            with open(log_file, "a") as f:
                f.write(msg + "\n")

            # 停滯就降LR
            plateau_scheduler.step(score)

            # 存最佳（以 Score）
            if score > best_score:
                best_score = score
                no_improve = 0
                save_path = os.path.join(run_dir, f"best_model_{best_score:.4f}.pth")
                torch.save(ema.ema.state_dict(), save_path)  # ★保存EMA權重為best
                print(f"💾 儲存最佳模型(EMA) -> {save_path}")
            else:
                no_improve += 1
                print(f"⚠️ 未改善次數: {no_improve}/{cfg['max_early_stop_count']}")
                if no_improve >= cfg["max_early_stop_count"]:
                    print("🛑 EarlyStopping 觸發，結束訓練。")
                    break

    print("="*80)
    print(f"✅ 訓練結束 | 最佳綜合分數: {best_score:.4f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
