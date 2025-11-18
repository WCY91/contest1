# -*- coding: utf-8 -*-
import os, json, torch, numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandAffined,
    RandCropByPosNegLabeld, EnsureTyped, ToTensord, AsDiscrete
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from datetime import datetime
from monai.networks.nets import SwinUNETR


def main():
    # ---------------- CUDA CHECK ----------------
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
        "a_min": -102,
        "a_max": 423,
        "spacing": (0.7, 0.7, 0.8),
        "roi_size": (128, 128, 96),
        "batch_size": 1,
        "val_batch_size": 1,
        "num_classes": 4,
        "feature_size": 48,
        "use_checkpoint": False,
        "use_amp": True,
        "lr": 5e-4,
        "weight_decay": 5e-4,
        "max_epochs": 90,
        "val_every": 1,
        "max_early_stop_count": 8,
    }

    # ---------------- LOAD DATA ----------------
    print(f"📂 載入資料描述檔: {data_json}")
    with open(data_json, "r") as f:
        d = json.load(f)
    train_files, val_files = d["train"], d["val"]
    print(f"📦 訓練樣本數: {len(train_files)} | 驗證樣本數: {len(val_files)}")

    rare_cls_ids = ["0001","0012","0013","0018","0032","0033","0036","0037","0047","0048"]
    rare_cls_files = [f for f in train_files if any(rid in f["label"] for rid in rare_cls_ids)]
    print(f"⚖️ 強化 class3 樣本數: {len(rare_cls_files)} 筆 * 3")
    train_files = rare_cls_files * 3 + train_files

    # --- Transform ---
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

    print("🧠 建立 CacheDataset 中 (約需 30~60 秒)...")
    train_ds = CacheDataset(train_files, train_tfms, cache_rate=0.0, num_workers=4)
    val_ds = CacheDataset(val_files, val_tfms, cache_rate=0.0, num_workers=2)
    print("✅ 資料載入完成!")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False, collate_fn=list_data_collate)

    # ---------------- MODEL ----------------
    print("⚙️ 初始化 SwinUNETR 模型中...")
    model = SwinUNETR(
        img_size=cfg["roi_size"],
        in_channels=1,
        out_channels=cfg["num_classes"],
        feature_size=cfg["feature_size"],
        use_checkpoint=cfg["use_checkpoint"]
    ).to(device)
    print("✅ 模型初始化完成!")

    # ---------------- LOSS / OPT ----------------
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler(enabled=cfg["use_amp"])

    # ✅ include_background=True
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=True, reduction="mean")

    def lr_lambda(epoch): return min(1.0, float(epoch + 1) / 3)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"] - 3, eta_min=1e-6)

    best_score = -1
    no_improve = 0
    log_file = os.path.join(run_dir, f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt")

    print(f"🚀 開始訓練 SwinUNETR | epochs={cfg['max_epochs']} | lr={cfg['lr']}")
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
            uniq = torch.unique(lab).cpu().numpy()
            print(f"   👀 labels in batch: {uniq}")

            with autocast(enabled=cfg["use_amp"]):
                out = model(img)
                loss = loss_fn(out, lab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if (i+1) % 5 == 0:
                print(f"   🔹 Batch {i+1}/{len(train_loader)} | Loss={loss.item():.4f}")

        if epoch < 3: warmup_scheduler.step()
        else: cosine_scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"📉 平均訓練 Loss={avg_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

        # ---------------- VALIDATION ----------------
        if (epoch + 1) % cfg["val_every"] == 0:
            print("🧪 驗證中...")
            model.eval()
            dice_metric.reset()
            per_class_iou = []

            # ✅ AsDiscrete + decollate_batch
            post_pred = AsDiscrete(argmax=True, to_onehot=cfg["num_classes"])
            post_label = AsDiscrete(to_onehot=cfg["num_classes"])

            with torch.no_grad():
                for val_data in val_loader:
                    img, lab = val_data["image"].to(device), val_data["label"].to(device)
                    with autocast(enabled=cfg["use_amp"]):
                        logits = sliding_window_inference(img, cfg["roi_size"], 2, model, mode="gaussian")
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
            mean_dice = dice_vals.mean().item()
            mean_iou = float(np.mean(per_class_iou)) if per_class_iou else 0.0
            score = (mean_dice + mean_iou) / 2

            msg = f"Epoch [{epoch+1}/{cfg['max_epochs']}], Loss={avg_loss:.4f}, Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, Score={score:.4f}"
            print("📊", msg)
            dice_str = " | ".join([f"class{i}={v:.4f}" for i, v in enumerate(dice_vals.tolist())])
            print(f"🎯 每類 Dice: {dice_str} | 平均 Dice={mean_dice:.4f}")

            with open(log_file, "a") as f:
                f.write(msg + "\n")

            if score > best_score:
                best_score = score
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                print(f"💾 儲存最佳模型 (Dice={mean_dice:.4f}, IoU={mean_iou:.4f})")
            else:
                no_improve += 1
                print(f"⚠️ 未改善次數: {no_improve}/{cfg['max_early_stop_count']}")
                if no_improve >= cfg["max_early_stop_count"]:
                    print("🛑 Early Stopping 觸發，結束訓練。")
                    break

    print("="*80)
    print(f"✅ 訓練結束 | 最佳綜合分數: {best_score:.4f}")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
