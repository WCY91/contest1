# -*- coding: utf-8 -*-
"""
🏆 AI CUP Cardiac Segmentation — SwinUNETR Fine-tune v2 (最終穩定版 - 移除不相容 Transform)
✅ 移除 RandElasticDeformationd，修復 ImportError。
"""

import os, json, numpy as np, torch
from datetime import datetime
from torch.amp import autocast
from torch.utils.data import DataLoader
from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandAffined,
    RandCropByPosNegLabeld, EnsureTyped, ToTensord,
    RandAdjustContrastd, RandGaussianSmoothd # <-- 保留這兩個
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ------------------ EMA (Exponential Moving Average) ------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k] = self.shadow[k] * self.decay + v * (1.0 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)

# ------------------ 自訂 Weighted DiceFocalLoss (修正版) ------------------
class DiceFocalLossWeighted(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super().__init__()
        self.base_loss = DiceFocalLoss(**kwargs)
        self.weight = weight

    def forward(self, input, target):
        loss = self.base_loss(input, target)
        if self.weight is not None and loss.ndim == 2:
             w = self.weight.to(loss.device)
             weighted_loss = torch.mean(loss * w, dim=1) 
             return weighted_loss.mean()
        
        return loss.mean() if loss.ndim > 0 else loss

# ------------------ Light TTA 函數 (4 組合固定) ------------------
def tta_inference(model, img, roi_size, sw_batch_size, overlap):
    """固定使用 4 組合 TTA：原始 + D, H, W 軸翻轉"""
    dims_list = [[2], [3], [4]] 
    outputs = []
    
    for dims in dims_list:
        aug_img = torch.flip(img, dims=dims)
        logits = sliding_window_inference(
            aug_img, roi_size, sw_batch_size, model, overlap=overlap, mode="gaussian"
        )
        outputs.append(torch.flip(F.softmax(logits, dim=1), dims=dims))
        
    logits_orig = sliding_window_inference(
        img, roi_size, sw_batch_size, model, overlap=overlap, mode="gaussian"
    )
    outputs.append(F.softmax(logits_orig, dim=1))
    
    return torch.mean(torch.stack(outputs), dim=0)


# ------------------ 主函數 ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = False 
    print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} (AMP: {USE_AMP})")

    workspace_dir = os.getcwd()
    data_name = "chgh"
    exp_name = "AICUP_training"
    data_json = os.path.join(workspace_dir, "exps", "data_dicts", data_name, f"{exp_name}.json")

    run_dir = os.path.join(workspace_dir, "exps", "exps", "unetrpp", data_name, "tune_results", exp_name)
    os.makedirs(run_dir, exist_ok=True)
    resume_ckpt = os.path.join(run_dir, "best_model.pth")

    cfg = {
        "a_min": -100, "a_max": 400,
        "spacing": (0.7, 0.7, 0.7),
        "roi_size": (128, 128, 96),
        "batch_size": 1, "val_batch_size": 1,
        "num_classes": 4, "feature_size": 48,
        "use_checkpoint": False, "use_amp": USE_AMP,
        "lr": 4e-4, "weight_decay": 1e-5,
        "max_epochs": 50, "val_every": 1,
        "max_early_stop_count": 12,
        "sw_batch_size": 2, "overlap": 0.5
    }

    # ------------------ 載入資料 ------------------
    with open(data_json, "r") as f:
        d = json.load(f)
    train_files, val_files = d["train"], d["val"]
    print(f"📦 訓練樣本: {len(train_files)} | 驗證樣本: {len(val_files)}")

    train_tfms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=cfg["roi_size"], pos=1, neg=1, num_samples=2),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandAffined(keys=["image", "label"], prob=0.25,
                    rotate_range=(0.05, 0.05, 0.05),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest")),
        # ⚠️ 已移除 RandElasticDeformationd
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.8, 1.2)),
        RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.5, 1.0)),
        EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"]),
    ])

    val_tfms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, train_tfms, cache_rate=0.0, num_workers=4)
    val_ds = CacheDataset(val_files, val_tfms, cache_rate=0.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False, collate_fn=list_data_collate)

    # ------------------ 模型與權重 ------------------
    model = SwinUNETR(
        img_size=cfg["roi_size"], in_channels=1, out_channels=cfg["num_classes"],
        feature_size=cfg["feature_size"], use_checkpoint=cfg["use_checkpoint"]
    ).to(device)

    if os.path.exists(resume_ckpt):
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"🔁 已載入先前最佳權重: {resume_ckpt}")
    else:
        print(f"⚠️ 找不到權重 {resume_ckpt}，將從頭開始。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # 移除 GradScaler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)

    # ------------------ Loss / Metric / EMA ------------------
    weights_tensor = torch.tensor([0.1, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)

    def focal_weight(epoch):
        return float(min(0.80, 0.55 + 0.25 * (epoch / 15.0)))

    loss_fn = DiceFocalLossWeighted(
        to_onehot_y=True, softmax=True,
        lambda_dice=0.35, lambda_focal=focal_weight(0),
        gamma=3.0, weight=weights_tensor
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    ema = EMA(model, decay=0.999)

    best_score, no_improve = 0.752, 0
    scores_window = []
    log_file = os.path.join(run_dir, f"finetune_log_{datetime.now().strftime('%m%d_%H%M')}.csv")
    print("epoch,loss,dice,iou,score,lr", file=open(log_file, "w"))

    # ------------------ 訓練主迴圈 ------------------
    print(f"🚀 開始 Focal 微調 | {cfg['max_epochs']} epochs | lr={cfg['lr']}")
    for epoch in range(cfg["max_epochs"]):
        model.train()
        running_loss = 0.0
        loss_fn.base_loss.lambda_focal = focal_weight(epoch)
        cur_lr = optimizer.param_groups[0]["lr"]

        for batch in train_loader:
            img, lab = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # 🚨 修正訓練步驟：移除 GradScaler 和 autocast 區塊
            logits = model(img)
            loss = loss_fn(logits, lab)
            
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        scheduler.step(epoch + 1)
        print(f"🌀 Epoch {epoch+1}/{cfg['max_epochs']} | TrainLoss={avg_loss:.4f} | LR={cur_lr:.7f} | λ_focal={loss_fn.base_loss.lambda_focal:.2f}")

        # ------------------ 驗證 ------------------
        if (epoch + 1) % cfg["val_every"] == 0:
            print("🧪 驗證中 (4-way TTA)...")
            model.eval()
            dice_metric.reset()
            per_class_iou = []

            with torch.no_grad():
                for val_data in val_loader:
                    img, lab = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # 💥 使用 TTA Inference 得到 Softmax 平均結果
                    pred_soft_tta = tta_inference(model, img, cfg["roi_size"], cfg["sw_batch_size"], cfg["overlap"])
                        
                    # 處理 Dice Metric
                    pred_arg = pred_soft_tta.argmax(dim=1)
                    y_pred = [F.one_hot(pred_arg[0], cfg["num_classes"]).permute(3,0,1,2).unsqueeze(0).float()]
                    y_true = [F.one_hot(lab[0,0].long(), cfg["num_classes"]).permute(3,0,1,2).unsqueeze(0).float()]
                    dice_metric(y_pred=y_pred, y=y_true)
                    
                    # IoU
                    ious = []
                    for c in range(cfg["num_classes"]):
                        inter = ((pred_arg == c) & (lab.squeeze(1) == c)).sum().item()
                        union = ((pred_arg == c) | (lab.squeeze(1) == c)).sum().item()
                        if union > 0:
                            ious.append(inter / union)
                    if ious:
                        per_class_iou.append(np.mean(ious))

            mean_dice = dice_metric.aggregate().item()
            mean_iou = float(np.mean(per_class_iou)) if per_class_iou else 0.0
            score = 0.5 * (mean_dice + mean_iou)
            scores_window.append(score)
            
            if len(scores_window) > 5:
                scores_window.pop(0)
            smooth_score = np.mean(scores_window)

            print(f"📊 Epoch {epoch+1}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, Score={score:.4f} (Smooth={smooth_score:.4f})")
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{mean_dice:.4f},{mean_iou:.4f},{score:.4f},{cur_lr:.6f}\n")

            if smooth_score > best_score:
                best_score, no_improve = smooth_score, 0
                ema.apply(model)
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model_finetune_v2.pth"))
                print(f"💾 儲存最佳微調模型 (Score={smooth_score:.4f})")
            else:
                no_improve += 1
                print(f"⚠️ 未改善 {no_improve}/{cfg['max_early_stop_count']}")
                if no_improve >= cfg["max_early_stop_count"]:
                    print("🛑 Early Stopping 觸發。")
                    break

    print(f"✅ 微調結束 | 最佳分數: {best_score:.4f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()