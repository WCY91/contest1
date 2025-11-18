# -*- coding: utf-8 -*-
"""
🏫 Teacher–Student Knowledge Distillation Training (train_distill_v1.py)
- Teacher: 已訓練的 SwinUNETR (Frozen)
- Student: 新 SwinUNETR 學習 Teacher 的 soft output + Ground Truth
- 損失: α * KD(T=2) + (1-α) * DiceCE
- 可直接放在 CardiacSegV2 目錄執行
"""

import os, json, torch, numpy as np
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from monai.data import CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandAffined,
    RandCropByPosNegLabeld, RandGaussianNoised, RandScaleIntensityd,
    RandAdjustContrastd, EnsureTyped, ToTensord, AsDiscrete
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

# ---------------- 基本設定 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

workspace_dir = os.getcwd()
data_name = "chgh"
exp_name = "AICUP_training"
teacher_ckpt = os.path.join(
    workspace_dir, "exps", "exps", "unetrpp", data_name, "tune_results", exp_name, "best_model.pth"
)
run_dir = os.path.join(workspace_dir, "exps", "exps", "unetrpp", data_name, "distill_results", exp_name)
os.makedirs(run_dir, exist_ok=True)

# ---------------- 訓練參數 ----------------
cfg = {
    "a_min": -75,
    "a_max": 450,
    "spacing": (0.7, 0.7, 0.8),
    "roi_size": (128, 128, 96),
    "batch_size": 1,
    "num_classes": 4,
    "feature_size": 48,
    "lr": 4e-4,
    "weight_decay": 2e-4,
    "max_epochs": 80,
    "val_every": 1,
    "alpha": 0.4,   # teacher signal weight
    "T": 2.0,       # temperature
}

# ---------------- 載入資料 ----------------
data_json = os.path.join(workspace_dir, "exps", "data_dicts", data_name, f"{exp_name}.json")
print(f"📂 載入資料描述檔: {data_json}")
with open(data_json, "r") as f:
    d = json.load(f)
train_files, val_files = d["train"], d["val"]
print(f"📦 訓練樣本數: {len(train_files)} | 驗證樣本數: {len(val_files)}")

# ---------------- Transform ----------------
train_tfms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear","nearest")),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"], b_min=0.0, b_max=1.0),
    CropForegroundd(keys=["image","label"], source_key="image"),
    RandCropByPosNegLabeld(keys=["image","label"], label_key="label",
                           spatial_size=cfg["roi_size"], pos=1, neg=1, num_samples=2),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[0,1,2]),
    RandAffined(keys=["image","label"], prob=0.25,
                rotate_range=(0.05,0.05,0.05), scale_range=(0.1,0.1,0.1),
                mode=("bilinear","nearest")),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0, std=0.01),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
    RandAdjustContrastd(keys=["image"], gamma=(0.7,1.3), prob=0.15),
    EnsureTyped(keys=["image","label"]), ToTensord(keys=["image","label"]),
])
val_tfms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=cfg["a_min"], a_max=cfg["a_max"], b_min=0.0, b_max=1.0),
    EnsureTyped(keys=["image","label"]), ToTensord(keys=["image","label"]),
])

train_ds = CacheDataset(train_files, train_tfms, cache_rate=0.0, num_workers=4)
val_ds = CacheDataset(val_files, val_tfms, cache_rate=0.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=list_data_collate)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=list_data_collate)
print("✅ 資料載入完成!")

# ---------------- 模型設定 ----------------
print("🧠 初始化 Teacher / Student 模型中...")
teacher = SwinUNETR(in_channels=1, out_channels=cfg["num_classes"], feature_size=cfg["feature_size"]).to(device)
teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False  # freeze teacher

student = SwinUNETR(in_channels=1,img_size=cfg["roi_size"], out_channels=cfg["num_classes"], feature_size=cfg["feature_size"]).to(device)
optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
scaler = GradScaler(enabled=True)
dicece = DiceCELoss(to_onehot_y=True, softmax=True)
kl = torch.nn.KLDivLoss(reduction="batchmean")

dice_metric = DiceMetric(include_background=True, reduction="mean")

# ---------------- 蒸餾損失 ----------------
def distill_loss(student_logits, teacher_logits, label, T=cfg["T"], alpha=cfg["alpha"]):
    loss_dice = dicece(student_logits, label)
    s_soft = torch.log_softmax(student_logits / T, dim=1)
    t_soft = torch.softmax(teacher_logits / T, dim=1)
    loss_kd = kl(s_soft, t_soft) * (T * T)
    return alpha * loss_kd + (1 - alpha) * loss_dice

# ---------------- 訓練 ----------------
best_dice = -1
log_path = os.path.join(run_dir, f"distill_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
print(f"🚀 開始蒸餾訓練 SwinUNETR | α={cfg['alpha']} | T={cfg['T']} | lr={cfg['lr']}")

for epoch in range(cfg["max_epochs"]):
    student.train()
    running_loss = 0
    for i, batch in enumerate(train_loader):
        img, lab = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast("cuda"):
            t_pred = teacher(img)
        with autocast("cuda"):
            s_pred = student(img)
            loss = distill_loss(s_pred, t_pred, lab)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    avg_loss = running_loss / max(1, len(train_loader))
    print(f"🌀 Epoch [{epoch+1}/{cfg['max_epochs']}] | Loss={avg_loss:.4f}")

    # ---------------- 驗證 ----------------
    if (epoch + 1) % cfg["val_every"] == 0:
        student.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_data in val_loader:
                img, lab = val_data["image"].to(device), val_data["label"].to(device)
                with autocast("cuda"):
                    logits = sliding_window_inference(img, cfg["roi_size"], 2, student)
                post_pred = [AsDiscrete(argmax=True, to_onehot=cfg["num_classes"])(i)
                             for i in decollate_batch(logits)]
                post_label = [AsDiscrete(to_onehot=cfg["num_classes"])(i)
                              for i in decollate_batch(lab)]
                dice_metric(y_pred=post_pred, y=post_label)
        dice_val = dice_metric.aggregate().mean().item()
        print(f"📊 Val Dice={dice_val:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{dice_val:.4f}\n")

        if dice_val > best_dice:
            best_dice = dice_val
            torch.save(student.state_dict(), os.path.join(run_dir, "best_student.pth"))
            print(f"💾 儲存最佳 Student (Dice={dice_val:.4f})")

print("="*80)
print(f"✅ 訓練結束 | 最佳 Dice={best_dice:.4f}")
