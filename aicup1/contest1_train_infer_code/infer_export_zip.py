# -*- coding: utf-8 -*-
"""
✅ Final Inference Script for AI CUP Cardiac Segmentation
- 完全對齊訓練時的 preprocessing / spacing
- 使用 Invertd 還原原始空間
- 自動附加 timestamp (年月日時分)
- 自動輸出可上傳 zip
"""

import os, glob, zipfile
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, AsDiscrete, Invertd, SaveImaged
)
from monai.inferers import sliding_window_inference


# ================== 基本設定 ==================
workspace_dir = os.getcwd()
data_name = "chgh"
exp_name = "AICUP_training"

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_path = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\exps\unetrpp\chgh\tune_results\AICUP_training\best_model_finetune_v2.pth"
test_dir   = r"C:\Users\aclab_public\Downloads\aicup_result"
output_dir = rf"C:\Users\aclab_public\Downloads\aicup_pred_output_SwinUNETR_{timestamp}"
zip_name   = f"result_strong_{timestamp}.zip"
# model_path = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\exps\unetrpp\chgh\distill_results\AICUP_training\best_student.pth"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ================== 模型設定 ==================
num_classes = 4
roi_size = (128, 128, 96)
spacing = (0.7, 0.7, 0.7)
a_min, a_max = -75, 450
sw_batch_size = 1
overlap = 0.25

model = SwinUNETR(
    img_size=roi_size,
    in_channels=1,
    out_channels=num_classes,
    feature_size=48,
    use_checkpoint=False
).to(device)

ckpt = torch.load(model_path, map_location=device)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)
model.eval()
print(f"💾 已載入模型權重: {os.path.basename(model_path)}")


# ================== 測試集轉換 ==================
test_files = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
data_dicts = [{"image": f} for f in test_files]

# ⚠️ 與訓練一致
test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
    ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image"], track_meta=True),
])

test_ds = Dataset(data=data_dicts, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

post_pred = AsDiscrete(argmax=True)
inverter = Invertd(
    keys="pred",
    transform=test_transforms,
    orig_keys="image",
    meta_keys="pred_meta_dict",
    orig_meta_keys="image_meta_dict",
    nearest_interp=True,
    to_tensor=False,
)

save_pred = SaveImaged(
    keys="pred",
    meta_keys="pred_meta_dict",
    output_dir=output_dir,
    output_postfix="",
    output_dtype=np.uint8,
    resample=False,
    print_log=False
)


# ================== 推論 ==================
print(f"🧠 開始推論 {len(test_files)} 筆測試影像...")

with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
    for batch in tqdm(test_loader):
        img = batch["image"].to(device)
        logits = sliding_window_inference(img, roi_size, sw_batch_size, model, overlap=overlap,mode="gaussian")
        preds = [post_pred(i) for i in decollate_batch(logits)]
        single = decollate_batch(batch)[0]
        single["pred"] = preds[0]
        single["pred_meta_dict"] = single["image_meta_dict"]

        # ✅ 還原原始 spacing / shape
        single = inverter(single)
        save_pred(single)

print(f"✅ 推論完成，結果儲存於: {output_dir}")


# ================== 打包 zip (修正版) ==================
zip_path = os.path.join(os.path.dirname(output_dir), zip_name)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 遞迴搜尋所有 .nii.gz，不論是否有子資料夾
    for f in sorted(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True)):
        zipf.write(f, os.path.basename(f))

print(f"📦 已建立上傳檔案: {zip_path}")
print("🎯 這份 zip 可直接上傳至 AI CUP Leaderboard 評分系統")

