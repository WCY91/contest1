# -*- coding: utf-8 -*-
"""
🏆 AI CUP Cardiac Segmentation — Single SwinUNETR + Light TTA (Softmax Averaging)
✅ 策略：單模型 SwinUNETR，使用 TTA 翻轉，在 logits 空間反轉後，Softmax 平均概率圖。
✅ 保證維度穩定，最終輸出 zip 可直接上傳。
"""

import os, glob, zipfile, copy
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F
from monai.data import Dataset, DataLoader, decollate_batch, MetaTensor
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, Invertd, AsDiscrete
)
from monai.inferers import sliding_window_inference
from monai.transforms.utils import convert_to_tensor # 引入用於處理 Argmax 結果

# ================== 基本設定 ==================
workspace_dir = os.getcwd()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 增加秒，確保唯一性
model_path = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\exps\unetrpp\chgh\tune_results\AICUP_training\best_model.pth"
test_dir = r"C:\Users\aclab_public\Downloads\aicup_result"
output_dir = rf"C:\Users\aclab_public\Downloads\aicup_pred_output_SwinUNETR_TTA_{timestamp}"
zip_name= f"result_TTA_{timestamp}.zip"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ================== 模型設定 & 載入 ==================
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

try:
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"💾 已載入模型權重: {os.path.basename(model_path)}")
except Exception as e:
    print(f"🛑 模型載入失敗: {e}")
    exit()

# ================== TTA 定義 (翻轉) ==================
# 這裡使用 4 組合 (無翻轉, D軸, H軸, W軸)
def tta_flips(img):
    # img 預期 shape: (B, C, D, H, W)
    return [
        (img, lambda x: x), # 1. 無翻轉
        (torch.flip(img, dims=[-3]), lambda x: torch.flip(x, dims=[-3])), # 2. D (Z) 軸
        (torch.flip(img, dims=[-2]), lambda x: torch.flip(x, dims=[-2])), # 3. H (Y) 軸
        (torch.flip(img, dims=[-1]), lambda x: torch.flip(x, dims=[-1])), # 4. W (X) 軸
    ]


# ================== 測試集轉換 & Post-processing ==================
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

post_pred_argmax = AsDiscrete(argmax=True)

inverter = Invertd(
    keys="pred",
    transform=test_transforms,
    orig_keys="image",
    meta_keys="pred_meta_dict",
    orig_meta_keys="image_meta_dict",
    nearest_interp=True, # Label Map 還原必須使用最近鄰插值
    to_tensor=False,
)

# 避免使用 SaveImaged 類，直接手動存檔，確保 shape/meta 嚴格控制
def save_nifti(data, filename, output_dir):
    from monai.data.nifti_writer import write_nifti
    
    pred_data = data["pred"]
    meta = data["pred_meta_dict"]
    
    # 確保 affine 矩陣正確
    meta_affine = meta["affine"][0] if isinstance(meta["affine"], list) else meta["affine"]
    if meta_affine.ndim != 2 or meta_affine.shape != (4, 4):
        meta_affine = np.eye(4) # Fallback to identity matrix
        
    # 確保 pred_data 是 3D (D, H, W) 且 dtype=uint8
    pred_3d = pred_data.squeeze().astype(np.uint8)
    
    # 確保最終 shape 對齊 (這是為了解決你之前的 mismatch 問題)
    orig_shape_meta = np.array(meta["spatial_shape"]).flatten()
    if orig_shape_meta.size == 4:
        orig_shape = orig_shape_meta[1:] # 忽略 Channel 維度 C, D, H, W
    elif orig_shape_meta.size == 3:
        orig_shape = orig_shape_meta
    else:
        # 如果 Meta Data 異常，發出警告但不終止，使用當前 Shape
        orig_shape = pred_3d.shape
        print(f"⚠️ {filename} Meta Shape 異常 ({orig_shape_meta})，使用當前 pred shape: {orig_shape}")

    if pred_3d.shape != tuple(orig_shape):
        print(f"⚠️ Shape mismatch {filename}: pred={pred_3d.shape} vs orig={tuple(orig_shape)}")
        
        diff = np.array(orig_shape) - np.array(pred_3d.shape)
        pad_list = []
        crop_slices = [slice(None)] * 3
        
        for i in range(3):
            if diff[i] < 0: # Crop (pred_shape > orig_shape)
                crop_slices[i] = slice(0, orig_shape[i])
            elif diff[i] > 0: # Pad (pred_shape < orig_shape)
                pad_list.extend([0, diff[i]]) # (W, H, D)
        
        # 執行 Pad (Label 0)
        if pad_list:
            pred_3d = F.pad(torch.from_numpy(pred_3d).unsqueeze(0).unsqueeze(0), 
                            tuple(pad_list[::-1]), "constant", 0).squeeze().numpy()
        
        # 執行 Crop
        if any(s != slice(None) for s in crop_slices):
            pred_3d = pred_3d[crop_slices[0], crop_slices[1], crop_slices[2]]

        final_shape = pred_3d.shape
        print(f"🔧 {filename} 已修正為 {final_shape}")
        
        if final_shape != tuple(orig_shape):
            raise ValueError(f"最終 Shape 無法對齊 (Final {final_shape} vs Orig {tuple(orig_shape)})")

    
    # 寫入 NIfTI 檔案
    output_path = os.path.join(output_dir, os.path.basename(filename))
    write_nifti(
        data=pred_3d,
        file_name=output_path,
        affine=meta_affine,
        dtype=np.uint8
    )
    print(f"✅ {os.path.basename(filename)} 儲存完成，最終 shape={pred_3d.shape}")


# ================== TTA 推論核心邏輯 ==================
print(f"🧠 開始單模型 + TTA 推論 {len(test_files)} 筆測試影像...")

with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
    for batch in tqdm(test_loader):
        img = batch["image"].to(device) # (1, 1, D_norm, H_norm, W_norm)
        meta = copy.deepcopy(batch["image_meta_dict"])
        fname = os.path.basename(str(meta.get("filename_or_obj")[0]))
        all_preds_prob = [] # 儲存 Softmax 機率結果

        # ---- TTA + Inference (Softmax 平均) ----
        for aug_img, inv_fn in tta_flips(img):
            # 1. Inference 輸出 logits
            logits = sliding_window_inference(aug_img, roi_size, sw_batch_size, model, overlap=overlap, mode="gaussian")
            
            # 2. TTA 反轉 (Logits 空間)
            logits_inv = inv_fn(logits)
            
            # 3. Softmax 轉為機率圖
            out_prob = F.softmax(logits_inv, dim=1)
            all_preds_prob.append(out_prob.cpu())

        # Ensemble: Softmax 機率平均 (預期 shape: 1, C, D_norm, H_norm, W_norm)
        avg_prob = torch.mean(torch.stack(all_preds_prob), dim=0) 

        # 4. Argmax 轉為 Label Map (4D: 1, D_norm, H_norm, W_norm)
        pred_label_map_4d = torch.argmax(avg_prob, dim=1, keepdim=False).to(torch.long)
        
        # 5. 準備 Invertd
        # 為了 Invertd，需要 5D (1, 1, D_norm, H_norm, W_norm) MetaTensor
        pred_label_map_5d = pred_label_map_4d.unsqueeze(1) 
        
        # 6. 包裝 MetaTensor
        single = decollate_batch(batch)[0]
        single["pred"] = MetaTensor(pred_label_map_5d.cpu(), meta=meta)
        single["pred_meta_dict"] = single["image_meta_dict"]

        # 7. Invertd 還原原始 spacing / shape (到 numpy 空間)
        single = inverter(single)
        
        # 8. Shape 檢查與修正 (使用強化的 save 函數)
        save_nifti(single, fname, output_dir)


# ================== 打包 zip ==================
print("\n" + "="*50)
zip_path = os.path.join(os.path.dirname(output_dir), zip_name)
print(f"📦 開始打包至 ZIP：{zip_path}")
try:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 遞迴搜尋所有 .nii.gz
        for f in sorted(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True)):
            zipf.write(f, os.path.basename(f))

    print(f"✅ 已建立上傳檔案: {zip_path}")
    print("🎯 這份 zip 可直接上傳至 AI CUP Leaderboard 評分系統")
except Exception as e:
    print(f"❌ ZIP 打包失敗: {e}")