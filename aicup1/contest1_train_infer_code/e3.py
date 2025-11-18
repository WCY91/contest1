# -*- coding: utf-8 -*-
"""
✅ Final Inference Script for AI CUP Cardiac Segmentation - Dual Model Weighted Soft Ensemble (Optimized)
- 實現在原始空間上對兩個模型的預測進行 Softmax 加權平均
- 整合 TTA (翻轉增強)
- 整合後處理 (LCC, 小物體移除)
"""

import os, glob, zipfile, sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR, SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, AsDiscrete, Invertd, SaveImaged
)
from monai.inferers import sliding_window_inference
from monai.utils import PostFix

# === 引入 Scipy/Skimage 進行後處理 (如果沒有安裝，請使用 pip install scipy scikit-image) ===
try:
    from scipy.ndimage import gaussian_filter, label
    from skimage.morphology import remove_small_objects
except ImportError:
    print("⚠️ 警告：缺少 Scipy 或 Scikit-Image。Gaussian Smoothing 和 Post-processing 將被跳過。")
    gaussian_filter = None
    label = None
    remove_small_objects = None
# =========================================================================================


# ================== 核心開關與設定 ==================
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# ⚠️ 請自行修改為您的模型與資料路徑
swinunetr_model_path = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\exps\unetrpp\chgh\tune_results\AICUP_training\best_model_finetune_v2.pth"
segresnet_model_path = r"C:\Users\aclab_public\Downloads\best_model_segresnet.pth"
test_dir    = r"C:\Users\aclab_public\Downloads\aicup_result"
output_dir = rf"C:\Users\aclab_public\Downloads\aicup_pred_weighted_soft_ensemble_{timestamp}"
zip_name    = f"result_weighted_soft_ensemble_{timestamp}.zip"

# --- 🎯 融合/增強開關 ---
ENABLE_TTA = True               # 方向二：是否啟用 TTA (RL, AP, SI 翻轉)
GAUSSIAN_SIGMA = 1.0            # 方向五：Logits Smoothing (0.0 禁用)
ENABLE_POST_PROCESSING = True   # 方向三：是否啟用後處理 (LCC + 小物體移除)
CRF_ENABLE = False              # 方向六：3D CRF (目前未實作，僅為開關)

# --- 🎯 權重設定 (Soft Ensemble) ---
WEIGHT_SWINUNETR = 1
WEIGHT_SEGRESNET = 2

# ================== 模型共同設定 ==================
num_classes = 4
sw_batch_size = 1
overlap = 0.25

# ================== 模型各自設定 ==================
swinunetr_cfg = {"spacing": (0.7, 0.7, 0.7), "roi_size": (128, 128, 96), "a_min": -75, "a_max": 450}
segresnet_cfg = {"spacing": (0.7, 0.7, 0.8), "roi_size": (128, 128, 96), "a_min": -75, "a_max": 450}

# ====== 裝置與初始化 ======
has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

def set_determinism(seed: int = 2025):
    import random
    import numpy as _np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    _np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def must_exist(path: str, kind: str):
    if not os.path.exists(path):
        print(f"❌ {kind} 不存在：{path}")
        sys.exit(1)

# ================== 輔助函數：模型載入與推論 ==================
def load_swinunetr_model(model_path: str, device: torch.device) -> SwinUNETR:
    model = SwinUNETR(img_size=swinunetr_cfg["roi_size"], in_channels=1, out_channels=num_classes, feature_size=48, use_checkpoint=False).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def load_segresnet_model(model_path: str, device: torch.device) -> SegResNet:
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=num_classes,
        init_filters=32, blocks_down=[1,2,2,4], blocks_up=[1,1,1], dropout_prob=0.2
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def get_transforms(cfg: Dict[str, Any], keys: List[str]) -> Compose:
    return Compose([
        LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=cfg["spacing"], mode="bilinear"),
        ScaleIntensityRanged(keys=keys, a_min=cfg["a_min"], a_max=cfg["a_max"], b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys, track_meta=True),
    ])

# 翻轉軸向定義 (RL=0, AP=1, SI=2)
FLIP_AXES = [(0,), (1,), (2,)] # 僅考慮單軸翻轉

@torch.inference_mode()
def inference_and_invert(
    model: torch.nn.Module, cfg: Dict[str, Any], data_loader: DataLoader, 
    transforms: Compose, pred_key: str
) -> Dict[str, np.ndarray]:
    """執行推論 (含 TTA/Smoothing)，Invert 回原始空間，並回傳 Softmax 概率圖 (C, D, H_orig, W_orig)"""
    inverted_probs: Dict[str, np.ndarray] = {}
    use_amp = has_cuda
    autocast_dtype = "cuda" if use_amp else "cpu"

    pbar = tqdm(data_loader, desc=f"🧠 推論 {model.__class__.__name__}", leave=False)
    for batch in pbar:
        # 準備 TTA 累積變數
        accumulated_softmax = None
        tta_count = 0
        
        # 取得原始影像資訊 (meta)
        single_original = decollate_batch(batch)[0]
        original_filename = os.path.basename(single_original["image_meta_dict"]["filename_or_obj"])
        
        # --------------------- TTA 迴圈 ---------------------
        tta_list = [()] # 初始包含原圖 (無翻轉)
        if ENABLE_TTA:
            tta_list.extend(FLIP_AXES)

        for axes in tta_list:
            img = batch["image"].to(device, non_blocking=True)
            
            # 1. 應用翻轉 (Augmentation)
            if axes:
                img = torch.flip(img, dims=axes)

            # 2. 進行推論
            with torch.amp.autocast(autocast_dtype, enabled=use_amp):
                logits = sliding_window_inference(
                    img, cfg["roi_size"], sw_batch_size, model,
                    overlap=overlap, mode="gaussian"
                )
            
            # 3. Gaussian Smoothing (方向五)
            if GAUSSIAN_SIGMA > 0 and gaussian_filter is not None:
                logits_np = logits.squeeze(0).cpu().numpy()
                # 對 logits (C, D, H, W) 每個類別的概率圖進行高斯平滑
                for c in range(logits_np.shape[0]):
                    logits_np[c] = gaussian_filter(logits_np[c], sigma=GAUSSIAN_SIGMA, order=0)
                logits = torch.from_numpy(logits_np).unsqueeze(0).to(device)


            # 4. 轉換為 Softmax 概率
            softmax_prob = F.softmax(logits, dim=1).cpu().squeeze(0) # (C, D, H, W)
            
            # 5. 反向翻轉 (De-Augmentation)
            if axes:
                softmax_prob = torch.flip(softmax_prob, dims=axes)

            # 6. 累積
            if accumulated_softmax is None:
                accumulated_softmax = softmax_prob
            else:
                accumulated_softmax += softmax_prob
            tta_count += 1
            
        # 7. TTA 平均 Softmax (在原圖轉換空間)
        avg_softmax = accumulated_softmax / tta_count
        
        # --------------------- Invert 還原 ---------------------
        inverter = Invertd(
            keys=pred_key, transform=transforms, orig_keys="image",
            meta_keys=f"{pred_key}_meta_dict", orig_meta_keys="image_meta_dict",
            nearest_interp=False, to_tensor=False, # Softmax 概率使用線性/雙線性插值 (False)
        )
        
        # 將 Softmax 概率張量作為預測結果
        single_original[pred_key] = avg_softmax 
        single_original[f"{pred_key}_meta_dict"] = single_original["image_meta_dict"]

        # 還原回原始 spacing & shape
        single_original = inverter(single_original)
        
        # 儲存還原後的 Softmax 概率 (C, D, H_orig, W_orig)
        arr = single_original[pred_key]
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        
        inverted_probs[original_filename] = arr.astype(np.float32, copy=False)

    return inverted_probs


# ================== 後處理輔助函數 ==================
def apply_post_processing(mask_in: np.ndarray, num_classes: int, threshold: int = 1000) -> np.ndarray:
    """方向三：對 Argmax 遮罩應用 LCC 和小物體移除"""
    if label is None or remove_small_objects is None:
        print("⚠️ 警告：後處理未執行，請確保 Scipy 和 Scikit-Image 已安裝。")
        return mask_in

    mask_out = mask_in.copy()
    
    # 針對除了背景 (Class 0) 以外的每個類別進行處理
    for c in range(1, num_classes):
        binary_mask = (mask_in == c)
        
        if not np.any(binary_mask):
            continue
            
        # 1. 移除小於閾值的小物體/噪點
        # True: 移除小於閾值的連通區塊
        cleaned_mask = remove_small_objects(binary_mask, min_size=threshold, connectivity=1)
        
        # 2. 保留最大連通區 (Largest Connected Component, LCC)
        labeled_array, num_features = label(cleaned_mask)
        
        if num_features > 0:
            # 找出最大連通區的標籤
            component_sizes = np.bincount(labeled_array.ravel())
            # 跳過背景標籤 0
            largest_component_label = np.argmax(component_sizes[1:]) + 1 
            
            # 建立 LCC 遮罩
            lcc_mask = (labeled_array == largest_component_label)
        else:
            lcc_mask = cleaned_mask # 如果清理後沒有連通區，則用清理後的結果 (應為全 False)
        
        # 更新輸出遮罩
        mask_out[lcc_mask] = c
        # 確保非 LCC 部分被背景或後續類別覆蓋，這裡簡單確保非 LCC 且原為 C 的部分設回 0
        mask_out[~lcc_mask & (mask_in == c)] = 0 
        
    return mask_out


# ================== 核心執行區塊 ==================
if __name__ == '__main__':
    
    # ====== 執行初始化 ======
    set_determinism()
    print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if has_cuda else 'CPU'}")
    print(f"🗳️ Ensemble 權重: SwinUNETR ({WEIGHT_SWINUNETR}), SegResNet ({WEIGHT_SEGRESNET})")
    print(f"✨ TTA: {'啟用' if ENABLE_TTA else '禁用'}, Smoothing: Sigma={GAUSSIAN_SIGMA}, Post-processing: {'啟用' if ENABLE_POST_PROCESSING else '禁用'}")

    os.makedirs(output_dir, exist_ok=True)
    must_exist(swinunetr_model_path, "SwinUNETR 權重")
    must_exist(segresnet_model_path, "SegResNet 權重")
    must_exist(test_dir, "測試資料夾")

    # ====== 推論流程 ======
    
    # 1) 載入模型
    print("-" * 30)
    swinunetr_model = load_swinunetr_model(swinunetr_model_path, device)
    segresnet_model = load_segresnet_model(segresnet_model_path, device)

    # 2) 準備測試資料
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
    if len(test_files) == 0:
        print(f"❌ 在 {test_dir} 找不到 *.nii.gz")
        sys.exit(1)
    data_dicts = [{"image": f} for f in test_files]

    num_workers = max(os.cpu_count() // 2, 0) # 設置為 0 以避免 Windows 多進程錯誤
    # num_workers = 0 # ⚠️ 如果持續出錯，請使用這行
    
    loader_cfg = dict(batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=has_cuda)

    # --- SwinUNETR 推論 (輸出 Softmax 概率) ---
    swinunetr_transforms = get_transforms(swinunetr_cfg, keys=["image"])
    swinunetr_ds = Dataset(data=data_dicts, transform=swinunetr_transforms)
    swinunetr_loader = DataLoader(swinunetr_ds, **loader_cfg)

    swinunetr_preds_prob = inference_and_invert(
        swinunetr_model, swinunetr_cfg, swinunetr_loader, swinunetr_transforms, pred_key="pred_swin"
    )

    # --- SegResNet 推論 (輸出 Softmax 概率) ---
    segresnet_transforms = get_transforms(segresnet_cfg, keys=["image"])
    segresnet_ds = Dataset(data=data_dicts, transform=segresnet_transforms)
    segresnet_loader = DataLoader(segresnet_ds, **loader_cfg)

    segresnet_preds_prob = inference_and_invert(
        segresnet_model, segresnet_cfg, segresnet_loader, segresnet_transforms, pred_key="pred_segres"
    )

    # ================== 加權 Soft Ensemble 與儲存 ==================
    print("🗳️ 進行 Softmax 加權平均 Ensemble...")

    # Ensemble 用的 DataLoader (僅用於迭代檔名和 meta data)
    final_transforms = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])
    final_ds = Dataset(data=data_dicts, transform=final_transforms)
    final_loader = DataLoader(final_ds, **loader_cfg)

    save_pred = SaveImaged(
        keys="ensemble_pred", meta_keys="image_meta_dict", output_dir=output_dir,
        output_postfix="", output_dtype=np.uint8, resample=False, print_log=False,
    )

    saved_count = 0
    pbar_save = tqdm(final_loader, desc="💾 儲存加權 Ensemble 結果")
    for batch in pbar_save:
        single = decollate_batch(batch)[0]
        original_filename = os.path.basename(single["image_meta_dict"]["filename_or_obj"])

        prob_swin  = swinunetr_preds_prob.get(original_filename)
        prob_segres= segresnet_preds_prob.get(original_filename)

        if prob_swin is None or prob_segres is None:
            raise KeyError(f"找不到 {original_filename} 的其中一個模型概率預測")
        
        # 1. Softmax 加權平均 (方向一)
        # Final_Prob = (Prob_Swin * W_Swin + Prob_SegRes * W_SegRes) / (W_Swin + W_SegRes)
        ensemble_prob = (
            (prob_swin * WEIGHT_SWINUNETR) + (prob_segres * WEIGHT_SEGRESNET)
        ) / (WEIGHT_SWINUNETR + WEIGHT_SEGRESNET)
        
        # 2. Argmax 轉換為離散遮罩
        ensemble_pred_np = np.argmax(ensemble_prob, axis=0).astype(np.uint8, copy=False)
        
        # 3. 後處理 (方向三)
        if ENABLE_POST_PROCESSING and label is not None:
            # 針對每個類別執行 LCC 和小物體移除
            ensemble_pred_np = apply_post_processing(ensemble_pred_np, num_classes=num_classes, threshold=1000)

        # 4. 儲存
        single["ensemble_pred"] = ensemble_pred_np
        single["image_meta_dict"][PostFix.meta_key("filename_or_obj")] = original_filename
        save_pred(single)
        saved_count += 1

    print(f"✅ 推論與加權 Soft Ensemble 完成，檔案數：{saved_count}，輸出資料夾：{output_dir}")

    # ================== 打包 zip ==================
    zip_path = os.path.join(os.path.dirname(output_dir), zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        nii_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True))
        for f in nii_files:
            zipf.write(f, os.path.basename(f))
    print(f"📦 已建立上傳檔案: {zip_path}（共 {len(nii_files)} 件）")
    print("🎯 這份 zip 可直接上傳至 AI CUP Leaderboard 評分系統")

    # 清理 GPU 記憶體（可選）
    if has_cuda:
        torch.cuda.empty_cache()