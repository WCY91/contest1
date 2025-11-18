# -*- coding: utf-8 -*-
"""
✅ Final Inference Script for AI CUP Cardiac Segmentation - Dual Model Weighted Ensemble (Optimized)
- 在原始空間上對兩個模型的預測進行加權多數決 Ensemble
- 修正 Windows 多進程 RuntimeError: 引入 if __name__ == '__main__':
"""

import os, glob, zipfile, sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List
from monai.transforms import KeepLargestConnectedComponent
from monai.networks.utils import one_hot

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

# ================== 核心配置區塊 (可放在 if __name__ 之外) ==================

# ⚠️ 請自行修改為您的模型與資料路徑
swinunetr_model_path = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\exps\exps\unetrpp\chgh\tune_results\AICUP_training\best_model_finetune_v2.pth"
segresnet_model_path = r"C:\Users\aclab_public\Downloads\best_model_segresnet.pth"
test_dir    = r"C:\Users\aclab_public\Downloads\aicup_result"
output_dir = rf"C:\Users\aclab_public\Downloads\aicup_pred_weighted_ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
zip_name    = f"result_weighted_ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"

# ================== 模型共同設定 (可放在 if __name__ 之外) ==================
num_classes = 4
sw_batch_size = 1
overlap = 0.75
WEIGHT_SWINUNETR = 7
WEIGHT_SEGRESNET = 4

swinunetr_cfg = {
    "spacing": (0.7, 0.7, 0.7),
    "roi_size": (128, 128, 96),
    "a_min": -75, "a_max": 450,
}
segresnet_cfg = {
    "spacing": (0.7, 0.7,0.8),  # ⚠️ 注意 spacing 不同
    "roi_size": (128, 128, 96),
    "a_min":-75, "a_max": 450,
}

# ====== 裝置與輔助函數 (可放在 if __name__ 之外) ======
has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

def set_determinism(seed: int = 2025):
    # ... (保持原來的 set_determinism 函數)
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

# ... (保持 load_swinunetr_model, load_segresnet_model, get_transforms, inference_and_invert, weighted_majority_vote_vectorized 函數定義不變)

def load_swinunetr_model(model_path: str, device: torch.device) -> SwinUNETR:
    model = SwinUNETR(
        img_size=swinunetr_cfg["roi_size"], in_channels=1, out_channels=num_classes,
        feature_size=48, use_checkpoint=False
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"💾 已載入 SwinUNETR 權重: {os.path.basename(model_path)}")
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
    print(f"💾 已載入 SegResNet 權重: {os.path.basename(model_path)}")
    return model

def get_transforms(cfg: Dict[str, Any], keys: List[str]) -> Compose:
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=cfg["spacing"], mode="bilinear"),
        ScaleIntensityRanged(
            keys=keys, a_min=cfg["a_min"], a_max=cfg["a_max"],
            b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=keys, track_meta=True),
    ])

@torch.inference_mode()
def inference_and_invert(
    model: torch.nn.Module, cfg: Dict[str, Any], data_loader: DataLoader, 
    transforms: Compose, pred_key: str
) -> Dict[str, np.ndarray]:
    """單一模型推論+Invert回原空間，回傳 {filename: np.uint8 mask}"""
    inverted_preds: Dict[str, np.ndarray] = {}
    inverter = Invertd(
        keys=pred_key, transform=transforms, orig_keys="image",
        meta_keys=f"{pred_key}_meta_dict", orig_meta_keys="image_meta_dict",
        nearest_interp=True, to_tensor=False,
    )

    use_amp = has_cuda
    autocast_dtype = "cuda" if use_amp else "cpu"

    pbar = tqdm(data_loader, desc=f"🧠 推論 {model.__class__.__name__}", leave=False)
    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(autocast_dtype, enabled=use_amp):
            logits = sliding_window_inference(
                img, cfg["roi_size"], sw_batch_size, model,
                overlap=overlap, mode="gaussian"
            )
            # (B, C, H, W, D) -> (1, 4, H_s, W_s, D_s)

        # ========================== 
        # ⚠️ 這是關鍵修正！
        # ==========================
        # 不使用 AsDiscrete，而是明確地在 C 軸 (dim=1) 上執行 argmax
        # logits shape: (1, 4, H_s, W_s, D_s)
        pred_argmax_tensor = torch.argmax(logits, dim=1)  
        # pred_argmax_tensor shape: (1, H_s, W_s, D_s)
        
        # 移除 Batch 軸 (dim=0)，並轉換為 CPU 上的 uint8
        pred_argmax_3d = pred_argmax_tensor.to(dtype=torch.uint8).cpu().squeeze(0)
        # pred_argmax_3d shape: (H_s, W_s, D_s)

        # ⚠️ 關鍵修正：Invertd 需要一個 4D (C, H, W, D) 的輸入，
        # 因為順向 transform (Spacingd) 是在 EnsureChannelFirstd 之後應用的。
        pred_argmax_4d = pred_argmax_3d.unsqueeze(0) 
        # pred_argmax_4d shape: (1, H_s, W_s, D_s)
        # ==========================

        single = decollate_batch(batch)[0]
        original_filename = os.path.basename(single["image_meta_dict"]["filename_or_obj"])

        single[pred_key] = pred_argmax_4d # 放入 4D 陣列
        single[f"{pred_key}_meta_dict"] = single["image_meta_dict"]
        single = inverter(single) # Inverter 會將 3D 陣列還原回原始空間

        arr = single[pred_key] # arr 現在是 (H_orig, W_orig, D_orig)
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        inverted_preds[original_filename] = arr.astype(np.uint8, copy=False)

    return inverted_preds

def weighted_majority_vote_vectorized(preds_list: List[np.ndarray], weights: List[int], num_classes: int) -> np.ndarray:
    """
    使用向量化操作進行加權多數決投票。
    """
    assert len(preds_list) == len(weights) and len(preds_list) > 0
    shape = preds_list[0].shape
    for p in preds_list:
        if p.shape != shape:
            raise ValueError(f"加權投票前 shape 不一致：{[pp.shape for pp in preds_list]}")

    scores = np.zeros((num_classes, *shape), dtype=np.int16)
    for pred, w in zip(preds_list, weights):
        for c in range(num_classes):
            mask = (pred == c)
            scores[c][mask] += w
    ensemble = np.argmax(scores, axis=0).astype(np.uint8, copy=False)
    return ensemble


# ==============================================================================
# 核心執行區塊：必須包含在 if __name__ == '__main__': 內
# ==============================================================================
if __name__ == '__main__':
    # ====== 執行初始化 ======
    set_determinism()
    print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if has_cuda else 'CPU'}")
    print(f"🗳️ Ensemble 權重: SwinUNETR ({WEIGHT_SWINUNETR}), SegResNet ({WEIGHT_SEGRESNET})")
    os.makedirs(output_dir, exist_ok=True)
    
    # ====== 安全檢查 ======
    must_exist(swinunetr_model_path, "SwinUNETR 權重")
    must_exist(segresnet_model_path, "SegResNet 權重")
    must_exist(test_dir, "測試資料夾")

    # ====== 推論流程 ======
    
    # 1) 載入模型
    swinunetr_model = load_swinunetr_model(swinunetr_model_path, device)
    segresnet_model = load_segresnet_model(segresnet_model_path, device)

    # 2) 準備測試資料
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
    if len(test_files) == 0:
        print(f"❌ 在 {test_dir} 找不到 *.nii.gz")
        sys.exit(1)
    data_dicts = [{"image": f} for f in test_files]

    num_workers = max(os.cpu_count() // 2, 2)
    # ⚠️ 修正: 當 num_workers > 0 時，可能會遇到 RuntimeError。在 Windows 上，
    # 如果這個錯誤持續出現，建議將 num_workers 設置為 0 來避免多進程問題，
    # 犧牲 I/O 速度以換取運行穩定性。
    # num_workers = 0 # <-- 如果持續出錯，請使用這行
    
    loader_cfg = dict(batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=has_cuda)

    # --- SwinUNETR 推論 ---
    swinunetr_transforms = get_transforms(swinunetr_cfg, keys=["image"])
    swinunetr_ds = Dataset(data=data_dicts, transform=swinunetr_transforms)
    swinunetr_loader = DataLoader(swinunetr_ds, **loader_cfg)

    swinunetr_preds_inv = inference_and_invert(
        swinunetr_model, swinunetr_cfg, swinunetr_loader, swinunetr_transforms, pred_key="pred_swin"
    )

    # --- SegResNet 推論 ---
    segresnet_transforms = get_transforms(segresnet_cfg, keys=["image"])
    segresnet_ds = Dataset(data=data_dicts, transform=segresnet_transforms)
    segresnet_loader = DataLoader(segresnet_ds, **loader_cfg)

    segresnet_preds_inv = inference_and_invert(
        segresnet_model, segresnet_cfg, segresnet_loader, segresnet_transforms, pred_key="pred_segres"
    )

    # ================== 加權 Ensemble 與儲存 ==================
    print("🗳️ 進行加權多數決 Ensemble...")

    final_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    final_ds = Dataset(data=data_dicts, transform=final_transforms)
    final_loader = DataLoader(final_ds, **loader_cfg)

    save_pred = SaveImaged(
        keys="ensemble_pred", meta_keys="image_meta_dict", output_dir=output_dir,
        output_postfix="", output_dtype=np.uint8, resample=False, print_log=False,
    )

    saved_count = 0

    post_processor = KeepLargestConnectedComponent(
    applied_labels=[1, 2, 3], independent=True, connectivity=1
    )


    pbar_save = tqdm(final_loader, desc="💾 儲存加權 Ensemble 結果")
    for batch in pbar_save:
        single = decollate_batch(batch)[0]
        original_filename = os.path.basename(single["image_meta_dict"]["filename_or_obj"])

        pred_swin  = swinunetr_preds_inv.get(original_filename)
        pred_segres= segresnet_preds_inv.get(original_filename)
        print("swinunetr pred shape =", pred_swin.shape)
        print("segresnet pred shape =", pred_segres.shape)
        if pred_swin is None or pred_segres is None:
            raise KeyError(f"找不到 {original_filename} 的其中一個模型預測")
        
        if pred_swin.shape != pred_segres.shape:
            raise ValueError(f"{original_filename} 的還原後形狀不一致：swin {pred_swin.shape} vs segres {pred_segres.shape}")

        # 加權投票（向量化）
        ensemble_pred_np = weighted_majority_vote_vectorized(
            preds_list=[pred_swin, pred_segres],
            weights=[WEIGHT_SWINUNETR, WEIGHT_SEGRESNET],
            num_classes=num_classes
        )

        pred_tensor = torch.from_numpy(ensemble_pred_np).to(device)
        pred_one_hot = one_hot(pred_tensor.unsqueeze(0), num_classes=num_classes, dim=0) # (C, H, W, D)
        pred_one_hot_pp = post_processor(pred_one_hot)
        ensemble_pred_pp_np = torch.argmax(pred_one_hot_pp, dim=0).cpu().numpy()


        # 儲存
        #ensemble_pred_np = ensemble_pred_np[None, ...].astype(np.uint8) 
        ensemble_pred_3d = ensemble_pred_np.astype(np.uint8, copy=False)
        ensemble_pred_3d = ensemble_pred_pp_np.astype(np.uint8, copy=False)
        
        single["ensemble_pred"] =ensemble_pred_3d
        #single["image_meta_dict"][PostFix.meta_key("filename_or_obj")] = original_filename
        save_pred(single)
        saved_count += 1

    print(f"✅ 推論與加權 Ensemble 完成，檔案數：{saved_count}，輸出資料夾：{output_dir}")

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