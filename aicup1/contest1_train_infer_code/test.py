# -*- coding: utf-8 -*-
import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss
from torch.cuda.amp import autocast, GradScaler

def main():
    # ---------------- 基本設定 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 模擬跟訓練一致的設定
    roi_size = (128, 128, 96)
    num_classes = 4
    feature_size = 48

    # ---------------- 模型 ----------------
    model = SwinUNETR(
        img_size=roi_size,
        in_channels=1,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=False
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    scaler = GradScaler(enabled=True)

    # ---------------- 建立隨機輸入資料 ----------------
    B = 1
    img = torch.randn(B, 1, *roi_size, device=device)
    lab = torch.randint(0, num_classes, (B, 1, *roi_size), device=device).long()

    # ---------------- 建立 DiceFocalLoss ----------------
    alpha_tensor = torch.tensor([0.05, 0.25, 0.35, 0.35], device=device)
    lam_focal = 0.3

    loss_fn = DiceFocalLoss(
        to_onehot_y=True,  # label 自動 one-hot
        softmax=True,      # output 走 softmax
        lambda_dice=0.7,   # Dice 權重
        lambda_focal=lam_focal,  # Focal 權重
        gamma=1.5,         # Focal gamma
    )

    # ---------------- Forward 測試 ----------------
    print("🚀 開始測試 DiceFocalLoss forward/backward ...")
    with autocast(enabled=True):
        out = model(img)
        loss = loss_fn(out, lab)

    print(f"✅ Loss 正常計算: {loss.item():.6f}")
    print(f"   輸入影像形狀: {tuple(img.shape)}")
    print(f"   模型輸出形狀: {tuple(out.shape)}")
    print(f"   標籤 shape: {tuple(lab.shape)} | dtype={lab.dtype}")

    # ---------------- Backward 測試 ----------------
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print("✅ Backward / Optimizer step 完成！")

    print("🎉 測試成功：DiceFocalLoss 在 MONAI 1.2.0 下運作正常！")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
