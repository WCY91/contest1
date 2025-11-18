# -*- coding: utf-8 -*-
"""
🎯 檔案重命名腳本 (只執行改名)

功能:
1. 讀取指定目錄中的所有 .nii.gz 檔案。
2. 將檔案名稱從 'patient00XX.nii_pred.nii.gz' 重命名為 'patient00XX.nii.gz'。
"""
import os
import glob
from tqdm import tqdm

# ================== 設定區 (請務必修改此處) ==================

# ❗ 1. 替換成您**已跑完推論結果**的資料夾路徑
# 範例: C:\Users\aclab_public\Downloads\aicup_pred_output_SwinUNETR_TTA_20251104_070000
source_dir = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\result_ensemble_20251103_211612" 

# 2. 根據您的檔案名，要移除的特定後綴
# 您的檔案格式是 patient0051.nii_pred.nii.gz，我們要移除 .nii_pred.nii.gz 中的 '_pred.nii' 部分
SUFFIX_TO_REMOVE = ".nii_pred.nii.gz"
SUFFIX_TO_KEEP = ".nii.gz"

# ================== 核心功能：重命名 ==================

def rename_files():
    if not os.path.exists(source_dir):
        print(f"❌ 錯誤: 原始目錄 {source_dir} 不存在。請檢查路徑。")
        return

    # 搜尋所有 .nii.gz 檔案
    files_to_rename = sorted(glob.glob(os.path.join(source_dir, "*.nii.gz")))
    
    if not files_to_rename:
        print("❌ 錯誤: 在目標目錄中找不到任何 .nii.gz 檔案。")
        return

    print(f"📁 找到 {len(files_to_rename)} 個檔案，開始重命名...")
    
    renamed_count = 0
    try:
        for full_path in tqdm(files_to_rename, desc="重命名進度"):
            original_filename = os.path.basename(full_path)
            
            # 檢查是否包含需要移除的特定後綴
            if original_filename.endswith(SUFFIX_TO_REMOVE):
                
                # 移除 SUFFIX_TO_REMOVE 並換成 SUFFIX_TO_KEEP
                # 例如: 'patient0051.nii_pred.nii.gz' -> 'patient0051.nii.gz'
                new_filename = original_filename.replace(SUFFIX_TO_REMOVE, SUFFIX_TO_KEEP)
                
                new_full_path = os.path.join(source_dir, new_filename)
                
                # 執行重命名
                os.rename(full_path, new_full_path)
                renamed_count += 1
            # else:
            #     print(f"ℹ️ 跳過 {original_filename}，無需重命名。")

        print("\n" + "="*50)
        print(f"✅ 重命名完成! 成功處理 {renamed_count} 個檔案。")
        print(f"檔案已在原始目錄 {source_dir} 中更新名稱。")

    except Exception as e:
        print(f"\n❌ 重命名失敗: {e}")
        print("請檢查檔案權限。")

# ================== 執行區 ==================
if __name__ == "__main__":
    rename_files()