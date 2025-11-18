import os
import gzip
import shutil
import glob
import zipfile
import tarfile

# === 參數設定 ===
outer_zip = r"C:\Users\aclab_public\Downloads\aaa\result_weighted_ensemble_20251109_1746.zip"
extract_dir = r"C:\Users\aclab_public\Downloads\aaa"
final_zip = r"C:\Users\aclab_public\Downloads\result_fixed.zip"
start_id = 51
os.makedirs(extract_dir, exist_ok=True)

# === Step 1: 解壓外層 ZIP ===
print(f"🔍 解壓外層 ZIP: {outer_zip}")
with zipfile.ZipFile(outer_zip, "r") as outer:
    outer.extractall(extract_dir)

# === Step 2: 找出所有 .gz (其實是資料夾壓縮檔) ===
inner_gz = sorted(glob.glob(os.path.join(extract_dir, "*.gz")))
print(f"📦 發現 {len(inner_gz)} 個內層 .gz")

converted = []
for idx, gzpath in enumerate(inner_gz, start=start_id):
    # temp_dir = os.path.join(extract_dir, f"temp_{idx}")
    # os.makedirs(temp_dir, exist_ok=True)

    # 嘗試以 gzip 解壓
    try:
        raw_name = os.path.basename(gzpath).replace(".gz", "")
        out_path = os.path.join(raw_name)

        with gzip.open(gzpath, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        print(f"⚠️ 無法解壓 {gzpath}: {e}")
        continue
import re
def natural_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]
nii_list = sorted(glob.glob(os.path.join(".", "*.nii")), key=natural_key)

# 依序改名
for i, src in enumerate(nii_list):
    num = i + 51
    if i == 49 : new_name = f"patient0100.nii.gz"
    new_name = f"patient00{num}.nii" 
    dst = os.path.join(extract_dir, new_name)
    os.rename(src, dst)
    print(f"✅ 已改名: {src} → {dst}")


# for i , src in enumerate(nii_list):
#     # 重新用 gzip 壓成標準 .nii.gzc
#     num = i + 51
#     if i == 49 : new_name = f"patient0100.nii.gz"
#     new_name = f"patient00{num}.nii.gz" 

# converted.append(dst)
# print(f"✅ {os.path.basename(gzpath)} → {new_name}")

# print(f"✅ 共成功轉換 {len(converted)} 個 .nii → .nii.gz")

# # === Step 3: 打包成單層 ZIP ===
# with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as z:
#     for f in sorted(converted):
#         z.write(f, os.path.basename(f))
# print(f"🎯 已建立單層 ZIP：{final_zip}")
# print("✅ 結構為單層，可直接上傳 AI CUP Leaderboard")

# # === Step 4: 清理暫存 ===
# try:
#     shutil.rmtree(extract_dir)
#     print(f"🧹 已刪除暫存資料夾：{extract_dir}")
# except Exception as e:
#     print(f"⚠️ 刪除暫存資料夾時發生錯誤：{e}")
