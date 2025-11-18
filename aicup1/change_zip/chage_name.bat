@echo off
chcp 65001 >nul
title AI CUP Cardiac Segmentation - ZIP 修正工具
echo ===============================================
echo 🔧 AI CUP ZIP 修正工具
echo ===============================================

:: === 路徑設定 ===
set "outer_zip=C:\Users\aclab_public\Downloads\result_seg_20251102_0015.zip"
set "extract_dir=C:\Users\0524e\Downloads\aaa"
set "final_zip=C:\Users\0524e\Downloads\result_fixed.zip"
set "start_id=51"

:: === 建立暫存資料夾 ===
if not exist "%extract_dir%" mkdir "%extract_dir%"

echo 🧩 開始處理，請稍候...
python - <<END
import os, gzip, shutil, glob, zipfile, re

outer_zip = r"%outer_zip%"
extract_dir = r"%extract_dir%"
final_zip = r"%final_zip%"
start_id = int("%start_id%")

os.makedirs(extract_dir, exist_ok=True)
print(f"🔍 解壓外層 ZIP: {outer_zip}")
with zipfile.ZipFile(outer_zip, "r") as outer:
    outer.extractall(extract_dir)

inner_gz = sorted(glob.glob(os.path.join(extract_dir, "*.gz")))
print(f"📦 發現 {len(inner_gz)} 個內層 .gz")

for idx, gzpath in enumerate(inner_gz, start=start_id):
    try:
        raw_name = os.path.basename(gzpath).replace(".gz", "")
        out_path = os.path.join(".", raw_name)
        with gzip.open(gzpath, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        print(f"⚠️ 無法解壓 {gzpath}: {e}")

def natural_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]

nii_list = sorted(glob.glob(os.path.join(".", "*.nii")), key=natural_key)
for i, src in enumerate(nii_list):
    num = i + start_id
    if i == 49:
        new_name = "patient0100.nii.gz"
    else:
        new_name = f"patient00{num}.nii"
    dst = os.path.join(extract_dir, new_name)
    os.rename(src, dst)
    print(f"✅ 已改名: {src} → {dst}")

print("✅ 全部完成！")
END

echo ===============================================
echo 🎯 已完成 ZIP 修正，請檢查輸出資料夾：
echo %extract_dir%
echo ===============================================
pause
