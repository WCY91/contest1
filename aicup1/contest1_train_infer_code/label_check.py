import nibabel as nib, numpy as np, os

lbl_dir = r"C:\Users\aclab_public\Desktop\aicup1\CardiacSegV2\dataset\chgh\aicup_1\labelsTr"
for f in sorted(os.listdir(lbl_dir)):
    if f.endswith(".nii.gz"):
        lbl = nib.load(os.path.join(lbl_dir, f)).get_fdata()
        u = np.unique(lbl)
        print(f"{f:25s}  ->  {u}")
