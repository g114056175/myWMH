"""
7通道特徵工程數據準備腳本
配置: FLAIR[t-1,t,t+1] + T1 + CLAHE + Top-hat + HighPass(σ=2.0)
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter, grey_opening
import cv2
from tqdm import tqdm

# ============ 特徵計算函數 ============

def normalize_image(img):
    """標準化到[0, 1]"""
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)

def apply_clahe(image, clip_limit=2.0):
    """CLAHE對比度增強"""
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_uint8)
    return clahe_img.astype(np.float32) / 255.0

def tophat_transform(image, size=15):
    """Top-hat轉換 - 檢測亮點"""
    opened = grey_opening(image, size=(size, size))
    tophat = image - opened
    return normalize_image(tophat)

def highpass_filter(image, sigma=2.0):
    """高通濾波"""
    blurred = gaussian_filter(image, sigma=sigma)
    highpass = image - blurred
    return normalize_image(highpass)

# ============ 數據路徑 ============

data_root = Path("d:/VSCode/AIOT_E1/data/wmh")
output_root = Path("d:/VSCode/AIOT_E1/nnUnet/nnUNet_raw/Dataset003_WMH_7CH")

# 創建輸出目錄
(output_root / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_root / "labelsTr").mkdir(parents=True, exist_ok=True)
(output_root / "imagesTs").mkdir(parents=True, exist_ok=True)
(output_root / "labelsTs").mkdir(parents=True, exist_ok=True)

# ============ 處理函數 ============

def process_case(case_dir, case_id, output_dir, label_output_dir):
    """處理單個案例"""
    # 載入數據
    flair_path = case_dir / "pre" / "FLAIR.nii.gz"
    t1_path = case_dir / "pre" / "T1.nii.gz"
    wmh_path = case_dir / "wmh.nii.gz"
    
    if not all([flair_path.exists(), t1_path.exists(), wmh_path.exists()]):
        print(f"⚠️  跳過 {case_id}: 缺少文件")
        return 0
    
    flair_nii = nib.load(flair_path)
    t1_nii = nib.load(t1_path)
    wmh_nii = nib.load(wmh_path)
    
    flair_data = flair_nii.get_fdata()
    t1_data = t1_nii.get_fdata()
    wmh_data = wmh_nii.get_fdata()
    
    # 標準化整個volume (在提取slice之前)
    flair_norm = normalize_image(flair_data)
    t1_norm = normalize_image(t1_data)
    
    num_slices = flair_data.shape[2]
    
    # 逐slice處理
    for z in range(num_slices):
        # 當前slice
        flair_slice = flair_norm[:, :, z]
        t1_slice = t1_norm[:, :, z]
        wmh_slice = wmh_data[:, :, z]
        
        # 跳過空slice
        if np.sum(wmh_slice > 0) == 0 and np.random.random() > 0.1:
            continue
        
        # 時序slices (2.5D)
        flair_prev = flair_norm[:, :, max(0, z-1)]
        flair_next = flair_norm[:, :, min(num_slices-1, z+1)]
        
        # === 計算特徵通道 ===
        # Channel 4: CLAHE
        clahe_slice = apply_clahe(flair_slice, clip_limit=2.0)
        
        # Channel 5: Top-hat
        tophat_slice = tophat_transform(flair_slice, size=15)
        
        # Channel 6: HighPass σ=2.0
        highpass_slice = highpass_filter(flair_slice, sigma=2.0)
        
        # === 保存7個通道 ===
        slice_name = f"{case_id}_slice{z:04d}"
        
        channels = [
            flair_prev,      # Channel 0
            flair_slice,     # Channel 1
            flair_next,      # Channel 2
            t1_slice,        # Channel 3
            clahe_slice,     # Channel 4
            tophat_slice,    # Channel 5
            highpass_slice   # Channel 6
        ]
        
        for ch_idx, channel_data in enumerate(channels):
            img_2d = nib.Nifti1Image(channel_data.astype(np.float32), flair_nii.affine)
            img_path = output_dir / f"{slice_name}_{ch_idx:04d}.nii.gz"
            nib.save(img_2d, img_path)
        
        # 保存label (將label=2轉為0，nnU-Net只接受0和1)
        wmh_slice_clean = wmh_slice.copy()
        wmh_slice_clean[wmh_slice == 2] = 0
        
        label_2d = nib.Nifti1Image(wmh_slice_clean.astype(np.uint8), wmh_nii.affine)
        label_path = label_output_dir / f"{slice_name}.nii.gz"
        nib.save(label_2d, label_path)
    
    return num_slices

# ============ 主處理流程 ============

print("=" * 80)
print("7通道數據準備 - Dataset003_WMH_7CH")
print("=" * 80)
print("通道配置:")
print("  0: FLAIR[t-1]")
print("  1: FLAIR[t]")
print("  2: FLAIR[t+1]")
print("  3: T1")
print("  4: CLAHE (對比度增強)")
print("  5: Top-hat (小亮點檢測)")
print("  6: HighPass σ=2.0 (邊緣)")
print("=" * 80)

# 處理訓練集
print("\n處理訓練集...")
train_dir = data_root / "training"
train_cases = []

for site_dir in sorted(train_dir.iterdir()):
    if not site_dir.is_dir():
        continue
    
    # 檢查是否有scanner子目錄
    has_scanner = any(d.is_dir() and not d.name.replace('_','').isdigit() 
                     for d in site_dir.iterdir())
    
    if has_scanner:
        for scanner_dir in sorted(site_dir.iterdir()):
            if not scanner_dir.is_dir():
                continue
            for case_dir in sorted(scanner_dir.iterdir()):
                if case_dir.is_dir() and case_dir.name.replace('_','').isdigit():
                    case_id = f"{site_dir.name}_{scanner_dir.name}_{case_dir.name}"
                    train_cases.append((case_dir, case_id))
    else:
        for case_dir in sorted(site_dir.iterdir()):
            if case_dir.is_dir() and case_dir.name.replace('_','').isdigit():
                case_id = f"{site_dir.name}_{case_dir.name}"
                train_cases.append((case_dir, case_id))

total_train_slices = 0
for case_dir, case_id in tqdm(train_cases, desc="Training"):
    slices = process_case(case_dir, case_id, 
                         output_root / "imagesTr",
                         output_root / "labelsTr")
    total_train_slices += slices

# 處理測試集
print("\n處理測試集...")
test_dir = data_root / "test"
test_cases = []

for site_dir in sorted(test_dir.iterdir()):
    if not site_dir.is_dir():
        continue
    
    has_scanner = any(d.is_dir() and not d.name.replace('_','').isdigit() 
                     for d in site_dir.iterdir())
    
    if has_scanner:
        for scanner_dir in sorted(site_dir.iterdir()):
            if not scanner_dir.is_dir():
                continue
            for case_dir in sorted(scanner_dir.iterdir()):
                if case_dir.is_dir() and case_dir.name.replace('_','').isdigit():
                    case_id = f"{site_dir.name}_{scanner_dir.name}_{case_dir.name}"
                    test_cases.append((case_dir, case_id))
    else:
        for case_dir in sorted(site_dir.iterdir()):
            if case_dir.is_dir() and case_dir.name.replace('_','').isdigit():
                case_id = f"{site_dir.name}_{case_dir.name}"
                test_cases.append((case_dir, case_id))

total_test_slices = 0
for case_dir, case_id in tqdm(test_cases, desc="Test"):
    slices = process_case(case_dir, case_id,
                         output_root / "imagesTs",
                         output_root / "labelsTs")
    total_test_slices += slices

# ============ 創建dataset.json ============

dataset_info = {
    "channel_names": {
        "0": "FLAIR_t-1",
        "1": "FLAIR_t",
        "2": "FLAIR_t+1",
        "3": "T1",
        "4": "CLAHE",
        "5": "Top-hat",
        "6": "HighPass_sigma2"
    },
    "labels": {
        "background": 0,
        "WMH": 1
    },
    "numTraining": len(list((output_root / "labelsTr").glob("*.nii.gz"))),
    "file_ending": ".nii.gz",
    "description": "WMH Segmentation 7-Channel (FLAIR 2.5D + T1 + Feature Engineering)",
    "name": "WMH_7CH",
    "reference": "Based on quantitative feature analysis",
    "release": "1.0"
}

with open(output_root / "dataset.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

# ============ 總結 ============

print("\n" + "=" * 80)
print("✅ 數據準備完成！")
print("=" * 80)
print(f"訓練集: {len(train_cases)} cases, ~{total_train_slices} slices")
print(f"測試集: {len(test_cases)} cases, ~{total_test_slices} slices")
print(f"輸出目錄: {output_root}")
print(f"dataset.json已創建")
print("=" * 80)
print("\n下一步:")
print("1. cd nnUnet")
print("2. .\\setup_env.bat")
print("3. nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity")
print("4. nnUNetv2_train 003 2d 0 --npz")
print("=" * 80)
