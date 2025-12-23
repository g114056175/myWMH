"""
7通道v2數據準備腳本 - 使用Butterworth高通專攻小病灶
配置: FLAIR[t-1,t,t+1] + T1 + Butterworth(15) + HighPass(σ=2) + Top-hat
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter, grey_opening
from tqdm import tqdm

# ============ 特徵計算函數 ============

def normalize_image(img):
    """標準化到[0, 1]"""
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)

def butterworth_highpass(image, cutoff=15, order=2):
    """
    Butterworth高通濾波 - 專攻小病灶
    
    Args:
        image: 2D array
        cutoff: 截止頻率（pixels）- 15適合小病灶（5-15 pixel）
        order: 階數，2為推薦值（平滑且有效）
    """
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    
    # 距離矩陣
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    
    # Butterworth濾波器
    H = 1 / (1 + (cutoff / (D + 1e-6))**(2*order))
    
    # FFT處理
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    
    return normalize_image(result)

def simple_highpass(image, sigma=2.0):
    """簡單高通濾波（已驗證有效，22.92%）"""
    blurred = gaussian_filter(image, sigma=sigma)
    highpass = image - blurred
    return normalize_image(highpass)

def tophat_transform(image, size=15):
    """Top-hat轉換 - 檢測小亮點"""
    opened = grey_opening(image, size=(size, size))
    tophat = image - opened
    return normalize_image(tophat)

# ============ 數據路徑 ============

data_root = Path("d:/VSCode/AIOT_E1/data/wmh")
output_root = Path("d:/VSCode/AIOT_E1/nnUnet/nnUNet_raw/Dataset004_WMH_7CH_v2")

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
    
    # 標準化整個volume
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
        # Channel 4: Butterworth高通 (專攻小病灶)
        butterworth_slice = butterworth_highpass(flair_slice, cutoff=15, order=2)
        
        # Channel 5: 簡單高通 σ=2 (已驗證)
        highpass_slice = simple_highpass(flair_slice, sigma=2.0)
        
        # Channel 6: Top-hat (小亮點)
        tophat_slice = tophat_transform(flair_slice, size=15)
        
        # === 保存7個通道 ===
        slice_name = f"{case_id}_slice{z:04d}"
        
        channels = [
            flair_prev,          # Channel 0
            flair_slice,         # Channel 1
            flair_next,          # Channel 2
            t1_slice,            # Channel 3
            butterworth_slice,   # Channel 4 - NEW!
            highpass_slice,      # Channel 5
            tophat_slice         # Channel 6
        ]
        
        for ch_idx, channel_data in enumerate(channels):
            img_2d = nib.Nifti1Image(channel_data.astype(np.float32), flair_nii.affine)
            img_path = output_dir / f"{slice_name}_{ch_idx:04d}.nii.gz"
            nib.save(img_2d, img_path)
        
        # 保存label (處理label=2)
        wmh_slice_clean = wmh_slice.copy()
        wmh_slice_clean[wmh_slice == 2] = 0
        
        label_2d = nib.Nifti1Image(wmh_slice_clean.astype(np.uint8), wmh_nii.affine)
        label_path = label_output_dir / f"{slice_name}.nii.gz"
        nib.save(label_2d, label_path)
    
    return num_slices

# ============ 主處理流程 ============

print("=" * 80)
print("7通道v2數據準備 - Dataset004_WMH_7CH_v2 (Butterworth優化)")
print("=" * 80)
print("通道配置:")
print("  0: FLAIR[t-1]")
print("  1: FLAIR[t]")
print("  2: FLAIR[t+1]")
print("  3: T1")
print("  4: Butterworth高通 (cutoff=15, order=2) ⭐ 專攻小病灶")
print("  5: 簡單高通 σ=2 (已驗證22.92%)")
print("  6: Top-hat (小亮點檢測)")
print("=" * 80)

# 處理訓練集
print("\n處理訓練集...")
train_dir = data_root / "training"
train_cases = []

for site_dir in sorted(train_dir.iterdir()):
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
        "4": "Butterworth_cutoff15",
        "5": "HighPass_sigma2",
        "6": "Top-hat"
    },
    "labels": {
        "background": 0,
        "WMH": 1
    },
    "numTraining": len(list((output_root / "labelsTr").glob("*.nii.gz"))),
    "file_ending": ".nii.gz",
    "description": "WMH 7CH v2 - Butterworth(15) targeting small lesions",
    "name": "WMH_7CH_v2",
    "reference": "Optimized for small lesion detection",
    "release": "2.0"
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
print("\n改進重點:")
print("  ⭐ Butterworth(cutoff=15): 專門檢測5-15 pixel的小病灶邊緣")
print("  ⭐ 精確頻率控制，無Gibbs振鈴")
print("  ⭐ 與簡單高通σ=2互補（中等病灶）")
print("=" * 80)
print("\n下一步:")
print("1. cd nnUnet")
print("2. .\\setup_env.bat")
print("3. nnUNetv2_plan_and_preprocess -d 004 --verify_dataset_integrity")
print("4. nnUNetv2_train 004 2d 0 --npz")
print("=" * 80)
