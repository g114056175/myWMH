"""
Diagnose visualization issue - check raw data and preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from dataset import WMHDataset, normalize_slice, center_crop
import config

def main():
    print("="*60)
    print("Diagnosing Data Loading Issue")
    print("="*60)
    
    # 1. Load raw FLAIR data directly
    train_dir = Path(config.TRAIN_DIR)
    flair_files = list(train_dir.rglob('*/pre/FLAIR.nii.gz'))
    
    if len(flair_files) == 0:
        print("ERROR: No FLAIR files found!")
        return
    
    print(f"\nFound {len(flair_files)} FLAIR files")
    print(f"Loading first file: {flair_files[0]}")
    
    # Load raw FLAIR
    flair_raw = nib.load(flair_files[0]).get_fdata()
    print(f"\nRaw FLAIR shape: {flair_raw.shape}")
    print(f"Raw FLAIR range: [{flair_raw.min():.1f}, {flair_raw.max():.1f}]")
    
    # Pick middle slice
    mid_z = flair_raw.shape[2] // 2
    slice_raw = flair_raw[:, :, mid_z]
    
    # Apply preprocessing
    slice_normalized = normalize_slice(slice_raw)
    print(f"\nAfter normalize: range [{slice_normalized.min():.3f}, {slice_normalized.max():.3f}]")
    
    slice_cropped = center_crop(slice_normalized, config.TARGET_SIZE)
    print(f"After crop: shape {slice_cropped.shape}")
    
    # 2. Load from dataset
    print("\n" + "="*60)
    print("Loading from WMHDataset")
    print("="*60)
    
    dataset = WMHDataset(config.TRAIN_DIR, transform=None)
    image_tensor, mask_tensor = dataset[0]
    image_np = image_tensor.cpu().numpy()
    
    print(f"\nDataset output shape: {image_np.shape}")
    print(f"Channel 0 (CLAHE[t-1]) range: [{image_np[0].min():.3f}, {image_np[0].max():.3f}]")
    print(f"Channel 1 (CLAHE[t]) range: [{image_np[1].min():.3f}, {image_np[1].max():.3f}]")
    print(f"Channel 3 (T1) range: [{image_np[3].min():.3f}, {image_np[3].max():.3f}]")
    
    # 3. Visualize comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Raw data processing
    axes[0, 0].imshow(slice_raw, cmap='gray')
    axes[0, 0].set_title('1. Raw FLAIR slice\n(Original size)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(slice_normalized, cmap='gray')
    axes[0, 1].set_title('2. After normalize_slice\n(Percentile clipping)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(slice_cropped, cmap='gray')
    axes[0, 2].set_title(f'3. After center_crop\n(224x224)', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].axis('off')
    
    # Row 2: Dataset outputs
    axes[1, 0].imshow(image_np[0], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Dataset Ch0\nCLAHE[t-1]', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_np[1], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Dataset Ch1\nCLAHE[t]', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(image_np[2], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Dataset Ch2\nCLAHE[t+1]', fontsize=12)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(image_np[3], cmap='gray', vmin=0, vmax=1)
    axes[1, 3].set_title('Dataset Ch3\nT1', fontsize=12)
    axes[1, 3].axis('off')
    
    plt.suptitle('Data Loading Pipeline Diagnosis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2/data_loading_diagnosis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    # Check if images look reasonable
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    # Check if CLAHE output is reasonable
    clahe_ch = image_np[1]
    unique_vals = len(np.unique(clahe_ch))
    
    print(f"\nCLAHE channel statistics:")
    print(f"  Unique values: {unique_vals}")
    print(f"  Mean: {clahe_ch.mean():.3f}")
    print(f"  Std: {clahe_ch.std():.3f}")
    print(f"  Non-zero pixels: {(clahe_ch > 0).sum()} / {clahe_ch.size}")
    
    if clahe_ch.std() < 0.05:
        print("\n⚠️ WARNING: Very low variance - image might be too uniform!")
    
    if unique_vals < 100:
        print(f"\n⚠️ WARNING: Only {unique_vals} unique values - image might be corrupted!")
    
    # Check if it's all zeros or ones
    if clahe_ch.max() < 0.01:
        print("\n❌ ERROR: Image is nearly all zeros!")
    elif clahe_ch.min() > 0.99:
        print("\n❌ ERROR: Image is nearly all ones!")
    else:
        print("\n✅ Image value range looks reasonable")


if __name__ == '__main__':
    main()
