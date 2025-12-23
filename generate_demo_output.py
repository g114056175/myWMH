"""
Generate 5 demo predictions using 7CH v2 model
Save visualizations to output folder
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import subprocess
import shutil
from scipy.ndimage import gaussian_filter, grey_opening

# Butterworth implementation
def normalize_image(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)

def butterworth_highpass(image, cutoff=15, order=2):
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + (cutoff / (D + 1e-6))**(2*order))
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return normalize_image(result)

def simple_highpass(image, sigma=2.0):
    blurred = gaussian_filter(image, sigma=sigma)
    highpass = image - blurred
    return normalize_image(highpass)

def tophat_transform(image, size=15):
    opened = grey_opening(image, size=(size, size))
    tophat = image - opened
    return normalize_image(tophat)

print("=" * 80)
print("7CH v2 Model - Demo Predictions")
print("=" * 80)

# Load test cases
with open("nnUnet/selected_3d_cases.json", "r") as f:
    all_cases = json.load(f)

# Select 5 diverse cases (different sizes, sites)
selected_indices = [
    0,   # Amsterdam_GE1T5_150: 968 voxels (Small)
    11,  # Amsterdam_GE3T_117: 13342 voxels (Large)
    50,  # Singapore_70: 2456 voxels (Medium)
    75,  # Utrecht_13: 18658 voxels (Very Large)
    95,  # Utrecht_42: 5113 voxels (Medium)
]

selected_cases = [all_cases[i] for i in selected_indices]

# Create demo prediction folder
demo_dir = Path("output/demo_predictions")
if demo_dir.exists():
    shutil.rmtree(demo_dir)
demo_dir.mkdir(parents=True)

images_dir = demo_dir / "temp_images"
predictions_dir = demo_dir / "temp_predictions"
images_dir.mkdir()
predictions_dir.mkdir()

print(f"\nPreparing 5 demo cases...")
print("-" * 80)

case_info = []
slice_count = 0

for case_idx, case in enumerate(selected_cases):
    case_id = case['id']
    case_path = Path(case['path'])
    volume = case['volume']
    
    print(f"{case_idx+1}. {case_id}: {volume} voxels")
    
    # Load data
    flair_path = case_path / "pre" / "FLAIR.nii.gz"
    t1_path = case_path / "pre" / "T1.nii.gz"
    wmh_path = case_path / "wmh.nii.gz"
    
    flair_nii = nib.load(flair_path)
    t1_nii = nib.load(t1_path)
    wmh_nii = nib.load(wmh_path)
    
    flair_data = flair_nii.get_fdata()
    t1_data = t1_nii.get_fdata()
    wmh_data = wmh_nii.get_fdata()
    
    # Normalize
    flair_norm = normalize_image(flair_data)
    t1_norm = normalize_image(t1_data)
    
    # Find best slice with lesions
    lesion_slices = np.where(np.sum(wmh_data == 1, axis=(0,1)) > 100)[0]
    if len(lesion_slices) > 0:
        target_slice = lesion_slices[len(lesion_slices)//2]
    else:
        target_slice = flair_data.shape[2] // 2
    
    # Prepare 7 channels for this slice
    z = target_slice
    flair_prev = flair_norm[:, :, max(0, z-1)]
    flair_curr = flair_norm[:, :, z]
    flair_next = flair_norm[:, :, min(flair_data.shape[2]-1, z+1)]
    t1_slice = t1_norm[:, :, z]
    
    butterworth_slice = butterworth_highpass(flair_curr, cutoff=15, order=2)
    highpass_slice = simple_highpass(flair_curr, sigma=2.0)
    tophat_slice = tophat_transform(flair_curr, size=15)
    
    channels = [flair_prev, flair_curr, flair_next, t1_slice,
               butterworth_slice, highpass_slice, tophat_slice]
    
    # Save 7 channels
    slice_name = f"demo{case_idx:02d}"
    for ch_idx, data in enumerate(channels):
        img_2d = nib.Nifti1Image(data.astype(np.float32), flair_nii.affine)
        img_path = images_dir / f"{slice_name}_{ch_idx:04d}.nii.gz"
        nib.save(img_2d, img_path)
    
    # Store info
    case_info.append({
        'case_id': case_id,
        'volume': volume,
        'slice': target_slice,
        'slice_name': slice_name,
        'flair_slice': flair_curr,
        'wmh_slice': wmh_data[:, :, target_slice]
    })
    
    slice_count += 1

print(f"\n✓ Prepared {slice_count} slices")

# Run prediction
print("\nRunning 7CH v2 prediction...")
print("-" * 80)

cmd = [
    "nnUNetv2_predict",
    "-i", str(images_dir.absolute()),
    "-o", str(predictions_dir.absolute()),
    "-d", "004",
    "-c", "2d",
    "-f", "0",
    "-chk", "checkpoint_best.pth"
]

result = subprocess.run(cmd, cwd="d:/VSCode/AIOT_E1/nnUnet", 
                       capture_output=True, text=True)

if result.returncode != 0:
    print(f"Prediction failed: {result.stderr}")
    exit(1)

print("✓ Prediction complete")

# Generate visualizations
print("\nGenerating visualizations...")
print("-" * 80)

for idx, info in enumerate(case_info):
    # Load prediction
    pred_file = predictions_dir / f"{info['slice_name']}.nii.gz"
    pred_nii = nib.load(pred_file)
    pred_data = pred_nii.get_fdata()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    fig.suptitle(f"Demo {idx+1}: {info['case_id']} (Lesion Volume: {info['volume']} voxels)", 
                 fontsize=14, fontweight='bold')
    
    # 1. FLAIR Original
    axes[0].imshow(info['flair_slice'], cmap='gray')
    axes[0].set_title('FLAIR Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth overlay
    flair_gt = np.stack([info['flair_slice'], info['flair_slice'], info['flair_slice']], axis=-1)
    lesion_mask = (info['wmh_slice'] == 1)
    flair_gt[lesion_mask, 0] = 1.0
    flair_gt[lesion_mask, 1] = 0.0
    flair_gt[lesion_mask, 2] = 0.0
    
    axes[1].imshow(flair_gt)
    axes[1].set_title('Ground Truth (Red = Lesion)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Prediction overlay
    flair_pred = np.stack([info['flair_slice'], info['flair_slice'], info['flair_slice']], axis=-1)
    pred_mask = (pred_data > 0)
    flair_pred[pred_mask, 0] = 0.0
    flair_pred[pred_mask, 1] = 1.0
    flair_pred[pred_mask, 2] = 0.0
    
    axes[2].imshow(flair_pred)
    axes[2].set_title('7CH v2 Prediction (Green = Predicted)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = demo_dir / f"demo_{idx+1}_{info['case_id'].replace('/', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")

# Calculate metrics for demo cases
print("\nCalculating metrics...")
print("-" * 80)

demo_results = []
for idx, info in enumerate(case_info):
    pred_file = predictions_dir / f"{info['slice_name']}.nii.gz"
    pred_data = nib.load(pred_file).get_fdata()
    
    valid_mask = (info['wmh_slice'] != 2)
    pred_bin = ((pred_data > 0) & valid_mask).astype(bool)
    gt_bin = ((info['wmh_slice'] == 1) & valid_mask).astype(bool)
    
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    
    demo_results.append({
        'demo_id': idx + 1,
        'case_id': info['case_id'],
        'volume': info['volume'],
        'dice': dice
    })
    
    print(f"Demo {idx+1}: {info['case_id']} - Dice: {dice:.4f}")

# Save demo info
demo_info = {
    'model': '7CH v2 (Butterworth)',
    'checkpoint': 'checkpoint_best.pth',
    'num_samples': len(case_info),
    'results': demo_results
}

with open(demo_dir / "demo_info.json", 'w') as f:
    json.dump(demo_info, f, indent=2)

# Create README for output folder
readme_content = f"""# Demo Predictions - 7CH v2 Model

## Model Information
- **Model**: 7-Channel v2 with Butterworth High-Pass Filter
- **Checkpoint**: checkpoint_best.pth (Epoch 70)
- **Overall Performance**: Dice 0.8575 on 110 test cases

## Demo Cases

{chr(10).join([f"{i+1}. **{r['case_id']}** - Volume: {r['volume']} voxels, Dice: {r['dice']:.4f}" for i, r in enumerate(demo_results)])}

## Visualization Guide

Each image shows:
- **Left**: FLAIR Original MRI image
- **Middle**: Ground Truth (Red = WMH lesions)
- **Right**: Model Prediction (Green = Predicted lesions)

## Files
"""

for idx, info in enumerate(case_info):
    readme_content += f"- `demo_{idx+1}_{info['case_id'].replace('/', '_')}.png`\n"

readme_content += "\n## Notes\n"
readme_content += "- These are slice-level predictions for demonstration\n"
readme_content += "- Full 3D evaluation achieves higher Dice scores\n"
readme_content += "- Green overlaps with Red indicate correct predictions\n"

with open(demo_dir / "README.md", 'w') as f:
    f.write(readme_content)

# Cleanup temp folders
shutil.rmtree(images_dir)
shutil.rmtree(predictions_dir)

print("\n" + "=" * 80)
print("✅ Demo generation complete!")
print("=" * 80)
print(f"\nOutput location: output/demo_predictions/")
print(f"Generated files:")
print(f"  - 5 prediction visualizations (.png)")
print(f"  - demo_info.json (metrics)")
print(f"  - README.md (documentation)")
