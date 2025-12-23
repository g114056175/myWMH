"""
Add TP/FP/FN/Dice metrics to demo prediction visualizations
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("=" * 80)
print("Adding Metrics to Demo Predictions")
print("=" * 80)

# Load demo info
demo_dir = Path("output/demo_predictions")
with open(demo_dir / "demo_info.json", 'r') as f:
    demo_info = json.load(f)

# Regenerate images with metrics
print("\nRegenerating visualizations with metrics...")
print("-" * 80)

# We need to recalculate from the stored predictions
# Since we cleaned up temp folders, let's recreate from case data

from scipy.ndimage import gaussian_filter, grey_opening
import subprocess
import shutil

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

# Load test cases
with open("nnUnet/selected_3d_cases.json", "r") as f:
    all_cases = json.load(f)

# Map case IDs to indices
case_map = {case['id']: idx for idx, case in enumerate(all_cases)}

# Prepare temp folders
temp_dir = Path("output/temp_metrics")
if temp_dir.exists():
    shutil.rmtree(temp_dir)
temp_dir.mkdir()
images_dir = temp_dir / "images"
predictions_dir = temp_dir / "predictions"
images_dir.mkdir()
predictions_dir.mkdir()

# Process each demo case
results_with_metrics = []

for result in demo_info['results']:
    case_id = result['case_id']
    demo_id = result['demo_id']
    
    # Find case in all_cases
    case_idx = case_map[case_id]
    case = all_cases[case_idx]
    case_path = Path(case['path'])
    
    print(f"Processing Demo {demo_id}: {case_id}")
    
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
    
    # Prepare 7 channels
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
    slice_name = f"demo{demo_id-1:02d}"
    for ch_idx, data in enumerate(channels):
        img_2d = nib.Nifti1Image(data.astype(np.float32), flair_nii.affine)
        img_path = images_dir / f"{slice_name}_{ch_idx:04d}.nii.gz"
        nib.save(img_2d, img_path)
    
    results_with_metrics.append({
        'demo_id': demo_id,
        'case_id': case_id,
        'volume': result['volume'],
        'flair_slice': flair_curr,
        'wmh_slice': wmh_data[:, :, target_slice],
        'slice_name': slice_name
    })

# Run prediction
print("\nRunning prediction...")
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

print("✓ Prediction complete")

# Generate visualizations with metrics
print("\nGenerating visualizations with metrics...")
print("-" * 80)

for info in results_with_metrics:
    # Load prediction
    pred_file = predictions_dir / f"{info['slice_name']}.nii.gz"
    pred_data = nib.load(pred_file).get_fdata()
    
    # Calculate metrics
    valid_mask = (info['wmh_slice'] != 2)
    pred_bin = ((pred_data > 0) & valid_mask).astype(bool)
    gt_bin = ((info['wmh_slice'] == 1) & valid_mask).astype(bool)
    
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    tn = np.sum(~pred_bin & ~gt_bin)
    
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Title with case info
    fig.suptitle(f"Demo {info['demo_id']}: {info['case_id']} (Lesion Volume: {info['volume']} voxels)", 
                 fontsize=16, fontweight='bold')
    
    # 1. FLAIR Original
    axes[0].imshow(info['flair_slice'], cmap='gray')
    axes[0].set_title('FLAIR Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth overlay
    flair_gt = np.stack([info['flair_slice'], info['flair_slice'], info['flair_slice']], axis=-1)
    lesion_mask = gt_bin
    flair_gt[lesion_mask, 0] = 1.0
    flair_gt[lesion_mask, 1] = 0.0
    flair_gt[lesion_mask, 2] = 0.0
    
    axes[1].imshow(flair_gt)
    axes[1].set_title('Ground Truth\n(Red = Lesion)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Prediction overlay with color coding
    flair_pred = np.stack([info['flair_slice'], info['flair_slice'], info['flair_slice']], axis=-1)
    
    # TP: Yellow (Red + Green)
    tp_mask = pred_bin & gt_bin
    flair_pred[tp_mask, 0] = 1.0
    flair_pred[tp_mask, 1] = 1.0
    flair_pred[tp_mask, 2] = 0.0
    
    # FP: Green
    fp_mask = pred_bin & ~gt_bin
    flair_pred[fp_mask, 0] = 0.0
    flair_pred[fp_mask, 1] = 1.0
    flair_pred[fp_mask, 2] = 0.0
    
    # FN: Red
    fn_mask = ~pred_bin & gt_bin
    flair_pred[fn_mask, 0] = 1.0
    flair_pred[fn_mask, 1] = 0.0
    flair_pred[fn_mask, 2] = 0.0
    
    axes[2].imshow(flair_pred)
    axes[2].set_title('7CH v2 Prediction\n(Yellow=TP, Green=FP, Red=FN)', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add metrics text box
    metrics_text = f"""Metrics:
Dice: {dice:.4f}
Sensitivity: {sens:.4f}
Precision: {prec:.4f}

Confusion:
TP: {tp:,} px
FP: {fp:,} px
FN: {fn:,} px
TN: {tn:,} px"""
    
    # Add text box to the right
    fig.text(0.98, 0.5, metrics_text, 
             fontsize=11, 
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    # Save
    output_path = demo_dir / f"demo_{info['demo_id']}_{info['case_id'].replace('/', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: demo_{info['demo_id']}_{info['case_id'].replace('/', '_')}.png")
    print(f"   Dice: {dice:.4f}, TP: {tp}, FP: {fp}, FN: {fn}")

# Cleanup
shutil.rmtree(temp_dir)

print("\n" + "=" * 80)
print("✅ Metrics visualizations complete!")
print("=" * 80)
print("\nEach image now shows:")
print("  - FLAIR Original")
print("  - Ground Truth (Red)")
print("  - Prediction with color-coded results:")
print("    * Yellow = True Positive (TP)")
print("    * Green = False Positive (FP)")
print("    * Red = False Negative (FN)")
print("  - Metrics box: Dice, Sensitivity, Precision, TP/FP/FN/TN")
