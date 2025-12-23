"""
7CH v1 Evaluation - Fixed version
Properly reconstruct 3D volumes from slice predictions
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("=" * 80)
print("7CH v1 - Test Set Evaluation (Fixed)")
print("=" * 80)

# Load test cases
with open("selected_3d_cases.json", "r") as f:
    all_cases = json.load(f)

pred_dir = Path("predictions_7ch_v1")

print(f"Test cases: {len(all_cases)}")
print(f"Prediction files: {len(list(pred_dir.glob('*.nii.gz')))}")
print("=" * 80)

results = []
all_tp = 0
all_fp = 0
all_fn = 0
all_tn = 0

for case in tqdm(all_cases, desc="Evaluating"):
    case_id = case['id'].replace('/', '_').replace('\\', '_')
    case_path = Path(case['path'])
    
    # Load GT
    gt_path = case_path / "wmh.nii.gz"
    if not gt_path.exists():
        continue
    
    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata()
    
    # Reconstruct prediction from slices
    # Prediction files are named: Site_Scanner_Case_sliceXXXX.nii.gz
    # Find pattern based on case path
    path_parts = case_path.parts
    
    # Determine naming pattern
    if 'Amsterdam' in str(case_path):
        # Amsterdam has scanner subdirs
        site = path_parts[-3]  # Amsterdam
        scanner = path_parts[-2]  # GE3T, GE1T5, Philips3T
        case_num = path_parts[-1]
        pred_prefix = f"{site}_{scanner}_{case_num}"
    elif 'Singapore' in str(case_path):
        site = path_parts[-2]
        case_num = path_parts[-1]
        pred_prefix = f"{site}_{case_num}"
    elif 'Utrecht' in str(case_path):
        site = path_parts[-2]
        case_num = path_parts[-1]
        pred_prefix = f"{site}_{case_num}"
    else:
        print(f"Unknown format: {case_path}")
        continue
    
    # Find all slices for this case
    pred_slices = sorted(pred_dir.glob(f"{pred_prefix}_slice*.nii.gz"))
    
    if not pred_slices:
        print(f"Warning: No predictions for {case_id}")
        continue
    
    # Reconstructaccumulate 3D prediction
    pred_3d = np.zeros_like(gt_data)
    
    for pred_slice_path in pred_slices:
        # Extract slice number from filename (remove .nii.gz)
        filename = pred_slice_path.stem  # Removes .gz
        filename = filename.replace('.nii', '')  # Remove .nii 
        slice_num = int(filename.split('_slice')[1])
        
        # Load slice prediction
        pred_slice_data = nib.load(pred_slice_path).get_fdata()
        
        # Store in 3D array
        if slice_num < pred_3d.shape[2]:
            pred_3d[:, :, slice_num] = pred_slice_data
    
    # Calculate metrics
    valid_mask = (gt_data != 2)
    pred_bin = ((pred_3d > 0) & valid_mask).astype(bool)
    gt_bin = ((gt_data == 1) & valid_mask).astype(bool)
    
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    tn = np.sum(~pred_bin & ~gt_bin)
    
    all_tp += tp
    all_fp += fp
    all_fn += fn
    all_tn += tn
    
    # Per-case metrics
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    results.append({
        'case': case_id,
        'volume': case['volume'],
        'dice': dice,
        'sensitivity': sens,
        'precision': prec,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    })

# Calculate overall metrics
overall_dice = 2 * all_tp / (2 * all_tp + all_fp + all_fn) if (2 * all_tp + all_fp + all_fn) > 0 else 0
overall_sens = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
overall_spec = all_tn / (all_tn + all_fp) if (all_tn + all_fp) > 0 else 0

print("\n" + "=" * 80)
print("Evaluation Results")
print("=" * 80)
print(f"Cases evaluated: {len(results)}")
print(f"\nOverall 3D Metrics:")
print(f"  Dice:        {overall_dice:.4f}")
print(f"  Sensitivity: {overall_sens:.4f}")
print(f"  Precision:   {overall_prec:.4f}")
print(f"  Specificity: {overall_spec:.4f}")

# Per-case statistics
dices = [r['dice'] for r in results]
print(f"\nPer-Case Dice Statistics:")
print(f"  Mean:   {np.mean(dices):.4f}")
print(f"  Median: {np.median(dices):.4f}")
print(f"  Std:    {np.std(dices):.4f}")
print(f"  Min:    {np.min(dices):.4f}")
print(f"  Max:    {np.max(dices):.4f}")

# By volume category
small = [r for r in results if r['volume'] < 100]
medium = [r for r in results if 100 <= r['volume'] < 1000]
large = [r for r in results if r['volume'] >= 1000]

print(f"\nBy Volume Category:")
print(f"  Small (<100 vx, n={len(small)}):   Dice {np.mean([r['dice'] for r in small]):.4f}" if small else "  Small: N/A")
print(f"  Medium (100-1k, n={len(medium)}): Dice {np.mean([r['dice'] for r in medium]):.4f}" if medium else "  Medium: N/A")
print(f"  Large (>1k, n={len(large)}):       Dice {np.mean([r['dice'] for r in large]):.4f}" if large else "  Large: N/A")

print("=" * 80)

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Confusion Matrix
cm = np.array([[all_tn, all_fp], 
               [all_fn, all_tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Pred Neg', 'Pred Pos'],
            yticklabels=['GT Neg', 'GT Pos'],
            cbar_kws={'label': 'Voxel Count'})
ax1.set_title('Confusion Matrix (Voxel-level)', fontsize=14, fontweight='bold')

# 2. Dice distribution
ax2.hist(dices, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(dices), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dices):.4f}')
ax2.axvline(0.8761, color='orange', linestyle=':', linewidth=2, label='4CH: 0.8761')
ax2.set_xlabel('Dice Score', fontsize=12)
ax2.set_ylabel('Number of Cases', fontsize=12)
ax2.set_title(f'Dice Distribution (n={len(results)})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Metrics comparison
metrics = ['Dice', 'Sensitivity', 'Precision', 'Specificity']
values = [overall_dice, overall_sens, overall_prec, overall_spec]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylim([0, 1])
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Dice vs Volume
volumes = [r['volume'] for r in results]
ax4.scatter(volumes, dices, alpha=0.5, s=50, c=dices, cmap='viridis')
ax4.set_xlabel('Lesion Volume (voxels)', fontsize=12)
ax4.set_ylabel('Dice Score', fontsize=12)
ax4.set_title('Dice vs Lesion Volume', fontsize=14, fontweight='bold')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=np.mean(dices), color='red', linestyle='--', linewidth=1)
ax4.axvline(x=100, color='orange', linestyle=':', alpha=0.5, label='Small/Med')
ax4.axvline(x=1000, color='green', linestyle=':', alpha=0.5, label='Med/Large')
ax4.legend(fontsize=8)

plt.tight_layout()
output_path = "7CH_v1_Archive/evaluation_results.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")

# Save results
output_json = "7CH_v1_Archive/evaluation_metrics.json"
with open(output_json, 'w') as f:
    json.dump({
        'summary': {
            'overall_dice': overall_dice,
            'overall_sensitivity': overall_sens,
            'overall_precision': overall_prec,
            'overall_specificity': overall_spec,
            'mean_dice': float(np.mean(dices)),
            'median_dice': float(np.median(dices)),
            'std_dice': float(np.std(dices)),
            'num_cases': len(results),
            'comparison_vs_4ch': overall_dice - 0.7983
        },
        'confusion_matrix': {
            'TP': int(all_tp),
            'FP': int(all_fp),
            'FN': int(all_fn),
            'TN': int(all_tn)
        },
        'per_case_results': results
    }, f, indent=2)

print(f"✓ Saved: {output_json}")
print("\n✓ Evaluation complete!")
