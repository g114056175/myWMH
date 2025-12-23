"""
完整7CH v2評估 + 通道重要性分析
包含真實混淆矩陣和詳細metrics
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
import torch

print("=" * 80)
print("7CH v2 - 完整評估")
print("=" * 80)

# 1. 通道重要性分析
print("\n【1/3】通道重要性分析...")
print("-" * 80)

checkpoint_path = Path("nnUNet_results/Dataset004_WMH_7CH_v2/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 提取第一層卷積權重
first_conv_weight = None
for key in checkpoint['network_weights'].keys():
    if 'conv_blocks_context.0.blocks.0.conv.weight' in key or \
       'encoder.stages.0' in key or \
       'model.0.conv' in key:
        first_conv_weight = checkpoint['network_weights'][key]
        print(f"Found first conv: {key}, shape: {first_conv_weight.shape}")
        break

if first_conv_weight is None:
    # 嘗試其他可能的keys
    for key in sorted(checkpoint['network_weights'].keys())[:10]:
        print(f"  {key}: {checkpoint['network_weights'][key].shape}")
    # 找第一個包含weight的conv層
    for key in sorted(checkpoint['network_weights'].keys()):
        if 'weight' in key and len(checkpoint['network_weights'][key].shape) == 4:
            w = checkpoint['network_weights'][key]
            if w.shape[1] == 7:  # 7 input channels
                first_conv_weight = w
                print(f"Using: {key}, shape: {w.shape}")
                break

if first_conv_weight is not None:
    # shape: [out_channels, 7, kernel_h, kernel_w]
    channel_importance = []
    num_channels = first_conv_weight.shape[1]
    
    for ch in range(num_channels):
        weight_ch = first_conv_weight[:, ch, :, :]
        l2_norm = torch.norm(weight_ch, p=2).item()
        channel_importance.append(l2_norm)
    
    total_norm = sum(channel_importance)
    channel_percentages = [(imp / total_norm) * 100 for imp in channel_importance]
    
    channel_names = [
        "FLAIR[t-1]",
        "FLAIR[t]", 
        "FLAIR[t+1]",
        "T1",
        "Butterworth(15)",
        "HighPass(σ=2)",
        "Top-hat"
    ]
    
    print("\n通道重要性排名:")
    for idx in np.argsort(channel_importance)[::-1]:
        print(f"  {idx+1}. {channel_names[idx]}: {channel_percentages[idx]:.2f}%")
    
    channel_analysis = {
        'rankings': [{'channel': channel_names[i], 'percentage': channel_percentages[i], 'l2_norm': channel_importance[i]} 
                     for i in range(num_channels)],
        'total_norm': total_norm
    }
else:
    print("⚠️ 無法找到第一層卷積，跳過通道分析")
    channel_analysis = None

# 2. 測試集評估
print("\n【2/3】測試集評估...")
print("-" * 80)

with open("selected_3d_cases.json", "r") as f:
    all_cases = json.load(f)

pred_dir = Path("batch_eval_7ch_v2/predictions")

if not pred_dir.exists():
    print("⚠️ Predictions不存在，需要先運行批量預測")
    print("請運行: python evaluate_7ch_v2_batch.py")
    exit(1)

results = []
all_tp = 0
all_fp = 0
all_fn = 0
all_tn = 0

for case_idx, case in enumerate(tqdm(all_cases, desc="評估")):
    case_id = case['id'].replace('/', '_').replace('\\', '_')
    case_path = Path(case['path'])
    
    # Load GT
    gt_path = case_path / "wmh.nii.gz"
    if not gt_path.exists():
        continue
    
    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata()
    
    # Reconstruct prediction
    pred_3d = np.zeros_like(gt_data)
    pred_files = sorted(pred_dir.glob(f"case{case_idx:02d}_slice*.nii.gz"))
    
    if not pred_files:
        print(f"Warning: No predictions for {case_id}")
        continue
    
    for pred_file in pred_files:
        filename = pred_file.stem.replace('.nii', '')
        slice_num = int(filename.split('_slice')[1])
        pred_slice = nib.load(pred_file).get_fdata()
        if slice_num < pred_3d.shape[2]:
            pred_3d[:, :, slice_num] = pred_slice
    
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

# Calculate overall
overall_dice = 2 * all_tp / (2 * all_tp + all_fp + all_fn) if (2 * all_tp + all_fp + all_fn) > 0 else 0
overall_sens = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
overall_spec = all_tn / (all_tn + all_fp) if (all_tn + all_fp) > 0 else 0

print(f"\n評估完成: {len(results)} cases")
print(f"Overall Dice: {overall_dice:.4f}")
print(f"Sensitivity: {overall_sens:.4f}")
print(f"Precision: {overall_prec:.4f}")

# 3. 創建可視化
print("\n【3/3】生成可視化...")
print("-" * 80)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Real Confusion Matrix
cm = np.array([[all_tn, all_fp], [all_fn, all_tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Pred Neg', 'Pred Pos'],
            yticklabels=['GT Neg', 'GT Pos'],
            cbar_kws={'label': 'Voxel Count'})
ax1.set_title('Confusion Matrix (Voxel-level)', fontsize=14, fontweight='bold')

# 2. Dice distribution
dices = [r['dice'] for r in results]
ax2.hist(dices, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(dices), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dices):.4f}')
ax2.axvline(0.7954, color='orange', linestyle=':', linewidth=2, label='v1 Mean: 0.7954')
ax2.set_xlabel('Dice Score', fontsize=12)
ax2.set_ylabel('Number of Cases', fontsize=12)
ax2.set_title(f'Dice Distribution (n={len(results)})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Metrics comparison
metrics_labels = ['Dice', 'Sensitivity', 'Precision', 'Specificity']
values = [overall_dice, overall_sens, overall_prec, overall_spec]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

bars = ax3.bar(metrics_labels, values, color=colors, alpha=0.7, edgecolor='black')
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
ax4.scatter(volumes, dices, alpha=0.5, s=50, c=dices, cmap='viridis', edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Lesion Volume (voxels)', fontsize=12)
ax4.set_ylabel('Dice Score', fontsize=12)
ax4.set_title('Dice vs Lesion Volume', fontsize=14, fontweight='bold')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=np.mean(dices), color='red', linestyle='--', linewidth=1)
ax4.axvline(x=100, color='orange', linestyle=':', alpha=0.5, label='Small (<100)')
ax4.axvline(x=1000, color='green', linestyle=':', alpha=0.5, label='Med (100-1k)')
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig("7CH_v2_Archive/evaluation_results.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Saved: 7CH_v2_Archive/evaluation_results.png")

# Save complete results
output_data = {
    'summary': {
        'overall_dice': overall_dice,
        'overall_sensitivity': overall_sens,
        'overall_precision': overall_prec,
        'overall_specificity': overall_spec,
        'mean_dice': float(np.mean(dices)),
        'median_dice': float(np.median(dices)),
        'std_dice': float(np.std(dices)),
        'num_cases': len(results)
    },
    'confusion_matrix': {
        'TP': int(all_tp),
        'FP': int(all_fp),
        'FN': int(all_fn),
        'TN': int(all_tn)
    },
    'per_case_results': results
}

if channel_analysis:
    output_data['channel_importance'] = channel_analysis

with open("7CH_v2_Archive/evaluation_metrics.json", 'w') as f:
    json.dump(output_data, f, indent=2)

print("✓ Saved: 7CH_v2_Archive/evaluation_metrics.json")
print("\n✅ 完整評估完成！")
