"""
Generate comprehensive V0.3 evaluation metrics
Including confusion matrix, per-class metrics, and visualizations
Following V0.2 evaluation format
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import AttentionUNet
from dataset import normalize_slice, center_crop, apply_clahe, apply_highpass, to_uint8
import config

def prepare_input(flair_3d, t1_3d, slice_idx, spatial_atlas):
    """Prepare 7-channel input"""
    flair_t_minus_1 = flair_3d[:, :, slice_idx - 1]
    flair_t = flair_3d[:, :, slice_idx]
    flair_t_plus_1 = flair_3d[:, :, slice_idx + 1]
    t1_t = t1_3d[:, :, slice_idx]
    
    clahe_t_minus_1 = apply_clahe(flair_t_minus_1, config.CLAHE_CLIP_LIMIT)
    clahe_t = apply_clahe(flair_t, config.CLAHE_CLIP_LIMIT)
    clahe_t_plus_1 = apply_clahe(flair_t_plus_1, config.CLAHE_CLIP_LIMIT)
    
    highpass_t = apply_highpass(flair_t, config.HIGHPASS_SIGMA)
    
    asymmetry = np.abs(flair_t - np.flipud(flair_t))
    asymmetry = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)
    
    t1_t_norm = to_uint8(t1_t).astype(np.float32) / 255.0
    
    image = np.stack([
        clahe_t_minus_1, clahe_t, clahe_t_plus_1,
        t1_t_norm, highpass_t, asymmetry, spatial_atlas
    ], axis=0).astype(np.float32)
    
    return torch.from_numpy(image)


def calculate_metrics(pred_volume, gt_volume, threshold=0.7):
    """Calculate comprehensive 3D metrics"""
    pred_binary = (pred_volume > threshold).astype(np.float32)
    valid_mask = (gt_volume != 2)
    gt_wmh = ((gt_volume == 1) & valid_mask).astype(np.float32)
    pred_valid = (pred_binary * valid_mask).astype(np.float32)
    
    tp = (pred_valid * gt_wmh).sum()
    fp = (pred_valid * (1 - gt_wmh) * valid_mask).sum()
    fn = ((1 - pred_valid) * gt_wmh).sum()
    tn = ((1 - pred_valid) * (1 - gt_wmh) * valid_mask).sum()
    
    dice = 2.0 * tp / (2*tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'dice': dice,
        'sensitivity': sensitivity,
        'precision': precision,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def main():
    output_dir = Path('d:/VSCode/AIOT_E1/newUnet2/V0.3_Final_Evaluation')
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("V0.3 Complete Evaluation with Confusion Matrix")
    print("="*70)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = AttentionUNet(in_channels=7).to(device)
    checkpoint = torch.load(
        Path(config.CHECKPOINT_DIR) / 'best_model.pth',
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Val Dice: {checkpoint.get('val_dice', 0):.4f}")
    
    # Load spatial atlas
    spatial_atlas = np.load(Path(config.CHECKPOINT_DIR).parent / 'spatial_atlas_prior.npy')
    
    # Get all test files
    test_dir = Path(config.TEST_DIR)
    test_files = sorted(list(test_dir.rglob('*/pre/FLAIR.nii.gz')))
    
    print(f"\nEvaluating {len(test_files)} test samples at threshold 0.7")
    print("="*70)
    
    # Accumulate confusion matrix
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    all_results = []
    
    for flair_path in tqdm(test_files, desc="Processing"):
        sample_path = flair_path.parent.parent
        sample_name = sample_path.name
        
        # Load data
        flair_raw = nib.load(flair_path).get_fdata()
        t1_raw = nib.load(sample_path / 'pre' / 'T1.nii.gz').get_fdata()
        mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
        
        flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
        t1 = center_crop(normalize_slice(t1_raw), config.TARGET_SIZE)
        mask = center_crop(mask_raw, config.TARGET_SIZE)
        
        # Predict
        pred_volume = np.zeros_like(mask)
        
        with torch.no_grad():
            for z in range(2, flair.shape[2] - 2):
                input_tensor = prepare_input(flair, t1, z, spatial_atlas).unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                pred_volume[:, :, z] = pred_prob
        
        # Calculate metrics
        metrics = calculate_metrics(pred_volume, mask, 0.7)
        metrics['sample_name'] = sample_name
        all_results.append(metrics)
        
        # Accumulate confusion matrix
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        total_tn += metrics['tn']
    
    # Calculate overall metrics
    overall_dice = 2.0 * total_tp / (2*total_tp + total_fp + total_fn)
    overall_sens = total_tp / (total_tp + total_fn)
    overall_prec = total_tp / (total_tp + total_fp)
    overall_spec = total_tn / (total_tn + total_fp)
    
    # Per-sample statistics
    dices = [r['dice'] for r in all_results]
    mean_dice = np.mean(dices)
    median_dice = np.median(dices)
    std_dice = np.std(dices)
    
    # Print results
    print("\n" + "="*70)
    print("V0.3 TEST SET EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n{'Samples':<30} {len(test_files)}")
    print(f"{'Threshold':<30} 0.7")
    print(f"{'Model Epoch':<30} {checkpoint.get('epoch', 'unknown')}")
    print(f"{'Val Dice (best)':<30} {checkpoint.get('val_dice', 0):.4f}")
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX (Aggregated across all test samples)")
    print("="*70)
    
    print(f"\n{'':>20} {'Predicted Positive':>20} {'Predicted Negative':>20}")
    print(f"{'Actual Positive':<20} {total_tp:>20,} {total_fn:>20,}")
    print(f"{'Actual Negative':<20} {total_fp:>20,} {total_tn:>20,}")
    
    total_pixels = total_tp + total_fp + total_fn + total_tn
    print(f"\n{'Total Pixels':<20} {total_pixels:>20,}")
    
    print("\n" + "="*70)
    print("AGGREGATE METRICS (Pixel-level)")
    print("="*70)
    
    print(f"\n{'Dice Coefficient':<30} {overall_dice:.4f}")
    print(f"{'Sensitivity (Recall)':<30} {overall_sens:.4f}")
    print(f"{'Precision':<30} {overall_prec:.4f}")
    print(f"{'Specificity':<30} {overall_spec:.4f}")
    
    print("\n" + "="*70)
    print("PER-SAMPLE STATISTICS")
    print("="*70)
    
    print(f"\n{'Mean Dice':<30} {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"{'Median Dice':<30} {median_dice:.4f}")
    print(f"{'Min Dice':<30} {min(dices):.4f}")
    print(f"{'Max Dice':<30} {max(dices):.4f}")
    
    # Save results
    results_dict = {
        'model_info': {
            'version': 'V0.3',
            'epoch': int(checkpoint.get('epoch', 0)),
            'val_dice': float(checkpoint.get('val_dice', 0)),
            'parameters': '31.4M',
            'architecture': '7-channel + DualAttention'
        },
        'confusion_matrix': {
            'TP': int(total_tp),
            'FP': int(total_fp),
            'FN': int(total_fn),
            'TN': int(total_tn),
            'total_pixels': int(total_pixels)
        },
        'aggregate_metrics': {
            'dice': float(overall_dice),
            'sensitivity': float(overall_sens),
            'precision': float(overall_prec),
            'specificity': float(overall_spec)
        },
        'per_sample_stats': {
            'mean_dice': float(mean_dice),
            'median_dice': float(median_dice),
            'std_dice': float(std_dice),
            'min_dice': float(min(dices)),
            'max_dice': float(max(dices))
        },
        'per_sample_results': [
            {
                'sample_name': r['sample_name'],
                'dice': float(r['dice']),
                'sensitivity': float(r['sensitivity']),
                'precision': float(r['precision']),
                'specificity': float(r['specificity']),
                'tp': int(r['tp']),
                'fp': int(r['fp']),
                'fn': int(r['fn']),
                'tn': int(r['tn'])
            }
            for r in all_results
        ]
    }
    
    # Save JSON
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'evaluation_results.json'}")
    
    # Create visualizations
    create_visualizations(results_dict, all_results, output_dir)
    
    print("\n✅ Complete evaluation finished!")


def create_visualizations(results_dict, all_results, output_dir):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrix Heatmap
    ax1 = plt.subplot(2, 3, 1)
    cm = np.array([
        [results_dict['confusion_matrix']['TP'], results_dict['confusion_matrix']['FN']],
        [results_dict['confusion_matrix']['FP'], results_dict['confusion_matrix']['TN']]
    ])
    
    im = ax1.imshow(cm, cmap='Blues', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Pred Positive', 'Pred Negative'])
    ax1.set_yticklabels(['Actual Positive', 'Actual Negative'])
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{cm[i, j]:,}',
                           ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                           fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # 2. Metrics bar chart
    ax2 = plt.subplot(2, 3, 2)
    metrics = ['Dice', 'Sensitivity', 'Precision', 'Specificity']
    values = [
        results_dict['aggregate_metrics']['dice'],
        results_dict['aggregate_metrics']['sensitivity'],
        results_dict['aggregate_metrics']['precision'],
        results_dict['aggregate_metrics']['specificity']
    ]
    
    bars = ax2.barh(metrics, values, color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#FECA57'])
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_title('Aggregate Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 3. Dice distribution
    ax3 = plt.subplot(2, 3, 3)
    dices = [r['dice'] for r in all_results]
    ax3.hist(dices, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(results_dict['per_sample_stats']['mean_dice'], 
                color='red', linestyle='--', linewidth=2, label=f"Mean: {results_dict['per_sample_stats']['mean_dice']:.4f}")
    ax3.axvline(results_dict['per_sample_stats']['median_dice'],
                color='green', linestyle='--', linewidth=2, label=f"Median: {results_dict['per_sample_stats']['median_dice']:.4f}")
    ax3.set_xlabel('Dice Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Dice Score Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Top 10 and Bottom 10 samples
    ax4 = plt.subplot(2, 3, 4)
    sorted_results = sorted(all_results, key=lambda x: x['dice'])
    
    bottom_5 = sorted_results[:5]
    top_5 = sorted_results[-5:]
    
    samples = [r['sample_name'][:10] for r in bottom_5] + [r['sample_name'][:10] for r in top_5]
    scores = [r['dice'] for r in bottom_5] + [r['dice'] for r in top_5]
    colors_list = ['red']*5 + ['green']*5
    
    y_pos = np.arange(len(samples))
    ax4.barh(y_pos, scores, color=colors_list, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(samples, fontsize=9)
    ax4.set_xlabel('Dice Score', fontsize=12)
    ax4.set_title('Top 5 & Bottom 5 Samples', fontsize=14, fontweight='bold')
    ax4.axvline(results_dict['per_sample_stats']['mean_dice'], color='blue', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. TP/FP/FN distribution
    ax5 = plt.subplot(2, 3, 5)
    tp_values = [r['tp'] for r in all_results]
    fp_values = [r['fp'] for r in all_results]
    fn_values = [r['fn'] for r in all_results]
    
    x = np.arange(len(all_results))
    ax5.scatter(x, tp_values, alpha=0.5, s=20, label='TP', color='green')
    ax5.scatter(x, fp_values, alpha=0.5, s=20, label='FP', color='red')
    ax5.scatter(x, fn_values, alpha=0.5, s=20, label='FN', color='orange')
    ax5.set_xlabel('Sample Index', fontsize=12)
    ax5.set_ylabel('Pixel Count', fontsize=12)
    ax5.set_title('TP/FP/FN Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
V0.3 Evaluation Summary

Confusion Matrix:
  TP: {results_dict['confusion_matrix']['TP']:,}
  FP: {results_dict['confusion_matrix']['FP']:,}
  FN: {results_dict['confusion_matrix']['FN']:,}
  TN: {results_dict['confusion_matrix']['TN']:,}

Metrics:
  Dice:        {results_dict['aggregate_metrics']['dice']:.4f}
  Sensitivity: {results_dict['aggregate_metrics']['sensitivity']:.4f}
  Precision:   {results_dict['aggregate_metrics']['precision']:.4f}
  Specificity: {results_dict['aggregate_metrics']['specificity']:.4f}

Sample Stats:
  Mean Dice:   {results_dict['per_sample_stats']['mean_dice']:.4f}
  Std Dice:    {results_dict['per_sample_stats']['std_dice']:.4f}
  Range:       [{results_dict['per_sample_stats']['min_dice']:.4f}, {results_dict['per_sample_stats']['max_dice']:.4f}]

Test Samples: {len(all_results)}
"""
    
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('V0.3 Complete Test Set Evaluation', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'comprehensive_metrics.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_dir / 'comprehensive_metrics.png'}")


if __name__ == '__main__':
    main()
