"""
Complete Test Set Evaluation (3D)
对完整测试集进行3D评估，每个样本计算整个volume的Dice
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import AttentionUNet
from dataset import normalize_slice, center_crop, to_uint8, apply_clahe, apply_highpass
import config


def prepare_input(flair_3d, t1_3d, slice_idx, spatial_atlas):
    """Prepare 7-channel input for V0.3"""
    flair_t_minus_1 = flair_3d[:, :, slice_idx - 1]
    flair_t = flair_3d[:, :, slice_idx]
    flair_t_plus_1 = flair_3d[:, :, slice_idx + 1]
    t1_t = t1_3d[:, :, slice_idx]
    
    # V0.3: Apply CLAHE to all 3 FLAIR slices
    clahe_t_minus_1 = apply_clahe(flair_t_minus_1, config.CLAHE_CLIP_LIMIT)
    clahe_t = apply_clahe(flair_t, config.CLAHE_CLIP_LIMIT)
    clahe_t_plus_1 = apply_clahe(flair_t_plus_1, config.CLAHE_CLIP_LIMIT)
    
    # HighPass on current slice
    highpass_t = apply_highpass(flair_t, config.HIGHPASS_SIGMA)
    
    # Asymmetry feature
    asymmetry = np.abs(flair_t - np.flipud(flair_t))
    asymmetry = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)
    
    # T1 normalization
    t1_t_norm = to_uint8(t1_t).astype(np.float32) / 255.0
    
    # Stack 7 channels
    image = np.stack([
        clahe_t_minus_1,    # Ch0
        clahe_t,            # Ch1
        clahe_t_plus_1,     # Ch2
        t1_t_norm,          # Ch3
        highpass_t,         # Ch4
        asymmetry,          # Ch5
        spatial_atlas,      # Ch6
    ], axis=0).astype(np.float32)
    
    return torch.from_numpy(image)


def calculate_3d_metrics(pred_volume, gt_volume, threshold=0.7):
    """
    Calculate 3D metrics for entire volume
    Excludes don't care regions (mask==2)
    """
    # Binarize prediction
    pred_binary = (pred_volume > threshold).astype(np.float32)
    
    # Exclude don't care regions
    valid_mask = (gt_volume != 2)
    gt_wmh = ((gt_volume == 1) & valid_mask).astype(np.float32)
    pred_valid = (pred_binary * valid_mask).astype(np.float32)
    
    # Calculate 3D confusion matrix
    tp = (pred_valid * gt_wmh).sum()
    fp = (pred_valid * (1 - gt_wmh) * valid_mask).sum()
    fn = ((1 - pred_valid) * gt_wmh).sum()
    tn = ((1 - pred_valid) * (1 - gt_wmh) * valid_mask).sum()
    
    # Calculate metrics
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


def evaluate_sample(model, flair_path, device, threshold=0.7):
    """Evaluate one sample (3D volume)"""
    sample_path = flair_path.parent.parent
    sample_name = sample_path.name
    
    # Load data
    flair_raw = nib.load(flair_path).get_fdata()
    t1_raw = nib.load(sample_path / 'pre' / 'T1.nii.gz').get_fdata()
    mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
    
    # Preprocess
    flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
    t1 = center_crop(normalize_slice(t1_raw), config.TARGET_SIZE)
    mask = center_crop(mask_raw, config.TARGET_SIZE)
    
    # Predict all slices
    pred_volume = np.zeros_like(mask)
    
    with torch.no_grad():
        for z in range(2, flair.shape[2] - 2):
            input_tensor = prepare_input(flair, t1, z).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_volume[:, :, z] = pred_prob
    
    # Calculate 3D metrics
    metrics = calculate_3d_metrics(pred_volume, mask, threshold)
    metrics['sample_name'] = sample_name
    
    return metrics


def main():
    print("="*70)
    print("Complete Test Set Evaluation (3D)")
    print("="*70)
    
    # Load model - V0.3: 7 channels
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print("\nLoading model...")
    model = AttentionUNet(
        in_channels=7,  # V0.3: 7 channels
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        depth=config.DEPTH
    ).to(device)
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Dice: {checkpoint.get('val_dice', 'unknown'):.4f}")
    
    # Load spatial atlas
    spatial_atlas_path = Path(config.CHECKPOINT_DIR).parent / 'spatial_atlas_prior.npy'
    spatial_atlas = np.load(spatial_atlas_path)
    print(f"\nSpatial atlas loaded: {spatial_atlas.shape}")
    
    # Find all test samples
    test_dir = Path(config.TEST_DIR)
    test_files = sorted(list(test_dir.rglob('*/pre/FLAIR.nii.gz')))
    
    print(f"\nFound {len(test_files)} test samples")
    print(f"Test directory: {test_dir}")
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    results = {thresh: [] for thresh in thresholds}
    
    print(f"\nEvaluating with thresholds: {thresholds}")
    print("="*70)
    
    # Evaluate each sample
    for flair_path in tqdm(test_files, desc="Processing samples"):
        sample_path = flair_path.parent.parent
        sample_name = sample_path.name
        
        # Load and preprocess
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
        
        # Evaluate at each threshold
        for thresh in thresholds:
            metrics = calculate_3d_metrics(pred_volume, mask, thresh)
            metrics['sample_name'] = sample_name
            results[thresh].append(metrics)
    
    # Print results
    print("\n" + "="*70)
    print("COMPLETE TEST SET RESULTS (3D Evaluation)")
    print("="*70)
    
    for thresh in thresholds:
        print(f"\n{'='*70}")
        print(f"Threshold: {thresh}")
        print(f"{'='*70}")
        
        sample_results = results[thresh]
        
        # Calculate statistics
        dices = [r['dice'] for r in sample_results]
        sens = [r['sensitivity'] for r in sample_results]
        precs = [r['precision'] for r in sample_results]
        specs = [r['specificity'] for r in sample_results]
        
        # Arithmetic mean
        mean_dice = np.mean(dices)
        mean_sens = np.mean(sens)
        mean_prec = np.mean(precs)
        mean_spec = np.mean(specs)
        
        # Median
        median_dice = np.median(dices)
        
        # Std
        std_dice = np.std(dices)
        
        print(f"\nOverall Statistics ({len(sample_results)} samples):")
        print(f"  Mean Dice:        {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"  Median Dice:      {median_dice:.4f}")
        print(f"  Mean Sensitivity: {mean_sens:.4f}")
        print(f"  Mean Precision:   {mean_prec:.4f}")
        print(f"  Mean Specificity: {mean_spec:.4f}")
        
        # Per-sample results (top 5 and bottom 5)
        sorted_results = sorted(sample_results, key=lambda x: x['dice'], reverse=True)
        
        print(f"\nTop 5 samples (Best Dice):")
        print(f"{'Sample':<15} {'Dice':<8} {'Sens':<8} {'Prec':<8} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-"*70)
        for r in sorted_results[:5]:
            print(f"{r['sample_name']:<15} {r['dice']:<8.4f} {r['sensitivity']:<8.4f} "
                  f"{r['precision']:<8.4f} {r['tp']:<8} {r['fp']:<8} {r['fn']:<8}")
        
        print(f"\nBottom 5 samples (Worst Dice):")
        print(f"{'Sample':<15} {'Dice':<8} {'Sens':<8} {'Prec':<8} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-"*70)
        for r in sorted_results[-5:]:
            print(f"{r['sample_name']:<15} {r['dice']:<8.4f} {r['sensitivity']:<8.4f} "
                  f"{r['precision']:<8.4f} {r['tp']:<8} {r['fp']:<8} {r['fn']:<8}")
    
    # Find best threshold
    print("\n" + "="*70)
    print("THRESHOLD COMPARISON")
    print("="*70)
    
    for thresh in thresholds:
        mean_dice = np.mean([r['dice'] for r in results[thresh]])
        print(f"Threshold {thresh}: Mean Dice = {mean_dice:.4f}")
    
    best_thresh = max(thresholds, key=lambda t: np.mean([r['dice'] for r in results[t]]))
    best_dice = np.mean([r['dice'] for r in results[best_thresh]])
    
    print(f"\nBest Threshold: {best_thresh}")
    print(f"Best Mean Dice: {best_dice:.4f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
