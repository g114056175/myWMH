"""
Test Time Augmentation (TTA) Evaluation for V0.6
Compare with baseline (no TTA) to see Dice improvement
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
    """Prepare 7-channel input"""
    flair_t_minus_1 = flair_3d[:, :, slice_idx - 1]
    flair_t = flair_3d[:, :, slice_idx]
    flair_t_plus_1 = flair_3d[:, :, slice_idx + 1]
    t1_t = t1_3d[:, :, slice_idx]
    
    # CLAHE
    clahe_t_minus_1 = apply_clahe(flair_t_minus_1, config.CLAHE_CLIP_LIMIT)
    clahe_t = apply_clahe(flair_t, config.CLAHE_CLIP_LIMIT)
    clahe_t_plus_1 = apply_clahe(flair_t_plus_1, config.CLAHE_CLIP_LIMIT)
    
    # HighPass
    highpass_t = apply_highpass(flair_t, config.HIGHPASS_SIGMA)
    
    # Asymmetry (注意：不能HFlip，会破坏此特征)
    asymmetry = np.abs(flair_t - np.flipud(flair_t))
    asymmetry = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)
    
    # T1
    t1_t_norm = to_uint8(t1_t).astype(np.float32) / 255.0
    
    # Stack 7 channels
    image = np.stack([
        clahe_t_minus_1,
        clahe_t,
        clahe_t_plus_1,
        t1_t_norm,
        highpass_t,
        asymmetry,
        spatial_atlas,
    ], axis=0).astype(np.float32)
    
    return torch.from_numpy(image)


def apply_augmentation(image, aug_type):
    """Apply augmentation to 7-channel input
    
    注意: 不能用HFlip因为会破坏Asymmetry特征
    """
    if aug_type == 'original':
        return image
    elif aug_type == 'vflip':
        # Vertical flip (上下翻转)
        return torch.flip(image, dims=[1])
    elif aug_type == 'rot90':
        # Rotate 90 degrees
        return torch.rot90(image, k=1, dims=[1, 2])
    elif aug_type == 'rot180':
        # Rotate 180 degrees
        return torch.rot90(image, k=2, dims=[1, 2])
    elif aug_type == 'rot270':
        # Rotate 270 degrees
        return torch.rot90(image, k=3, dims=[1, 2])
    else:
        raise ValueError(f"Unknown augmentation: {aug_type}")


def reverse_augmentation(pred, aug_type):
    """Reverse augmentation on prediction"""
    if aug_type == 'original':
        return pred
    elif aug_type == 'vflip':
        return torch.flip(pred, dims=[2])
    elif aug_type == 'rot90':
        return torch.rot90(pred, k=-1, dims=[2, 3])
    elif aug_type == 'rot180':
        return torch.rot90(pred, k=-2, dims=[2, 3])
    elif aug_type == 'rot270':
        return torch.rot90(pred, k=-3, dims=[2, 3])
    else:
        raise ValueError(f"Unknown augmentation: {aug_type}")


def predict_with_tta(model, input_tensor, device, augmentations):
    """Predict with TTA (average multiple augmentations)"""
    predictions = []
    
    for aug in augmentations:
        # Apply augmentation
        aug_input = apply_augmentation(input_tensor, aug)
        aug_input = aug_input.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(aug_input, deep_supervision=False)
            pred = torch.sigmoid(output)
        
        # Reverse augmentation
        pred = reverse_augmentation(pred, aug)
        predictions.append(pred.cpu())
    
    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred.numpy()[0, 0]


def calculate_3d_metrics(pred_volume, gt_volume, threshold=0.7):
    """Calculate 3D metrics"""
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
    
    return {
        'dice': dice,
        'sensitivity': sensitivity,
        'precision': precision,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def main():
    print("="*70)
    print("TTA Evaluation for V0.6")
    print("="*70)
    
    # TTA configurations to test
    tta_configs = {
        'Baseline (No TTA)': ['original'],
        'TTA-VFlip': ['original', 'vflip'],
        'TTA-Rot4': ['original', 'rot90', 'rot180', 'rot270'],
        'TTA-Full': ['original', 'vflip', 'rot90', 'rot180', 'rot270'],
    }
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = AttentionUNet(
        in_channels=7,
        base_channels=config.BASE_CHANNELS,
        depth=config.DEPTH,
        dropout=config.DROPOUT
    ).to(device)
    
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel: V0.6 (Epoch {checkpoint['epoch']}, Val Dice {checkpoint['val_dice']:.4f})")
    
    # Load spatial atlas
    spatial_atlas = np.load(Path(config.CHECKPOINT_DIR).parent / 'spatial_atlas_prior.npy')
    
    # Find test samples
    test_dir = Path(config.TEST_DIR)
    test_files = sorted(list(test_dir.rglob('*/pre/FLAIR.nii.gz')))
    print(f"\nTest samples: {len(test_files)}")
    
    # Test each TTA configuration
    threshold = 0.7
    all_results = {}
    
    for tta_name, augmentations in tta_configs.items():
        print(f"\n{'='*70}")
        print(f"Testing: {tta_name}")
        print(f"Augmentations: {augmentations}")
        print(f"{'='*70}")
        
        results = []
        
        for flair_path in tqdm(test_files, desc=f"{tta_name}"):
            sample_path = flair_path.parent.parent
            
            # Load data
            flair_raw = nib.load(flair_path).get_fdata()
            t1_raw = nib.load(sample_path / 'pre' / 'T1.nii.gz').get_fdata()
            mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
            
            flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
            t1 = center_crop(normalize_slice(t1_raw), config.TARGET_SIZE)
            mask = center_crop(mask_raw, config.TARGET_SIZE)
            
            # Predict with TTA
            pred_volume = np.zeros_like(mask)
            
            for z in range(2, flair.shape[2] - 2):
                input_tensor = prepare_input(flair, t1, z, spatial_atlas)
                pred_prob = predict_with_tta(model, input_tensor, device, augmentations)
                pred_volume[:, :, z] = pred_prob
            
            # Calculate metrics
            metrics = calculate_3d_metrics(pred_volume, mask, threshold)
            metrics['sample_name'] = sample_path.name
            results.append(metrics)
        
        all_results[tta_name] = results
    
    # Print comparison
    print("\n" + "="*70)
    print("TTA COMPARISON (Threshold 0.7)")
    print("="*70)
    
    print(f"\n{'Method':<25} {'Mean Dice':<12} {'Std':<10} {'Sens':<10} {'Prec':<10} {'Improvement'}")
    print("-"*70)
    
    baseline_dice = np.mean([r['dice'] for r in all_results['Baseline (No TTA)']])
    
    for tta_name, results in all_results.items():
        dices = [r['dice'] for r in results]
        sens = [r['sensitivity'] for r in results]
        precs = [r['precision'] for r in results]
        
        mean_dice = np.mean(dices)
        std_dice = np.std(dices)
        mean_sens = np.mean(sens)
        mean_prec = np.mean(precs)
        
        improvement = mean_dice - baseline_dice
        imp_str = f"+{improvement*100:.2f}%" if improvement > 0 else f"{improvement*100:.2f}%"
        
        print(f"{tta_name:<25} {mean_dice:<12.4f} {std_dice:<10.4f} {mean_sens:<10.4f} {mean_prec:<10.4f} {imp_str}")
    
    # Best configuration
    best_tta = max(all_results.keys(), key=lambda k: np.mean([r['dice'] for r in all_results[k]]))
    best_dice = np.mean([r['dice'] for r in all_results[best_tta]])
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nBest TTA: {best_tta}")
    print(f"Dice: {best_dice:.4f}")
    print(f"Improvement: +{(best_dice - baseline_dice)*100:.2f}%")
    
    if best_dice > 0.77:
        print("\n✅ TTA successfully improved performance!")
    else:
        print("\n⚠️  Improvement limited, may need other methods")


if __name__ == '__main__':
    main()
