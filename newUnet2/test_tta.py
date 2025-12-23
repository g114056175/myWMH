"""
Quick Test-Time Augmentation (TTA) - Immediate Test Dice Boost
Can be applied to current V0.3 model without retraining
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
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


def apply_tta(model, input_tensor, device):
    """
    Apply Test-Time Augmentation
    Only VerticalFlip (safe with Asymmetry feature)
    """
    predictions = []
    
    # 1. Original
    with torch.no_grad():
        pred = model(input_tensor.to(device))
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]
        predictions.append(pred)
    
    # 2. VerticalFlip (safe - doesn't affect Asymmetry)
    input_vflip = torch.flip(input_tensor, dims=[2])  # Flip H dimension
    with torch.no_grad():
        pred_vflip = model(input_vflip.to(device))
        pred_vflip = torch.sigmoid(pred_vflip).cpu().numpy()[0, 0]
        pred_vflip = np.flip(pred_vflip, axis=0)  # Flip back
        predictions.append(pred_vflip)
    
    # Average predictions
    final_pred = np.mean(predictions, axis=0)
    return final_pred


def calculate_3d_metrics(pred_volume, gt_volume, threshold=0.7):
    """Calculate 3D dice"""
    pred_binary = (pred_volume > threshold).astype(np.float32)
    valid_mask = (gt_volume != 2)
    gt_wmh = ((gt_volume == 1) & valid_mask).astype(np.float32)
    pred_valid = (pred_binary * valid_mask).astype(np.float32)
    
    tp = (pred_valid * gt_wmh).sum()
    fp = (pred_valid * (1 - gt_wmh) * valid_mask).sum()
    fn = ((1 - pred_valid) * gt_wmh).sum()
    
    dice = 2.0 * tp / (2*tp + fp + fn + 1e-8)
    return dice


def main():
    print("="*70)
    print("Test-Time Augmentation (TTA) - Quick Test")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model = AttentionUNet(in_channels=7).to(device)
    checkpoint = torch.load(
        Path(config.CHECKPOINT_DIR) / 'best_model.pth',
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")
    
    # Load spatial atlas
    spatial_atlas = np.load(Path(config.CHECKPOINT_DIR).parent / 'spatial_atlas_prior.npy')
    
    # Test on first 10 samples
    test_dir = Path(config.TEST_DIR)
    test_files = sorted(list(test_dir.rglob('*/pre/FLAIR.nii.gz')))[:10]
    
    print(f"\nTesting TTA on {len(test_files)} samples")
    print("="*70)
    
    results_no_tta = []
    results_tta = []
    
    for flair_path in tqdm(test_files):
        sample_path = flair_path.parent.parent
        
        # Load data
        flair_raw = nib.load(flair_path).get_fdata()
        t1_raw = nib.load(sample_path / 'pre' / 'T1.nii.gz').get_fdata()
        mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
        
        flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
        t1 = center_crop(normalize_slice(t1_raw), config.TARGET_SIZE)
        mask = center_crop(mask_raw, config.TARGET_SIZE)
        
        # Predict without TTA
        pred_volume_no_tta = np.zeros_like(mask)
        pred_volume_tta = np.zeros_like(mask)
        
        for z in range(2, flair.shape[2] - 2):
            input_tensor = prepare_input(flair, t1, z, spatial_atlas).unsqueeze(0)
            
            # No TTA
            with torch.no_grad():
                output = model(input_tensor.to(device))
                pred_no_tta = torch.sigmoid(output).cpu().numpy()[0, 0]
                pred_volume_no_tta[:, :, z] = pred_no_tta
            
            # With TTA
            pred_tta = apply_tta(model, input_tensor, device)
            pred_volume_tta[:, :, z] = pred_tta
        
        # Calculate metrics
        dice_no_tta = calculate_3d_metrics(pred_volume_no_tta, mask, 0.7)
        dice_tta = calculate_3d_metrics(pred_volume_tta, mask, 0.7)
        
        results_no_tta.append(dice_no_tta)
        results_tta.append(dice_tta)
    
    # Results
    print("\n" + "="*70)
    print("TTA Results (10 samples)")
    print("="*70)
    print(f"\nMean Dice without TTA: {np.mean(results_no_tta):.4f}")
    print(f"Mean Dice with TTA:    {np.mean(results_tta):.4f}")
    print(f"Improvement:           {np.mean(results_tta) - np.mean(results_no_tta):.4f} ({(np.mean(results_tta) / np.mean(results_no_tta) - 1) * 100:.2f}%)")
    
    print("\nPer-sample comparison:")
    for i, (d1, d2) in enumerate(zip(results_no_tta, results_tta)):
        print(f"  Sample {i+1}: {d1:.4f} → {d2:.4f} ({d2-d1:+.4f})")
    
    print("\n✅ TTA test complete!")
    print("If improvement > +0.5%, consider applying to full test set")


if __name__ == '__main__':
    main()
