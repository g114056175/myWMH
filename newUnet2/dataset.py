"""
Dataset for WMH segmentation - V0.3
7-channel: CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas
"""

import numpy as np
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


def normalize_slice(img_slice):
    """
    Z-score normalization with numerical stability
    """
    mean = np.mean(img_slice)
    std = np.std(img_slice)
    
    if std == 0 or std < 1e-6:
        return img_slice - mean
    
    normalized = (img_slice - mean) / std
    normalized = np.clip(normalized, -5.0, 5.0)
    
    return normalized


def to_uint8(img):
    """Convert to uint8 format (0-255)"""
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img_norm * 255).astype(np.uint8)


def apply_clahe(img_slice, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to enhance contrast"""
    img_uint8 = to_uint8(img_slice)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8).astype(np.float32) / 255.0


def apply_highpass(img_slice, sigma=2.0):
    """Apply high-pass filter"""
    img_uint8 = to_uint8(img_slice)
    lowpass = cv2.GaussianBlur(img_uint8, (0, 0), sigma)
    highpass = cv2.subtract(img_uint8, lowpass)
    return highpass.astype(np.float32) / 255.0


def generate_spatial_map(shape=(224, 224)):
    """
    Generate spatial coord

inate map (location prior)
    
    Returns a map where center=1 (high weight), edge=0 (low weight)
    This provides strong prior knowledge that WMH tends to appear
    in periventricular and deep white matter (center of brain).
    
    Args:
        shape: (H, W) tuple
    
    Returns:
        spatial_map: (H, W) array, normalized distance from center
    """
    h, w = shape
    
    # Create coordinate grids
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(x, y)
    
    # Calculate Euclidean distance from center (0.5, 0.5)
    center_x, center_y = 0.5, 0.5
    dist = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
    
    # Invert: center=1, edge=0
    # This gives higher weight to central regions (where WMH commonly appears)
    spatial_map = 1.0 - (dist / dist.max())
    
    return spatial_map.astype(np.float32)


def center_crop(img, target_size):
    """Center crop or pad to target size"""
    h, w = img.shape[:2]
    
    # Pad if needed
    if h < target_size or w < target_size:
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        if img.ndim == 2:
            img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), 
                              (pad_w//2, pad_w - pad_w//2)), mode='constant')
        else:  # 3D
            img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), 
                              (pad_w//2, pad_w - pad_w//2), 
                              (0, 0)), mode='constant')
        h, w = img.shape[:2]
    
    # Crop
    crop_h = (h - target_size) // 2
    crop_w = (w - target_size) // 2
    
    if img.ndim == 2:
        return img[crop_h:crop_h + target_size, crop_w:crop_w + target_size]
    else:
        return img[crop_h:crop_h + target_size, crop_w:crop_w + target_size, :]


class WMHDataset(Dataset):
    """
    WMH Segmentation Dataset - V0.3
    
    7-channel input:
    - Ch0-2: CLAHE(FLAIR[t-1, t, t+1])  (CLAHE-enhanced 2.5D)
    - Ch3:   T1[t]                      (tissue contrast)
    - Ch4:   HighPass(FLAIR[t])         (edge detection)
    - Ch5:   Asymmetry(FLAIR[t])        (left-right asymmetry)
    - Ch6:   Spatial Atlas              (population-based prior)
    """
    
    def __init__(self, data_root, target_size=224, transform=None, 
                 clahe_clip_limit=2.0, highpass_sigma=2.0):
        """
        Args:
            data_root: Path to training or test directory
            target_size: Size for center crop (224)
            transform: Albumentations transform pipeline
            clahe_clip_limit: CLAHE clip limit
            highpass_sigma: High-pass filter sigma
        """
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.transform = transform
        self.clahe_clip_limit = clahe_clip_limit
        self.highpass_sigma = highpass_sigma
        
        # Load spatial atlas prior (V0.3: population-based probability map)
        atlas_path = Path(__file__).parent / 'spatial_atlas_prior.npy'
        if not atlas_path.exists():
            raise FileNotFoundError(f"Spatial atlas not found: {atlas_path}")
        self.spatial_atlas = np.load(atlas_path)
        print(f"Loaded spatial atlas: shape={self.spatial_atlas.shape}, range=[{self.spatial_atlas.min():.3f}, {self.spatial_atlas.max():.3f}]")
        
        # Find all samples
        self.samples = self._find_samples()
        
        # Preload all 3D volumes to memory
        print(f"Preloading {len(self.samples)} 3D volumes (FLAIR + T1) to memory...")
        self.flair_volumes = []
        self.t1_volumes = []
        self.mask_volumes = []
        
        for sample_path in self.samples:
            # Load FLAIR
            flair = nib.load(sample_path / 'pre' / 'FLAIR.nii.gz').get_fdata()
            flair_normalized = normalize_slice(flair)
            flair_cropped = center_crop(flair_normalized, target_size)
            self.flair_volumes.append(flair_cropped)
            
            # Load T1
            t1 = nib.load(sample_path / 'pre' / 'T1.nii.gz').get_fdata()
            t1_normalized = normalize_slice(t1)
            t1_cropped = center_crop(t1_normalized, target_size)
            self.t1_volumes.append(t1_cropped)
            
            # Load mask
            mask = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
            mask_cropped = center_crop(mask, target_size)
            self.mask_volumes.append(mask_cropped)
        
        # Create index: list of (volume_idx, slice_idx)
        self.slice_indices = []
        for vol_idx, flair in enumerate(self.flair_volumes):
            n_slices = flair.shape[2]
            # Skip first 2 and last 2 slices
            for slice_idx in range(2, n_slices - 2):
                self.slice_indices.append((vol_idx, slice_idx))
        
        print(f"Loaded {len(self.samples)} volumes, {len(self.slice_indices)} slices")
    
    def _find_samples(self):
        """Find all sample directories"""
        samples = []
        flair_files = list(self.data_root.rglob('*/pre/FLAIR.nii.gz'))
        
        for flair_file in flair_files:
            sample_path = Path(flair_file).parent.parent
            # Check if both T1 and mask exist
            if (sample_path / 'pre' / 'T1.nii.gz').exists() and \
               (sample_path / 'wmh.nii.gz').exists():
                samples.append(sample_path)
        
        if len(samples) == 0:
            raise FileNotFoundError(f"No samples found in {self.data_root}")
        
        return samples
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_indices[idx]
        
        flair_3d = self.flair_volumes[vol_idx]
        t1_3d = self.t1_volumes[vol_idx]
        mask_3d = self.mask_volumes[vol_idx]
        
        # Extract 2.5D slices
        flair_t_minus_1 = flair_3d[:, :, slice_idx - 1]
        flair_t = flair_3d[:, :, slice_idx]
        flair_t_plus_1 = flair_3d[:, :, slice_idx + 1]
        t1_t = t1_3d[:, :, slice_idx]
        
        # V0.3: Apply CLAHE to all 3 FLAIR slices (instead of just current)
        clahe_t_minus_1 = apply_clahe(flair_t_minus_1, self.clahe_clip_limit)
        clahe_t = apply_clahe(flair_t, self.clahe_clip_limit)
        clahe_t_plus_1 = apply_clahe(flair_t_plus_1, self.clahe_clip_limit)
        
        # Apply high-pass filter (only on current slice)
        highpass_t = apply_highpass(flair_t, self.highpass_sigma)
        
        # V0.3: Calculate Asymmetry feature (left-right asymmetry)
        # Use flipud (axis=0) for correct left-right flip
        asymmetry = np.abs(flair_t - np.flipud(flair_t))
        asymmetry = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)
        asymmetry = asymmetry.astype(np.float32)
        
        # Normalize T1 to [0, 1]
        t1_t_norm = to_uint8(t1_t).astype(np.float32) / 255.0
        
        # V0.3: Stack to 7 channels (H, W, 7)
        # Order: CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas
        image = np.stack([
            clahe_t_minus_1,        # ch0: CLAHE(FLAIR[t-1])
            clahe_t,                # ch1: CLAHE(FLAIR[t])
            clahe_t_plus_1,         # ch2: CLAHE(FLAIR[t+1])
            t1_t_norm,              # ch3: T1
            highpass_t,             # ch4: HighPass
            asymmetry,              # ch5: Asymmetry (NEW)
            self.spatial_atlas,     # ch6: Spatial Atlas (NEW)
        ], axis=-1).astype(np.float32)
        
        # Mask
        mask = mask_3d[:, :, slice_idx].astype(np.float32)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return image, mask


def get_train_transform(aug_p_vertical=0.5, aug_p_rotate=0.7,
                       aug_p_brightness=0.7, aug_p_gamma=0.5,
                       aug_p_elastic=0.2, aug_p_grid=0.3, aug_p_noise=0.0):
    """
    Get training augmentation pipeline - V0.3 FINAL CORRECTED
    
    修正策略:
    - 移除HorizontalFlip（与Asymmetry特征冲突！）
    - 降低Brightness到±15%（避免破坏WMH对比度）
    - 降低Elastic到alpha=30, 20%（轻度变形）
    - 保守增强，保护特征
    """
    return A.Compose([
        # 几何增强（V0.3修正）
        # ❌ 移除 HorizontalFlip - 与Asymmetry特征（np.flipud）冲突
        A.VerticalFlip(p=aug_p_vertical),      # 0.5 - 上下翻转
        A.Rotate(limit=20, p=aug_p_rotate),     # ±20°, 70%
        A.ElasticTransform(alpha=30, sigma=6, p=aug_p_elastic),  # ⚠️ 降低：alpha=30, 20%
        A.GridDistortion(p=aug_p_grid),        # 0.3 - 降低
        
        # 强度增强（极保守）
        A.RandomBrightnessContrast(
            brightness_limit=0.1,   # ⚠️ 最终：±10% (非常保守)
            contrast_limit=0.1,     # ⚠️ 最终：±10%
            p=aug_p_brightness
        ),
        A.RandomGamma(gamma_limit=(90, 110), p=aug_p_gamma),  # ⚠️ 最终：90-110 (±10%)
        # A.GaussNoise(var_limit=0.003, p=aug_p_noise),  # 移除：担心破坏图像结构
        
        ToTensorV2(),
    ])


def get_val_transform():
    """Get validation transform (no augmentation)"""
    return A.Compose([
        ToTensorV2(),
    ])


if __name__ == '__main__':
    # Test dataset
    from torch.utils.data import DataLoader
    
    data_root = 'd:/VSCode/AIOT_E1/data/wmh/training'
    dataset = WMHDataset(data_root, target_size=224, transform=get_train_transform())
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")  # Should be (7, 224, 224)
    print(f"Mask shape: {mask.shape}")    # Should be (1, 224, 224)
    print(f"Image dtype: {image.dtype}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask)}")
    
    # Check each channel
    print("\nChannel statistics:")
    ch_names = ['CLAHE[t-1]', 'CLAHE[t]', 'CLAHE[t+1]', 'T1', 
                'HighPass', 'Asymmetry', 'Spatial Atlas']
    for i in range(7):  # V0.3: 7 channels
        ch_mean = image[i].mean()
        ch_std = image[i].std()
        ch_min = image[i].min()
        ch_max = image[i].max()
        print(f"  Ch{i} ({ch_names[i]}): mean={ch_mean:.3f}, std={ch_std:.3f}, range=[{ch_min:.3f}, {ch_max:.3f}]")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_images, batch_masks = next(iter(dataloader))
    print(f"\nBatch images shape: {batch_images.shape}")  # (4, 7, 224, 224)
    print(f"Batch masks shape: {batch_masks.shape}")      # (4, 1, 224, 224)
    
    print("\n✓ Dataset test passed!")
