"""
CORRECTED Visualization - Shows proper augmentation on brain MRI
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from dataset import WMHDataset
import config
import albumentations as A

def main():
    print("Loading dataset...")
    dataset = WMHDataset(config.TRAIN_DIR, transform=None)
    
    # Get a sample
    image_raw, mask_raw = dataset[0]
    image_np = image_raw.cpu().numpy()
    
    # Use CLAHE[t] for visualization (most representative)
    img = image_np[1]  # CLAHE[t], shape (224, 224)
    
    print(f"Original image shape: {img.shape}")
    print(f"Original image range: [{img.min():.3f}, {img.max():.3f}]")
    
    # Create different augmentation transforms WITHOUT ToTensorV2
    transforms = {
        'Original': None,
        'Rotate ±20°': A.Rotate(limit=20, p=1.0),
        'HorizontalFlip\n(左右翻转)': A.HorizontalFlip(p=1.0),
        'VerticalFlip\n(上下翻转)': A.VerticalFlip(p=1.0),
        'ElasticTransform\n(alpha=40)': A.ElasticTransform(alpha=40, sigma=8, p=1.0),
        'GridDistortion': A.GridDistortion(p=1.0),
        'Brightness/Contrast': A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=1.0
        ),
        'Gamma': A.RandomGamma(gamma_limit=(70, 130), p=1.0),
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (name, transform) in enumerate(transforms.items()):
        if transform is None:
            # Original
            axes[idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            # Apply transform - CORRECTED METHOD
            augmented = transform(image=img)  # Directly use 2D image
            img_aug = augmented['image']
            axes[idx].imshow(img_aug, cmap='gray', vmin=0, vmax=1)
        
        axes[idx].set_title(name, fontsize=14, fontweight='bold', pad=10)
        axes[idx].axis('off')
        
        # Add border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor('blue' if idx == 0 else 'gray')
            spine.set_linewidth(3 if idx == 0 else 1)
    
    plt.suptitle('V0.3 数据增强效果 - 已修正 (CLAHE通道)', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v03_augmentation_CORRECTED.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    # Show different rotation angles
    angles = [-20, -10, 0, 10, 20]
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, angle in enumerate(angles):
        if angle == 0:
            img_aug = img
        else:
            transform = A.Rotate(limit=(angle, angle), p=1.0)
            augmented = transform(image=img)
            img_aug = augmented['image']
        
        axes2[idx].imshow(img_aug, cmap='gray', vmin=0, vmax=1)
        axes2[idx].set_title(f'{angle:+d}°', fontsize=14, fontweight='bold')
        axes2[idx].axis('off')
    
    plt.suptitle('Rotate增强示例：V0.3使用±20°范围', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = Path('d:/VSCode/AIOT_E1/TEMP2/v03_rotate_CORRECTED.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path2}")
    
    print("\n" + "="*60)
    print("✅ 修正后的可视化完成！")
    print("="*60)
    print("现在应该可以看到正确的大脑结构")
    print("之前的问题：ToTensorV2后取错了维度")
    print("修正方法：直接对2D图像应用albumentations")


if __name__ == '__main__':
    main()
