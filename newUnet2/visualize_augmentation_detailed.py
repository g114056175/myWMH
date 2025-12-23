"""
Visualize specific augmentation types with annotations
Shows Rotate, ElasticTransform, and HorizontalFlip clearly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from dataset import WMHDataset
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    print("Loading dataset...")
    dataset = WMHDataset(config.TRAIN_DIR, transform=None)
    
    # Get a sample
    image_raw, mask_raw = dataset[0]
    image_np = image_raw.cpu().numpy()
    
    # Use CLAHE[t] for visualization (most representative)
    img = image_np[1]  # CLAHE[t]
    
    # Create different augmentation transforms
    transforms = {
        'Original': None,
        'Rotate ±30°\n(p=0.95)': A.Compose([
            A.Rotate(limit=30, p=1.0),
            ToTensorV2()
        ]),
        'HorizontalFlip\n左右翻转 (p=0.9)': A.Compose([
            A.HorizontalFlip(p=1.0),
            ToTensorV2()
        ]),
        'VerticalFlip\n上下翻转 (p=0.7)': A.Compose([
            A.VerticalFlip(p=1.0),
            ToTensorV2()
        ]),
        'ElasticTransform\n弹性变形 (p=0.8)\nalpha=120, sigma=10': A.Compose([
            A.ElasticTransform(alpha=120, sigma=10, p=1.0),
            ToTensorV2()
        ]),
        'GridDistortion\n网格扭曲 (p=0.5)': A.Compose([
            A.GridDistortion(p=1.0),
            ToTensorV2()
        ]),
        'Brightness/Contrast\n亮度对比度 (p=0.9)': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            ToTensorV2()
        ]),
        'Gamma\nGamma变换 (p=0.7)': A.Compose([
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ToTensorV2()
        ]),
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (name, transform) in enumerate(transforms.items()):
        if transform is None:
            # Original
            axes[idx].imshow(img, cmap='gray')
        else:
            # Apply transform
            # Need to create a 3D array for albumentations
            img_3d = np.stack([img] * 3, axis=-1)  # (H, W, 3)
            augmented = transform(image=img_3d)
            img_aug = augmented['image'][0].cpu().numpy()  # Take first channel
            axes[idx].imshow(img_aug, cmap='gray')
        
        axes[idx].set_title(name, fontsize=14, fontweight='bold', pad=10)
        axes[idx].axis('off')
        
        # Add border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor('blue' if idx == 0 else 'gray')
            spine.set_linewidth(3 if idx == 0 else 1)
    
    plt.suptitle('V0.3 数据增强详解（示例：CLAHE[t]通道）', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v03_augmentation_explained.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    # Create a second figure showing the same image with multiple augmentations
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    
    # Show Rotate at different angles
    angles = [-30, -15, 0, 15, 30]
    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, angle in enumerate(angles):
        transform = A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0),  # Fixed angle
            ToTensorV2()
        ])
        img_3d = np.stack([img] * 3, axis=-1)
        augmented = transform(image=img_3d)
        img_aug = augmented['image'][0].cpu().numpy()
        
        axes3[idx].imshow(img_aug, cmap='gray')
        axes3[idx].set_title(f'{angle:+d}°', fontsize=14, fontweight='bold')
        axes3[idx].axis('off')
    
    plt.suptitle('Rotate增强示例：不同角度（V0.3: ±30°范围）', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = Path('d:/VSCode/AIOT_E1/TEMP2/v03_rotate_angles.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path2}")
    
    print("\n" + "="*60)
    print("V0.3 数据增强总结")
    print("="*60)
    print("几何变换（保持WMH结构）：")
    print("  • Rotate:     ±30° (95%概率)")
    print("  • HFlip:      左右翻转 (90%概率)")
    print("  • VFlip:      上下翻转 (70%概率)")
    print("  • Elastic:    弹性变形 alpha=120 (80%概率)")
    print("  • Grid:       网格扭曲 (50%概率)")
    print("")
    print("强度变换（改变亮度/对比度）：")
    print("  • Brightness: ±30% (90%概率)")
    print("  • Gamma:      70-130 (70%概率)")
    print("")
    print("❌ 未使用: GaussNoise (担心破坏结构)")
    print("="*60)
    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
