"""
Visualize V0.3 Data Augmentation Effects
Shows how GaussNoise and other augmentations affect training data
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from dataset import WMHDataset, get_train_transform
import config
import torch

def main():
    print("Loading dataset...")
    dataset = WMHDataset(config.TRAIN_DIR, transform=None)  # No transform first
    
    # Get a sample
    sample_idx = 0  # First sample
    image_raw, mask_raw = dataset[sample_idx]
    
    # Convert to numpy for visualization
    image_np = image_raw.cpu().numpy()  # (7, 224, 224)
    mask_np = mask_raw.cpu().numpy()[0]  # (224, 224)
    
    print(f"Image shape: {image_np.shape}")
    print(f"Channels: CLAHE[t-1], CLAHE[t], CLAHE[t+1], T1, HighPass, Asymmetry, Spatial Atlas")
    
    # Now get augmented version
    dataset.transform = get_train_transform(aug_p_noise=1.0)  # Force GaussNoise to apply
    image_aug, mask_aug = dataset[sample_idx]
    image_aug_np = image_aug.cpu().numpy()
    mask_aug_np = mask_aug.cpu().numpy()[0]
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    channels = ['CLAHE[t-1]', 'CLAHE[t]', 'CLAHE[t+1]', 'T1', 'HighPass', 'Asymmetry', 'Spatial Atlas']
    
    for i in range(7):
        # Original
        ax = plt.subplot(3, 7, i + 1)
        plt.imshow(image_np[i], cmap='gray')
        plt.title(f'{channels[i]}\n(Original)', fontsize=10)
        plt.axis('off')
        
        # Augmented
        ax = plt.subplot(3, 7, i + 8)
        plt.imshow(image_aug_np[i], cmap='gray')
        plt.title(f'{channels[i]}\n(+GaussNoise)', fontsize=10)
        plt.axis('off')
        
        # Difference
        ax = plt.subplot(3, 7, i + 15)
        diff = np.abs(image_aug_np[i] - image_np[i])
        plt.imshow(diff, cmap='hot')
        plt.title(f'Difference\n(Noise)', fontsize=10)
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.suptitle('V0.3 Data Augmentation: GaussNoise Effect on 7 Channels', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v03_gaussnoise_augmentation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    # Additional visualization: Show all augmentation types
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original CLAHE[t]
    axes[0, 0].imshow(image_np[1], cmap='gray')
    axes[0, 0].set_title('Original CLAHE[t]', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Different augmentations
    aug_types = [
        ('GaussNoise', get_train_transform(aug_p_noise=1.0, aug_p_horizontal=0, aug_p_rotate=0, 
                                           aug_p_brightness=0, aug_p_gamma=0, aug_p_elastic=0, aug_p_grid=0)),
        ('HorizontalFlip', get_train_transform(aug_p_horizontal=1.0, aug_p_noise=0, aug_p_rotate=0, 
                                               aug_p_brightness=0, aug_p_gamma=0, aug_p_elastic=0, aug_p_grid=0)),
        ('Rotate', get_train_transform(aug_p_rotate=1.0, aug_p_noise=0, aug_p_horizontal=0, 
                                       aug_p_brightness=0, aug_p_gamma=0, aug_p_elastic=0, aug_p_grid=0)),
        ('Brightness', get_train_transform(aug_p_brightness=1.0, aug_p_noise=0, aug_p_horizontal=0, 
                                          aug_p_rotate=0, aug_p_gamma=0, aug_p_elastic=0, aug_p_grid=0)),
        ('Gamma', get_train_transform(aug_p_gamma=1.0, aug_p_noise=0, aug_p_horizontal=0, 
                                      aug_p_rotate=0, aug_p_brightness=0, aug_p_elastic=0, aug_p_grid=0)),
        ('ElasticTransform', get_train_transform(aug_p_elastic=1.0, aug_p_noise=0, aug_p_horizontal=0, 
                                                 aug_p_rotate=0, aug_p_brightness=0, aug_p_gamma=0, aug_p_grid=0)),
        ('GridDistortion', get_train_transform(aug_p_grid=1.0, aug_p_noise=0, aug_p_horizontal=0, 
                                               aug_p_rotate=0, aug_p_brightness=0, aug_p_gamma=0, aug_p_elastic=0)),
    ]
    
    for idx, (name, transform) in enumerate(aug_types):
        dataset.transform = transform
        img_aug, _ = dataset[sample_idx]
        img_aug_np = img_aug.cpu().numpy()
        
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        axes[row, col].imshow(img_aug_np[1], cmap='gray')  # Show CLAHE[t]
        axes[row, col].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.suptitle('V0.3 Data Augmentation Types (showing CLAHE[t] channel)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = Path('d:/VSCode/AIOT_E1/TEMP2/v03_all_augmentations.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path2}")
    
    print("\n✅ Visualization complete!")
    print(f"\nGaussNoise variance: 0.003")
    print(f"Effect: Adds random Gaussian noise to simulate sensor noise")


if __name__ == '__main__':
    main()
