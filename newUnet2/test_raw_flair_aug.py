"""
Simple direct visualization - NO dataset, just raw FLAIR + augmentation
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import albumentations as A

# Load raw FLAIR directly
train_dir = Path('d:/VSCode/AIOT_E1/data/wmh/training')
flair_file = list(train_dir.rglob('*/pre/FLAIR.nii.gz'))[0]

print(f"Loading: {flair_file}")
flair_3d = nib.load(flair_file).get_fdata()
print(f"Shape: {flair_3d.shape}")

# Pick a middle slice with brain tissue
mid_z = flair_3d.shape[2] // 2
flair_slice = flair_3d[:, :, mid_z]

# Simple normalization
flair_slice = flair_slice / flair_slice.max()  # Scale to [0, 1]

print(f"Slice shape: {flair_slice.shape}")
print(f"Slice range: [{flair_slice.min():.3f}, {flair_slice.max():.3f}]")

# Create augmentations
transforms = {
    'Original': None,
    'Rotate +20°': A.Rotate(limit=(20, 20), p=1.0),
    'Rotate -20°': A.Rotate(limit=(-20, -20), p=1.0),
    'HorizontalFlip': A.HorizontalFlip(p=1.0),
    'VerticalFlip': A.VerticalFlip(p=1.0),
    'Elastic (alpha=40)': A.ElasticTransform(alpha=40, sigma=8, p=1.0),
    'Brightness (+30%)': A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), p=1.0),
    'Gamma (0.7)': A.RandomGamma(gamma_limit=(70, 70), p=1.0),
}

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (name, transform) in enumerate(transforms.items()):
    if transform is None:
        img_aug = flair_slice
    else:
        result = transform(image=flair_slice)
        img_aug = result['image']
    
    axes[idx].imshow(img_aug, cmap='gray', vmin=0, vmax=1)
    axes[idx].set_title(name, fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle('Raw FLAIR Augmentation Test (No Dataset Processing)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = Path('d:/VSCode/AIOT_E1/TEMP2/raw_flair_augmentation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved to: {output_path}")
print("\n✅ If this looks normal, the problem is in the dataset preprocessing.")
print("✅ If this still looks weird, the problem is in the raw data or augmentation.")
