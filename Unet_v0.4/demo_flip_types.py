"""
清晰展示两种翻转的效果
让用户确认哪种是需要的
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
sys.path.append('d:/VSCode/AIOT_E1/Unet_v0.4')

from dataset import normalize_slice, center_crop

# 加载一个真实的FLAIR样本
data_dir = Path('d:/VSCode/AIOT_E1/data/wmh/training')
flair_files = list(data_dir.rglob('*/pre/FLAIR.nii.gz'))

print(f"Loading sample from: {flair_files[0]}")
flair_3d = nib.load(flair_files[0]).get_fdata()
flair_normalized = normalize_slice(flair_3d)
flair_cropped = center_crop(flair_normalized, 224)

# 选择中间层
mid_z = flair_cropped.shape[2] // 2
flair_slice = flair_cropped[:, :, mid_z]

# 归一化到[0, 1]以便显示
flair_display = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min())

# 创建两种翻转
flip_ud = np.flipud(flair_display)  # up-down 上下翻转 (axis=0)
flip_lr = np.fliplr(flair_display)  # left-right 左右翻转 (axis=1)

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# 原始图像
axes[0, 0].imshow(flair_display, cmap='gray', origin='upper')
axes[0, 0].set_title('Original FLAIR\n(原始图像)', fontsize=16, fontweight='bold')
axes[0, 0].axis('off')

# 添加坐标标注
axes[0, 0].text(10, 20, 'Left (左)', color='red', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 0].text(200, 20, 'Right (右)', color='red', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 0].text(10, 210, 'Bottom (下)', color='blue', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 0].text(200, 210, 'Bottom (下)', color='blue', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Asymmetry特征计算
asymmetry = np.abs(flair_display - np.flipud(flair_display))
axes[0, 1].imshow(asymmetry, cmap='hot', origin='upper')
axes[0, 1].set_title('Asymmetry = |Original - np.flipud(Original)|\n(使用np.flipud检测左右不对称)', 
                     fontsize=16, fontweight='bold')
axes[0, 1].axis('off')

# np.flipud (上下翻转)
axes[1, 0].imshow(flip_ud, cmap='gray', origin='upper')
axes[1, 0].set_title('np.flipud(Original)\n= 上下翻转 (flip axis=0)\n= Albumentations.VerticalFlip', 
                     fontsize=16, fontweight='bold', color='green')
axes[1, 0].axis('off')

# 标注翻转后的位置
axes[1, 0].text(10, 20, 'Left (左)', color='red', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 0].text(200, 20, 'Right (右)', color='red', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 0].text(10, 210, 'Top (上)← 原来的Bottom', color='blue', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# np.fliplr (左右翻转)
axes[1, 1].imshow(flip_lr, cmap='gray', origin='upper')
axes[1, 1].set_title('np.fliplr(Original)\n= 左右翻转 (flip axis=1)\n= Albumentations.HorizontalFlip', 
                     fontsize=16, fontweight='bold', color='red')
axes[1, 1].axis('off')

# 标注翻转后的位置
axes[1, 1].text(10, 20, 'Right (右) ← 原来的Left', color='red', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
axes[1, 1].text(200, 20, 'Left (左) ← 原来的Right', color='red', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
axes[1, 1].text(10, 210, 'Bottom (下)', color='blue', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('翻转类型对比 - 请确认哪种与Asymmetry冲突', fontsize=18, fontweight='bold', y=0.98)

# 添加说明文字
explanation = """
关键说明:
1. Asymmetry特征使用 np.flipud (上下翻转) 来检测左右不对称性
2. ✅ np.flipud / A.VerticalFlip: 安全，不破坏Asymmetry
3. ❌ np.fliplr / A.HorizontalFlip: 会破坏Asymmetry (左右位置对调)

当前V0.4配置:
- ✅ 保留 A.VerticalFlip (上下翻转)
- ❌ 移除 A.HorizontalFlip (左右翻转)
"""

plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=12, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

output_path = Path('d:/VSCode/AIOT_E1/TEMP2/flip_comparison_clear.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\n✓ Saved to: {output_path}")
print("\n" + "="*70)
print("请查看图像确认:")
print("="*70)
print("1. np.flipud (上下翻转) = A.VerticalFlip - 是否正确?")
print("2. np.fliplr (左右翻转) = A.HorizontalFlip - 这个会破坏Asymmetry")
print("3. Asymmetry使用哪种翻转? 答案: np.flipud")
print("="*70)

# 额外显示：旋转后的效果
import albumentations as A

transform_vflip = A.VerticalFlip(p=1.0)
transform_hflip = A.HorizontalFlip(p=1.0)

# 需要3通道才能用albumentations
flair_3ch = np.stack([flair_display]*3, axis=-1)

result_vflip = transform_vflip(image=flair_3ch)
result_hflip = transform_hflip(image=flair_3ch)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

axes2[0].imshow(flair_display, cmap='gray')
axes2[0].set_title('Original', fontsize=14, fontweight='bold')
axes2[0].axis('off')

axes2[1].imshow(result_vflip['image'][:,:,0], cmap='gray')
axes2[1].set_title('A.VerticalFlip\n(上下翻转)\n✅ 安全', fontsize=14, fontweight='bold', color='green')
axes2[1].axis('off')

axes2[2].imshow(result_hflip['image'][:,:,0], cmap='gray')
axes2[2].set_title('A.HorizontalFlip\n(左右翻转)\n❌ 与Asymmetry冲突', fontsize=14, fontweight='bold', color='red')
axes2[2].axis('off')

plt.suptitle('Albumentations 翻转效果验证', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path2 = Path('d:/VSCode/AIOT_E1/TEMP2/albumentations_flip_demo.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Also saved: {output_path2}")
