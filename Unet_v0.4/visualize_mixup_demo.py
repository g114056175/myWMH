"""
MixUp可视化演示
展示5组不同λ值的MixUp效果
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('d:/VSCode/AIOT_E1/Unet_v0.4')

from dataset import normalize_slice, center_crop, apply_clahe

# 设置
data_dir = Path('d:/VSCode/AIOT_E1/data/wmh/training')
output_dir = Path('d:/VSCode/AIOT_E1/TEMP2')
output_dir.mkdir(exist_ok=True)

# 获取所有样本（递归查找所有包含pre/FLAIR.nii.gz的目录）
all_samples = []
for center_dir in data_dir.iterdir():
    if center_dir.is_dir():
        for sample_dir in center_dir.iterdir():
            if sample_dir.is_dir() and (sample_dir / 'pre' / 'FLAIR.nii.gz').exists():
                all_samples.append(sample_dir)

samples = sorted(all_samples)
print("="*70)
print("MixUp可视化演示")
print("="*70)
print(f"找到 {len(samples)} 个样本\n")

# 随机选择10个样本（用于5组）
np.random.seed(42)
selected = np.random.choice(len(samples), size=min(10, len(samples)), replace=len(samples)<10)

# 5组不同的λ值
lambdas = [0.2, 0.35, 0.5, 0.65, 0.8]

def load_and_prepare(sample_dir, slice_idx=None):
    """加载并预处理FLAIR和mask"""
    flair_path = sample_dir / 'pre' / 'FLAIR.nii.gz'
    mask_path = sample_dir / 'wmh.nii.gz'
    
    flair_3d = nib.load(flair_path).get_fdata()
    mask_3d = nib.load(mask_path).get_fdata()
    
    # 如果没指定slice，选择有WMH的中间层
    if slice_idx is None:
        # 找有WMH的层
        wmh_slices = [i for i in range(flair_3d.shape[2]) if mask_3d[:,:,i].sum() > 100]
        if wmh_slices:
            slice_idx = wmh_slices[len(wmh_slices)//2]
        else:
            slice_idx = flair_3d.shape[2] // 2
    
    # 提取slice
    flair_slice = flair_3d[:,:,slice_idx]
    mask_slice = mask_3d[:,:,slice_idx]
    
    # 预处理
    flair_norm = normalize_slice(flair_slice)
    flair_crop = center_crop(flair_norm, 224)
    mask_crop = center_crop(mask_slice, 224)
    
    # CLAHE
    flair_clahe = apply_clahe(flair_crop, clip_limit=2.0)
    
    return flair_clahe, mask_crop

# 生成5组MixUp示例
for i, lambda_ in enumerate(lambdas):
    print(f"\n生成示例 {i+1}/5 (λ={lambda_:.2f})...")
    
    # 选择两个样本
    sample_a = samples[selected[i*2]]
    sample_b = samples[selected[i*2 + 1]]
    
    # 加载
    flair_a, mask_a = load_and_prepare(sample_a)
    flair_b, mask_b = load_and_prepare(sample_b)
    
    # MixUp
    mixed_flair = lambda_ * flair_a + (1 - lambda_) * flair_b
    mixed_mask = lambda_ * mask_a + (1 - lambda_) * mask_b
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：FLAIR图像
    # 原图A
    axes[0, 0].imshow(flair_a, cmap='gray')
    axes[0, 0].set_title(f'Sample A (FLAIR)\n{sample_a.name}', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 原图B
    axes[0, 1].imshow(flair_b, cmap='gray')
    axes[0, 1].set_title(f'Sample B (FLAIR)\n{sample_b.name}', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 混合图
    axes[0, 2].imshow(mixed_flair, cmap='gray')
    axes[0, 2].set_title(f'Mixed FLAIR\nλ={lambda_:.2f} × A + {1-lambda_:.2f} × B', 
                         fontsize=12, fontweight='bold', color='red')
    axes[0, 2].axis('off')
    
    # 第二行：叠加WMH contour
    # 原图A + mask
    axes[1, 0].imshow(flair_a, cmap='gray')
    if mask_a.max() > 0:
        axes[1, 0].contour(mask_a.astype(bool), colors='red', linewidths=2, levels=[0.5])
    axes[1, 0].set_title(f'A with WMH\n(WMH pixels: {int(mask_a.sum())})', 
                         fontsize=11)
    axes[1, 0].axis('off')
    
    # 原图B + mask
    axes[1, 1].imshow(flair_b, cmap='gray')
    if mask_b.max() > 0:
        axes[1, 1].contour(mask_b.astype(bool), colors='blue', linewidths=2, levels=[0.5])
    axes[1, 1].set_title(f'B with WMH\n(WMH pixels: {int(mask_b.sum())})', 
                         fontsize=11)
    axes[1, 1].axis('off')
    
    # 混合图 + 软mask
    axes[1, 2].imshow(mixed_flair, cmap='gray')
    # 显示软mask作为半透明overlay
    if mixed_mask.max() > 0:
        overlay = np.zeros((*mixed_mask.shape, 4))
        overlay[..., 0] = 1.0  # Red
        overlay[..., 3] = mixed_mask / mixed_mask.max() * 0.5  # Alpha
        axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(
        f'Mixed with Soft Label\n(Soft mask range: {mixed_mask.min():.2f}-{mixed_mask.max():.2f})', 
        fontsize=11, color='red'
    )
    axes[1, 2].axis('off')
    
    # 总标题
    fig.suptitle(
        f'MixUp Demonstration #{i+1}\nλ = {lambda_:.2f} (Mix ratio: {lambda_*100:.0f}% A + {(1-lambda_)*100:.0f}% B)',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 添加说明
    fig.text(0.5, 0.02, 
             'Top row: FLAIR images | Bottom row: FLAIR with WMH masks\n'
             'Red contours = Sample A WMH | Blue contours = Sample B WMH | '
             'Red overlay = Mixed soft label',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 保存
    output_path = output_dir / f'mixup_demo_{i+1}_lambda_{lambda_:.2f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")

# 创建一个汇总图
print(f"\n生成汇总对比图...")

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for i, lambda_ in enumerate(lambdas):
    # 重新加载一组
    sample_a = samples[selected[i*2]]
    sample_b = samples[selected[i*2 + 1]]
    
    flair_a, mask_a = load_and_prepare(sample_a)
    flair_b, mask_b = load_and_prepare(sample_b)
    
    mixed_flair = lambda_ * flair_a + (1 - lambda_) * flair_b
    
    axes[i].imshow(mixed_flair, cmap='gray')
    axes[i].set_title(f'λ = {lambda_:.2f}\n{lambda_*100:.0f}% A + {(1-lambda_)*100:.0f}% B',
                     fontsize=12, fontweight='bold')
    axes[i].axis('off')

fig.suptitle('MixUp with Different λ Values - Visual Comparison',
             fontsize=16, fontweight='bold')

summary_path = output_dir / 'mixup_summary_comparison.png'
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {summary_path.name}")

print("\n" + "="*70)
print("✓ 完成！生成了6张图片:")
print("  - 5张详细示例 (不同λ值)")
print("  - 1张汇总对比图")
print(f"\n保存位置: {output_dir}")
print("="*70)
