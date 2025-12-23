"""
诊断脚本：确定正确的左右翻转轴
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import sys

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent / 'newUnet'))
from dataset import normalize_slice, center_crop
import config


def visualize_flip_directions(sample_name='151'):
    """可视化不同翻转方向，让用户确认哪个是左右翻转"""
    
    test_dir = Path(config.TEST_DIR)
    sample_paths = list(test_dir.rglob(f'*/{sample_name}/pre/FLAIR.nii.gz'))
    
    if len(sample_paths) == 0:
        print(f"Sample {sample_name} not found")
        return
    
    sample_path = sample_paths[0].parent.parent
    
    # 加载数据
    flair_raw = nib.load(sample_path / 'pre' / 'FLAIR.nii.gz').get_fdata()
    mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
    
    # 预处理
    flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
    mask = center_crop(mask_raw, config.TARGET_SIZE)
    
    # 找到有WMH的切片
    best_slice = None
    max_wmh = 0
    for z in range(10, flair.shape[2] - 10):
        wmh_count = (mask[:, :, z] == 1).sum()
        if wmh_count > max_wmh:
            max_wmh = wmh_count
            best_slice = z
    
    flair_slice = flair[:, :, best_slice]
    mask_slice = mask[:, :, best_slice]
    
    # 归一化显示
    flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min())
    
    # 创建RGB叠加
    def create_overlay(img, msk):
        rgb = np.stack([img, img, img], axis=-1).copy()
        wmh_mask = (msk == 1)
        rgb[wmh_mask, 0] = 1.0
        rgb[wmh_mask, 1] = 0.0
        rgb[wmh_mask, 2] = 0.0
        return rgb
    
    # 测试不同翻转
    fliplr_result = np.fliplr(flair_slice)  # 翻转列（水平）
    flipud_result = np.flipud(flair_slice)  # 翻转行（垂直）
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'翻转轴诊断 - Sample {sample_name}, Slice {best_slice}\n'
                 f'请观察哪个翻转是"左右镜像"（大脑中线对称）', 
                 fontsize=16, fontweight='bold')
    
    # 第一行：原图对比
    overlay = create_overlay(flair_norm, mask_slice)
    
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title('原图 + WMH标注\n(红色=WMH)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.1, 'Shape: ' + str(flair_slice.shape), 
                    transform=axes[0, 0].transAxes, ha='center', fontsize=10)
    
    # fliplr (水平翻转 - 翻转列)
    axes[0, 1].imshow(np.fliplr(overlay))
    axes[0, 1].set_title('方案A: np.fliplr(原图)\n翻转axis=1（列/第2维）', 
                         fontsize=12, fontweight='bold', color='blue')
    axes[0, 1].axis('off')
    
    # flipud (垂直翻转 - 翻转行)
    axes[0, 2].imshow(np.flipud(overlay))
    axes[0, 2].set_title('方案B: np.flipud(原图)\n翻转axis=0（行/第1维）', 
                         fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # 第二行：不对称图
    axes[1, 0].imshow(flair_norm, cmap='gray')
    axes[1, 0].set_title('原图 (灰度)', fontsize=12)
    axes[1, 0].axis('off')
    
    # fliplr asymmetry
    asym_lr = np.abs(flair_slice - fliplr_result)
    asym_lr_norm = (asym_lr - asym_lr.min()) / (asym_lr.max() - asym_lr.min() + 1e-8)
    im1 = axes[1, 1].imshow(asym_lr_norm, cmap='hot')
    axes[1, 1].set_title('方案A不对称图\nabs(原图 - fliplr)', fontsize=12, color='blue')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    
    # 计算WMH区域的不对称度
    if (mask_slice == 1).sum() > 0:
        wmh_asym_lr = asym_lr_norm[mask_slice == 1].mean()
        normal_asym_lr = asym_lr_norm[mask_slice == 0].mean()
        axes[1, 1].text(0.5, -0.15, f'WMH: {wmh_asym_lr:.3f} | Normal: {normal_asym_lr:.3f}', 
                       transform=axes[1, 1].transAxes, ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # flipud asymmetry
    asym_ud = np.abs(flair_slice - flipud_result)
    asym_ud_norm = (asym_ud - asym_ud.min()) / (asym_ud.max() - asym_ud.min() + 1e-8)
    im2 = axes[1, 2].imshow(asym_ud_norm, cmap='hot')
    axes[1, 2].set_title('方案B不对称图\nabs(原图 - flipud)', fontsize=12, color='green')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    if (mask_slice == 1).sum() > 0:
        wmh_asym_ud = asym_ud_norm[mask_slice == 1].mean()
        normal_asym_ud = asym_ud_norm[mask_slice == 0].mean()
        axes[1, 2].text(0.5, -0.15, f'WMH: {wmh_asym_ud:.3f} | Normal: {normal_asym_ud:.3f}', 
                       transform=axes[1, 2].transAxes, ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # 添加说明
    instruction = """
观察指南：
1. 第一行：观察哪个翻转看起来是"左右镜像"（大脑应该沿中线对称）
2. 第二行：正确的不对称图应该在大脑中线附近值较低（暗），病变区域值高（亮）
3. 比较WMH和Normal的数值，哪个方案的对比度更大？

如果方案A（蓝色）正确 → 使用 np.fliplr() 或 np.flip(axis=1)
如果方案B（绿色）正确 → 使用 np.flipud() 或 np.flip(axis=0)
    """
    
    fig.text(0.5, 0.01, instruction, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2') / f'flip_diagnosis_{sample_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n诊断图已保存: {output_path}")
    print("\n请查看图片并告知：")
    print("  - 哪个方案（A或B）是正确的左右镜像？")
    print("  - 我会据此修正代码")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("生成翻转轴诊断图")
    print("="*70)
    visualize_flip_directions('151')
    print("\n" + "="*70)
