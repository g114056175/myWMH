"""
测试左右对称不对称特征提取 (修正版)
使用 np.flipud (翻转axis=0/第一维) - 正确的左右镜像
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import sys

# Fix Chinese font
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent / 'newUnet'))
from dataset import normalize_slice, center_crop
import config


def compute_asymmetry_map(flair_slice):
    """
    计算左右不对称特征图
    使用 flipud (axis=0) - 正确的左右镜像
    
    Args:
        flair_slice: 2D FLAIR slice
    
    Returns:
        asymmetry_map: abs(original - flipped)
    """
    # ✅ 修正：使用 flipud (翻转第一维/axis=0)
    flipped = np.flipud(flair_slice)
    
    # 计算差异的绝对值
    asymmetry = np.abs(flair_slice - flipped)
    
    # 归一化到 [0, 1]
    asymmetry_norm = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)
    
    return asymmetry_norm


def create_overlay(flair, mask):
    """创建FLAIR + GT的叠加图"""
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    
    # 创建RGB图像
    rgb = np.stack([flair_norm, flair_norm, flair_norm], axis=-1).copy()
    
    # WMH标记为红色
    wmh_mask = (mask == 1)
    rgb[wmh_mask, 0] = 1.0
    rgb[wmh_mask, 1] = 0.0
    rgb[wmh_mask, 2] = 0.0
    
    # Don't care标记为绿色
    dont_care_mask = (mask == 2)
    rgb[dont_care_mask, 0] = 0.0
    rgb[dont_care_mask, 1] = 1.0
    rgb[dont_care_mask, 2] = 0.0
    
    return rgb


def visualize_asymmetry(sample_name, flair_3d, mask_3d, output_dir):
    """
    可视化一个样本的不对称特征
    """
    # 找到有WMH的最佳切片
    best_slice = None
    max_wmh = 0
    for z in range(10, flair_3d.shape[2] - 10):
        wmh_count = (mask_3d[:, :, z] == 1).sum()
        if wmh_count > max_wmh:
            max_wmh = wmh_count
            best_slice = z
    
    if best_slice is None or max_wmh == 0:
        best_slice = flair_3d.shape[2] // 2
    
    # 提取切片
    flair_slice = flair_3d[:, :, best_slice]
    mask_slice = mask_3d[:, :, best_slice]
    
    # 计算不对称图
    asymmetry_map = compute_asymmetry_map(flair_slice)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Left-Right Asymmetry Feature - Sample {sample_name}\nSlice {best_slice} | Using flipud(axis=0)', 
                 fontsize=14, fontweight='bold')
    
    # 1. FLAIR原图
    ax = axes[0]
    ax.imshow(flair_slice, cmap='gray')
    ax.set_title('FLAIR Original', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 2. FLAIR + GT
    ax = axes[1]
    rgb_overlay = create_overlay(flair_slice, mask_slice)
    ax.imshow(rgb_overlay)
    ax.set_title('FLAIR + Ground Truth\n(Red=WMH, Green=DontCare)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 3. 不对称特征图 (灰阶)
    ax = axes[2]
    im = ax.imshow(asymmetry_map, cmap='gray')  # ✅ 改用灰阶
    ax.set_title('Asymmetry Map (Grayscale)\nabs(FLAIR - flipud(FLAIR))', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 统计信息
    wmh_pixels = (mask_slice == 1).sum()
    
    # 计算WMH区域的平均不对称度
    if wmh_pixels > 0:
        wmh_asymmetry = asymmetry_map[mask_slice == 1].mean()
    else:
        wmh_asymmetry = 0
    
    # 计算非WMH区域的平均不对称度
    normal_mask = (mask_slice == 0)
    if normal_mask.sum() > 0:
        normal_asymmetry = asymmetry_map[normal_mask].mean()
    else:
        normal_asymmetry = 0
    
    # 对比度（WMH vs Normal的不对称度比值）
    if normal_asymmetry > 0:
        contrast_ratio = wmh_asymmetry / normal_asymmetry
    else:
        contrast_ratio = 0
    
    stats_text = (f"WMH Pixels: {wmh_pixels} | "
                 f"WMH Asym: {wmh_asymmetry:.4f} | "
                 f"Normal Asym: {normal_asymmetry:.4f} | "
                 f"Contrast: {contrast_ratio:.2f}x")
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = output_dir / f'asymmetry_corrected_{sample_name}_slice{best_slice}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"  WMH Asymmetry: {wmh_asymmetry:.4f} vs Normal: {normal_asymmetry:.4f} (Contrast: {contrast_ratio:.2f}x)")
    plt.close()
    
    return {
        'sample': sample_name,
        'wmh_asymmetry': wmh_asymmetry,
        'normal_asymmetry': normal_asymmetry,
        'contrast_ratio': contrast_ratio
    }


def main():
    print("="*70)
    print("测试左右不对称特征提取 (修正版 - 使用 flipud)")
    print("="*70)
    
    # 测试5个样本：好、中、差各选一些
    test_samples = ['151', '150', '26', '153', '168']  # 从好到差
    
    test_dir = Path(config.TEST_DIR)
    output_dir = Path('d:/VSCode/AIOT_E1/TEMP2')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n测试样本: {test_samples}")
    print(f"翻转方法: np.flipud (axis=0 / 第一维)")
    print(f"特征图颜色: 灰阶 (gray)")
    print(f"输出目录: {output_dir}\n")
    
    results = []
    
    for sample_name in test_samples:
        print(f"\n{'='*70}")
        print(f"Processing Sample: {sample_name}")
        print(f"{'='*70}")
        
        # 查找样本
        sample_paths = list(test_dir.rglob(f'*/{sample_name}/pre/FLAIR.nii.gz'))
        if len(sample_paths) == 0:
            print(f"⚠️  Sample {sample_name} not found, skipping...")
            continue
        
        sample_path = sample_paths[0].parent.parent
        
        # 加载数据
        flair_raw = nib.load(sample_path / 'pre' / 'FLAIR.nii.gz').get_fdata()
        mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
        
        # 预处理
        flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
        mask = center_crop(mask_raw, config.TARGET_SIZE)
        
        # 可视化
        result = visualize_asymmetry(sample_name, flair, mask, output_dir)
        results.append(result)
    
    # 总结
    print("\n" + "="*70)
    print("总结分析 (使用 flipud 方法)")
    print("="*70)
    
    print(f"\n{'Sample':<10} {'WMH Asym':<12} {'Normal Asym':<12} {'Contrast':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['sample']:<10} {r['wmh_asymmetry']:<12.4f} {r['normal_asymmetry']:<12.4f} {r['contrast_ratio']:<10.2f}x")
    
    # 计算平均
    avg_wmh = np.mean([r['wmh_asymmetry'] for r in results])
    avg_normal = np.mean([r['normal_asymmetry'] for r in results])
    avg_contrast = np.mean([r['contrast_ratio'] for r in results if r['contrast_ratio'] > 0])
    
    print("-"*70)
    print(f"{'Average':<10} {avg_wmh:<12.4f} {avg_normal:<12.4f} {avg_contrast:<10.2f}x")
    
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    if avg_contrast > 1.5:
        print("✅ WMH区域的不对称度显著高于正常组织！")
        print(f"   平均对比度: {avg_contrast:.2f}x")
        print("   建议: 可以将此特征作为额外输入通道")
    elif avg_contrast > 1.0:
        print("⚠️  WMH区域的不对称度略高于正常组织")
        print(f"   平均对比度: {avg_contrast:.2f}x")
        print("   建议: 可以尝试，但效果可能有限")
    else:
        print("❌ WMH区域的不对称度与正常组织相近")
        print(f"   平均对比度: {avg_contrast:.2f}x")
        print("   建议: 此特征可能不太有用")
    
    print(f"\n📁 所有图片已保存到: {output_dir}")
    print(f"    文件名: asymmetry_corrected_*.png")
    print("="*70)


if __name__ == '__main__':
    main()
