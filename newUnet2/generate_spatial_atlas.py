"""
生成空间先验图谱 (Spatial Atlas Prior Map)
基于训练集所有WMH mask的统计聚合

策略：
1. 聚合所有训练集mask
2. 除以样本总数得到概率图
3. 强制对称化 (利用大脑左右对称性)
4. 高斯模糊平滑 (避免过拟合)
5. 保存为.npy文件供训练使用
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib
import sys

# Fix Chinese font
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent / 'newUnet'))
from dataset import center_crop, normalize_slice
import config


def generate_spatial_atlas_prior(train_dir, target_size=224, sigma=7.0, 
                                  output_path='spatial_atlas_prior.npy',
                                  visualization_path='spatial_atlas_prior_viz.png'):
    """
    生成空间先验图谱
    
    Args:
        train_dir: 训练集路径
        target_size: 统一尺寸 (224)
        sigma: 高斯模糊sigma (5-10, 越大越平滑)
        output_path: 输出.npy文件路径
        visualization_path: 可视化图片路径
    """
    print("="*70)
    print("生成空间先验图谱 (Spatial Atlas Prior Map)")
    print("="*70)
    
    train_path = Path(train_dir)
    
    # 1. 找到所有训练样本的mask
    mask_files = list(train_path.rglob('*/wmh.nii.gz'))
    print(f"\n找到 {len(mask_files)} 个训练样本")
    
    if len(mask_files) == 0:
        raise FileNotFoundError(f"未找到训练mask文件在 {train_dir}")
    
    # 2. 初始化累加器
    accumulator = np.zeros((target_size, target_size), dtype=np.float32)
    valid_slices = 0
    
    print(f"\n开始聚合mask...")
    
    # 3. 遍历所有样本，累加mask
    for i, mask_file in enumerate(mask_files):
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(mask_files)}")
        
        # 加载mask
        mask_raw = nib.load(mask_file).get_fdata()
        
        # Center crop到统一尺寸
        mask_cropped = center_crop(mask_raw, target_size)
        
        # 遍历所有有效切片
        for z in range(2, mask_cropped.shape[2] - 2):
            mask_slice = mask_cropped[:, :, z]
            
            # 只统计WMH像素 (值为1)
            # 忽略don't care区域 (值为2)
            wmh_mask = (mask_slice == 1).astype(np.float32)
            
            # 累加
            accumulator += wmh_mask
            valid_slices += 1
    
    print(f"\n总共处理了 {valid_slices} 个有效切片")
    
    # 4. 归一化：除以样本数得到概率图
    probability_map = accumulator / valid_slices
    
    print(f"原始概率图统计:")
    print(f"  - 最小值: {probability_map.min():.6f}")
    print(f"  - 最大值: {probability_map.max():.6f}")
    print(f"  - 平均值: {probability_map.mean():.6f}")
    print(f"  - 非零像素比例: {(probability_map > 0).sum() / probability_map.size * 100:.2f}%")
    
    # 5. 强制对称化 (大脑左右对称性)
    print(f"\n应用对称化...")
    # 注意：根据之前的测试，正确的左右翻转应该是flipud (axis=0)
    flipped = np.flipud(probability_map)
    symmetric_map = (probability_map + flipped) / 2.0
    
    print(f"对称化后统计:")
    print(f"  - 最大值: {symmetric_map.max():.6f}")
    print(f"  - 平均值: {symmetric_map.mean():.6f}")
    
    # 6. 高斯模糊平滑 (避免过拟合)
    print(f"\n应用高斯模糊 (sigma={sigma})...")
    
    # 转换到uint8以便使用cv2
    symmetric_uint8 = (symmetric_map * 255).astype(np.uint8)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(symmetric_uint8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 转回float并归一化
    smoothed_map = blurred.astype(np.float32) / 255.0
    
    print(f"平滑后统计:")
    print(f"  - 最大值: {smoothed_map.max():.6f}")
    print(f"  - 平均值: {smoothed_map.mean():.6f}")
    
    # 7. 最终归一化到 [0, 1]，最大值=1.0
    final_map = smoothed_map / (smoothed_map.max() + 1e-8)
    
    print(f"\n最终图谱统计:")
    print(f"  - 最小值: {final_map.min():.6f}")
    print(f"  - 最大值: {final_map.max():.6f}")
    print(f"  - 平均值: {final_map.mean():.6f}")
    print(f"  - Shape: {final_map.shape}")
    
    # 8. 保存为.npy文件
    np.save(output_path, final_map)
    print(f"\n✅ 保存到: {output_path}")
    
    # 9. 生成可视化
    print(f"\n生成可视化...")
    visualize_spatial_atlas(
        probability_map, symmetric_map, smoothed_map, final_map,
        sigma, visualization_path
    )
    
    return final_map


def visualize_spatial_atlas(raw_prob, symmetric, smoothed, final, sigma, output_path):
    """可视化处理流程"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Spatial Atlas Prior Generation Pipeline (sigma={sigma})', 
                 fontsize=16, fontweight='bold')
    
    # 1. 原始累加概率图
    ax = axes[0, 0]
    im = ax.imshow(raw_prob, cmap='hot')
    ax.set_title(f'1. Raw Probability Map\nMax: {raw_prob.max():.4f}, Mean: {raw_prob.mean():.6f}', 
                fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 2. 对称化后
    ax = axes[0, 1]
    im = ax.imshow(symmetric, cmap='hot')
    ax.set_title(f'2. Symmetrized Map\n(Raw + FlipUD) / 2\nMax: {symmetric.max():.4f}', 
                fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 3. 高斯平滑后
    ax = axes[0, 2]
    im = ax.imshow(smoothed, cmap='hot')
    ax.set_title(f'3. Gaussian Smoothed\nsigma={sigma}\nMax: {smoothed.max():.4f}', 
                fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. 最终归一化
    ax = axes[1, 0]
    im = ax.imshow(final, cmap='hot')
    ax.set_title(f'4. Final Normalized\nMax: {final.max():.4f}, Mean: {final.mean():.4f}', 
                fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 5. 灰度显示
    ax = axes[1, 1]
    im = ax.imshow(final, cmap='gray')
    ax.set_title('5. Final (Grayscale)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 6. 3D视图
    ax = axes[1, 2]
    from mpl_toolkits.mplot3d import Axes3D
    ax.remove()
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    
    x = np.arange(final.shape[1])
    y = np.arange(final.shape[0])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, final, cmap='hot', alpha=0.8)
    ax.set_title('6. 3D Surface View', fontsize=11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Prior Probability')
    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化保存到: {output_path}")
    plt.close()


def test_spatial_atlas(atlas_path, test_sample_name='151'):
    """测试spatial atlas在实际样本上的叠加效果"""
    
    # 加载atlas
    atlas = np.load(atlas_path)
    
    # 加载测试样本
    test_dir = Path(config.TEST_DIR)
    sample_paths = list(test_dir.rglob(f'*/{test_sample_name}/pre/FLAIR.nii.gz'))
    
    if len(sample_paths) == 0:
        print(f"未找到测试样本 {test_sample_name}")
        return
    
    sample_path = sample_paths[0].parent.parent
    
    # 加载数据
    flair_raw = nib.load(sample_path / 'pre' / 'FLAIR.nii.gz').get_fdata()
    mask_raw = nib.load(sample_path / 'wmh.nii.gz').get_fdata()
    
    # 预处理
    flair = center_crop(normalize_slice(flair_raw), config.TARGET_SIZE)
    mask = center_crop(mask_raw, config.TARGET_SIZE)
    
    # 选择中间切片
    z = flair.shape[2] // 2
    flair_slice = flair[:, :, z]
    mask_slice = mask[:, :, z]
    
    # 归一化FLAIR用于显示
    flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min())
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Spatial Atlas Prior Test on Sample {test_sample_name}', 
                 fontsize=14, fontweight='bold')
    
    # FLAIR原图
    axes[0].imshow(flair_norm, cmap='gray')
    axes[0].set_title('FLAIR Original', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(flair_norm, cmap='gray')
    axes[1].imshow(mask_slice, cmap='Reds', alpha=0.5 * (mask_slice == 1))
    axes[1].set_title('Ground Truth (Red)', fontsize=12)
    axes[1].axis('off')
    
    # Spatial Atlas Prior
    axes[2].imshow(atlas, cmap='hot')
    axes[2].set_title('Spatial Atlas Prior\n(All Training Samples)', fontsize=12)
    axes[2].axis('off')
    
    # FLAIR + Atlas叠加
    axes[3].imshow(flair_norm, cmap='gray')
    axes[3].imshow(atlas, cmap='hot', alpha=0.3)
    axes[3].set_title('FLAIR + Atlas Overlay', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2') / f'atlas_test_{test_sample_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 测试图保存到: {output_path}")
    plt.close()


def main():
    # 设置参数
    train_dir = config.TRAIN_DIR
    target_size = config.TARGET_SIZE
    sigma = 7.0  # 较强的平滑 (5-10范围)
    
    output_dir = Path('d:/VSCode/AIOT_E1/newUnet')
    output_npy = output_dir / 'spatial_atlas_prior.npy'
    output_viz = Path('d:/VSCode/AIOT_E1/TEMP2') / 'spatial_atlas_prior_viz.png'
    
    # 生成spatial atlas
    atlas = generate_spatial_atlas_prior(
        train_dir=train_dir,
        target_size=target_size,
        sigma=sigma,
        output_path=output_npy,
        visualization_path=output_viz
    )
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. {output_npy} - 供训练使用")
    print(f"  2. {output_viz} - 处理流程可视化")
    
    # 测试在实际样本上的效果
    print(f"\n生成测试样本叠加效果...")
    test_spatial_atlas(output_npy, '151')
    
    print("\n" + "="*70)
    print("使用方法:")
    print("="*70)
    print("""
在 dataset.py 中:

1. 加载atlas:
   self.spatial_atlas = np.load('spatial_atlas_prior.npy')

2. 在 __getitem__ 中添加为通道:
   image = np.stack([
       flair_t_minus_1_norm,
       flair_t_norm,
       flair_t_plus_1_norm,
       clahe_t,
       highpass_t,
       t1_t_norm,
       self.spatial_atlas,  # ⭐ 新增：空间先验
   ], axis=-1)

3. 更新 config.py:
   IN_CHANNELS = 7  # 从6改为7
""")
    print("="*70)


if __name__ == '__main__':
    main()
