"""
验证数据增强中的HorizontalFlip到底翻转哪个轴
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import sys
import albumentations as A

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent / 'newUnet'))
from dataset import center_crop, normalize_slice, to_uint8
import config


def test_horizontal_flip(sample_name='151'):
    """测试albumentations的HorizontalFlip"""
    
    # 加载样本
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
    
    if best_slice is None:
        best_slice = flair.shape[2] // 2
    
    flair_slice = flair[:, :, best_slice]
    mask_slice = mask[:, :, best_slice]
    
    # 归一化为[0, 1]用于显示
    flair_norm = to_uint8(flair_slice).astype(np.float32) / 255.0
    
    # 创建RGB叠加
    def create_overlay(img, msk):
        rgb = np.stack([img, img, img], axis=-1).copy()
        wmh_mask = (msk == 1)
        rgb[wmh_mask, 0] = 1.0
        rgb[wmh_mask, 1] = 0.0
        rgb[wmh_mask, 2] = 0.0
        return rgb
    
    overlay_original = create_overlay(flair_norm, mask_slice)
    
    # 应用albumentations的HorizontalFlip
    transform = A.HorizontalFlip(p=1.0)  # 100%概率翻转
    
    # Albumentations需要(H, W, C)格式
    image_hwc = overlay_original  # 已经是(H, W, 3)
    
    augmented = transform(image=image_hwc)
    flipped_overlay = augmented['image']
    
    # 同时测试numpy的fliplr和flipud
    flipped_lr = np.fliplr(overlay_original)
    flipped_ud = np.flipud(overlay_original)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'数据增强HorizontalFlip验证 - Sample {sample_name}, Slice {best_slice}\n'
                 f'Shape: {flair_slice.shape}',
                 fontsize=16, fontweight='bold')
    
    # 原图
    axes[0, 0].imshow(overlay_original)
    axes[0, 0].set_title('原图 (FLAIR + WMH红色标注)\n这是参考图', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Albumentations HorizontalFlip
    axes[0, 1].imshow(flipped_overlay)
    axes[0, 1].set_title('Albumentations HorizontalFlip\n(数据增强使用的方法)', 
                        fontsize=14, fontweight='bold', color='blue')
    axes[0, 1].axis('off')
    
    # np.fliplr
    axes[1, 0].imshow(flipped_lr)
    axes[1, 0].set_title('np.fliplr (翻转axis=1/列)\n水平翻转', 
                        fontsize=14, color='green')
    axes[1, 0].axis('off')
    
    # np.flipud
    axes[1, 1].imshow(flipped_ud)
    axes[1, 1].set_title('np.flipud (翻转axis=0/行)\n垂直翻转', 
                        fontsize=14, color='red')
    axes[1, 1].axis('off')
    
    # 添加说明
    instruction = """
观察指南：
1. 蓝色标题（Albumentations HorizontalFlip）与哪个numpy方法一致？
2. 如果与绿色（fliplr）一致 → 数据增强正确（水平翻转/左右镜像）
3. 如果与红色（flipud）一致 → 数据增强错误（垂直翻转/上下镜像）

期望结果：
- Albumentations HorizontalFlip 应该 = np.fliplr (绿色)
- 这样才是正确的"水平翻转"（左右镜像）
"""
    
    fig.text(0.5, 0.02, instruction, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2') / f'augmentation_flip_check_{sample_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 验证图保存到: {output_path}")
    
    # 自动比较
    print(f"\n自动验证结果:")
    
    # 比较与fliplr的差异
    diff_lr = np.abs(flipped_overlay - flipped_lr).sum()
    diff_ud = np.abs(flipped_overlay - flipped_ud).sum()
    
    print(f"  Albumentations与np.fliplr的差异: {diff_lr:.6f}")
    print(f"  Albumentations与np.flipud的差异: {diff_ud:.6f}")
    
    if diff_lr < 1e-5:
        print(f"\n✅ 结论: Albumentations HorizontalFlip = np.fliplr (水平翻转)")
        print(f"   数据增强配置正确！")
    elif diff_ud < 1e-5:
        print(f"\n⚠️ 结论: Albumentations HorizontalFlip = np.flipud (垂直翻转)")
        print(f"   这是错误的！需要修改配置")
    else:
        print(f"\n❓ 结论: 无法确定，请人工查看图片")
    
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("验证数据增强HorizontalFlip的翻转轴")
    print("="*70)
    test_horizontal_flip('151')
    print("="*70)
