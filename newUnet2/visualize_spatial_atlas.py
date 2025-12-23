"""
可视化Spatial Atlas Prior并展示如何使用
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载spatial atlas
atlas = np.load('spatial_atlas_prior.npy')

print("="*70)
print("Spatial Atlas Prior 详细信息")
print("="*70)

print(f"\n基本信息:")
print(f"  Shape: {atlas.shape}")
print(f"  Dtype: {atlas.dtype}")
print(f"  Min: {atlas.min():.6f}")
print(f"  Max: {atlas.max():.6f}")
print(f"  Mean: {atlas.mean():.6f}")
print(f"  Std: {atlas.std():.6f}")

print(f"\n统计分布:")
print(f"  值 > 0.5的像素: {(atlas > 0.5).sum()} ({(atlas > 0.5).sum()/atlas.size*100:.1f}%)")
print(f"  值 > 0.3的像素: {(atlas > 0.3).sum()} ({(atlas > 0.3).sum()/atlas.size*100:.1f}%)")
print(f"  值 > 0.1的像素: {(atlas > 0.1).sum()} ({(atlas > 0.1).sum()/atlas.size*100:.1f}%)")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Spatial Atlas Prior - 详细可视化\n'
             'Shape: (224, 224) - 与所有通道相同尺寸',
             fontsize=16, fontweight='bold')

# 1. 热力图
ax = axes[0, 0]
im = ax.imshow(atlas, cmap='hot')
ax.set_title('Spatial Atlas (热力图)\n高值=高WMH概率区域', fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# 2. 灰度图
ax = axes[0, 1]
im = ax.imshow(atlas, cmap='gray')
ax.set_title('Spatial Atlas (灰度)\n作为输入通道的样子', fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# 3. 等高线图
ax = axes[0, 2]
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
contour = ax.contourf(atlas, levels=20, cmap='hot')
ax.contour(atlas, levels=levels, colors='white', linewidths=1)
ax.set_title('等高线图\n显示概率分布区域', fontsize=12)
ax.axis('off')
plt.colorbar(contour, ax=ax, fraction=0.046)

# 4. 阈值可视化
ax = axes[1, 0]
threshold_map = np.zeros_like(atlas)
threshold_map[atlas > 0.5] = 1.0
threshold_map[(atlas > 0.3) & (atlas <= 0.5)] = 0.66
threshold_map[(atlas > 0.1) & (atlas <= 0.3)] = 0.33
im = ax.imshow(threshold_map, cmap='Reds')
ax.set_title('阈值分区\n红: >0.5, 粉: 0.3-0.5, 淡: 0.1-0.3', fontsize=12)
ax.axis('off')

# 5. 直方图
ax = axes[1, 1]
ax.hist(atlas.flatten(), bins=100, color='red', alpha=0.7, edgecolor='black')
ax.set_xlabel('Atlas值', fontsize=11)
ax.set_ylabel('像素数', fontsize=11)
ax.set_title('值分布直方图', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axvline(atlas.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {atlas.mean():.3f}')
ax.legend()

# 6. 3D表面图
from mpl_toolkits.mplot3d import Axes3D
ax = axes[1, 2]
ax.remove()
ax = fig.add_subplot(2, 3, 6, projection='3d')

x = np.arange(atlas.shape[1])
y = np.arange(atlas.shape[0])
X, Y = np.meshgrid(x, y)

surf = ax.plot_surface(X, Y, atlas, cmap='hot', alpha=0.9)
ax.set_title('3D表面图', fontsize=12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Prior值')
ax.view_init(elev=25, azim=45)

plt.tight_layout()
output_path = Path('d:/VSCode/AIOT_E1/TEMP2') / 'spatial_atlas_explained.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 详细可视化保存到: {output_path}")
plt.close()

print("\n" + "="*70)
print("如何使用:")
print("="*70)
print("""
这个Spatial Atlas是一个 (224, 224) 的2D特征图，包含了：
- 基于训练集统计的WMH出现概率
- 左右对称（符合大脑解剖）
- 平滑处理（避免过拟合）

在训练时，它会作为第7个通道添加到每个切片：
  Ch0-5: 现有6个通道 (FLAIR×3, CLAHE, HighPass, T1)
  Ch6: Spatial Atlas ⭐ 新增

关键特性：
- ✅ 所有样本共用同一张图（全局先验）
- ✅ 不随切片变化（2D固定）
- ✅ 尺寸与其他通道完全一致 (224×224)
- ✅ 数值范围 [0, 1]，归一化完成
""")
print("="*70)
