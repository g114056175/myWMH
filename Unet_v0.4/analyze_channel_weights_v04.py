"""
分析V0.4 Best Model的7通道权重
确定是否需要删除某些通道
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('d:/VSCode/AIOT_E1/Unet_v0.4')

from model import AttentionUNet

print("="*70)
print("V0.4模型7通道权重分析")
print("="*70)

# 加载best model
checkpoint_path = Path('d:/VSCode/AIOT_E1/Unet_v0.4/checkpoints/best_model.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\nBest Model信息:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val Dice: {checkpoint['val_dice']:.4f}")
print(f"  Val Sens: {checkpoint['val_sens']:.4f}")
print(f"  Val Prec: {checkpoint['val_prec']:.4f}")

# 创建模型并加载权重
model = AttentionUNet(in_channels=7, base_channels=64, depth=5)
model.load_state_dict(checkpoint['model_state_dict'])

# 获取第一层卷积的权重
first_conv_weight = model.encoders[0].conv[0].weight.data  # [out_channels, 7, 3, 3]

print(f"\n第一层卷积权重shape: {first_conv_weight.shape}")
print(f"  输出通道: {first_conv_weight.shape[0]}")
print(f"  输入通道: {first_conv_weight.shape[1]} (7通道)")
print(f"  卷积核: {first_conv_weight.shape[2]}x{first_conv_weight.shape[3]}")

# 计算每个输入通道的重要性
# 方法: 对每个输入通道，计算所有输出通道和空间位置的权重绝对值平均
channel_importance = first_conv_weight.abs().mean(dim=[0, 2, 3]).numpy()

# 归一化为百分比
total_weight = channel_importance.sum()
channel_percentage = (channel_importance / total_weight) * 100

# 通道名称
channel_names = [
    'CLAHE[t-1]',
    'CLAHE[t]',
    'CLAHE[t+1]',
    'T1',
    'HighPass',
    'Asymmetry',
    'SpatialAtlas'
]

print("\n" + "="*70)
print("通道权重分析结果")
print("="*70)

print(f"\n{'#':<4} {'通道名称':<20} {'绝对权重':<12} {'百分比':<10} {'评估':<10}")
print("-"*70)

for i in range(7):
    percentage = channel_percentage[i]
    
    # 评估
    if percentage > 15:
        assessment = "⭐⭐⭐ 关键"
    elif percentage > 13:
        assessment = "⭐⭐ 重要"
    elif percentage > 11:
        assessment = "⭐ 有用"
    else:
        assessment = "⚠️ 较弱"
    
    print(f"{i+1:<4} {channel_names[i]:<20} {channel_importance[i]:<12.6f} "
          f"{percentage:>6.2f}%    {assessment}")

print("="*70)

# 排序
sorted_indices = np.argsort(channel_percentage)[::-1]

print(f"\n按重要性排序:")
print(f"{'排名':<6} {'通道':<20} {'权重%':<10}")
print("-"*50)
for rank, idx in enumerate(sorted_indices):
    print(f"#{rank+1:<5} {channel_names[idx]:<20} {channel_percentage[idx]:>6.2f}%")

# 统计分析
print("\n" + "="*70)
print("统计分析")
print("="*70)

mean_pct = channel_percentage.mean()
std_pct = channel_percentage.std()
max_pct = channel_percentage.max()
min_pct = channel_percentage.min()

print(f"\n平均权重: {mean_pct:.2f}%")
print(f"标准差: {std_pct:.2f}%")
print(f"最高权重: {max_pct:.2f}% ({channel_names[channel_percentage.argmax()]})")
print(f"最低权重: {min_pct:.2f}% ({channel_names[channel_percentage.argmin()]})")
print(f"权重比 (最高/最低): {max_pct/min_pct:.2f}x")

# 识别弱通道
threshold = mean_pct - 0.5 * std_pct  # 低于平均-0.5标准差
weak_channels = [i for i in range(7) if channel_percentage[i] < threshold]

print(f"\n低于阈值 ({threshold:.2f}%) 的通道:")
if weak_channels:
    for idx in weak_channels:
        print(f"  - {channel_names[idx]}: {channel_percentage[idx]:.2f}%")
else:
    print("  无")

# 建议
print("\n" + "="*70)
print("删除建议")
print("="*70)

# 规则1: 如果权重<11%且明显最低
if min_pct < 11 and (max_pct / min_pct) > 1.4:
    weakest_idx = channel_percentage.argmin()
    print(f"\n⚠️  发现明显弱通道:")
    print(f"  通道: {channel_names[weakest_idx]}")
    print(f"  权重: {min_pct:.2f}% (最低)")
    print(f"  与最高权重比: 1:{max_pct/min_pct:.2f}")
    print(f"\n  建议: 可以考虑删除此通道")
    print(f"  预期影响: Test Dice可能下降 <0.5%")
    print(f"  好处: 训练速度提升约 {(1/7)*100:.0f}%")
else:
    print(f"\n✓ 所有通道权重相对均衡")
    print(f"  最低权重: {min_pct:.2f}%")
    print(f"  权重差异: {max_pct/min_pct:.2f}x (<1.4倍)")
    print(f"\n  建议: 保留所有7个通道")
    print(f"  理由: 每个通道都有显著贡献")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 柱状图
colors = ['green' if p > mean_pct else 'orange' if p > threshold else 'red' 
          for p in channel_percentage]
axes[0].bar(range(7), channel_percentage, color=colors, edgecolor='black', linewidth=1.5)
axes[0].axhline(mean_pct, color='blue', linestyle='--', linewidth=2, label=f'平均 ({mean_pct:.2f}%)')
axes[0].axhline(threshold, color='red', linestyle=':', linewidth=2, label=f'阈值 ({threshold:.2f}%)')
axes[0].set_xlabel('通道', fontsize=12, fontweight='bold')
axes[0].set_ylabel('权重百分比 (%)', fontsize=12, fontweight='bold')
axes[0].set_title('V0.4模型 - 7通道权重分布', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(channel_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (p, c) in enumerate(zip(channel_percentage, colors)):
    axes[0].text(i, p + 0.3, f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')

# 饼图
axes[1].pie(channel_percentage, labels=channel_names, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
axes[1].set_title('通道权重占比', fontsize=14, fontweight='bold')

plt.tight_layout()
output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v04_channel_weights_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ 可视化图已保存: {output_path}")

print("\n" + "="*70)
print("分析完成")
print("="*70)
