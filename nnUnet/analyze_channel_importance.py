"""
分析7通道模型的通道重要性
通過查看第一層卷積權重的L2 norm來評估各通道的貢獻
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 載入checkpoint
checkpoint_path = Path("nnUNet_results/Dataset003_WMH_7CH/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth")

if not checkpoint_path.exists():
    print("⚠️  checkpoint_best.pth 尚未生成，訓練可能還在進行中")
    print("嘗試使用 checkpoint_latest.pth...")
    checkpoint_path = Path("nnUNet_results/Dataset003_WMH_7CH/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_latest.pth")

if not checkpoint_path.exists():
    print("❌ 找不到任何checkpoint，訓練可能剛開始")
    exit(1)

print(f"載入checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 獲取網絡state_dict
if 'network_weights' in checkpoint:
    state_dict = checkpoint['network_weights']
else:
    state_dict = checkpoint

# 找到第一層卷積層的權重
# 在nnU-Net中，第一層通常是 encoder.stages.0.0.conv.weight
first_conv_keys = [k for k in state_dict.keys() if 'encoder' in k and 'conv' in k and 'weight' in k]
first_conv_keys.sort()

if len(first_conv_keys) == 0:
    print("❌ 找不到第一層卷積權重")
    exit(1)

first_conv_key = first_conv_keys[0]
print(f"分析層: {first_conv_key}")

# 權重shape: [out_channels, in_channels, kernel_h, kernel_w]
weights = state_dict[first_conv_key]
print(f"權重形狀: {weights.shape}")

# 計算每個輸入通道的L2 norm（所有輸出通道和kernel位置的總和）
channel_importance = []
for ch in range(weights.shape[1]):  # in_channels
    # 對該通道的所有權重計算L2 norm
    ch_weights = weights[:, ch, :, :]
    l2_norm = torch.norm(ch_weights).item()
    channel_importance.append(l2_norm)

channel_importance = np.array(channel_importance)

# 標準化為百分比
channel_importance_pct = (channel_importance / channel_importance.sum()) * 100

# 通道名稱
channel_names = [
    'FLAIR[t-1]',
    'FLAIR[t]',
    'FLAIR[t+1]',
    'T1',
    'CLAHE',
    'Top-hat',
    'HighPass σ=2.0'
]

# 打印結果
print("\n" + "=" * 80)
print("7通道重要性分析（基於第一層卷積權重L2 norm）")
print("=" * 80)
print(f"{'通道':<20} {'L2 Norm':<15} {'重要性%':<15} {'評級'}")
print("-" * 80)

for i, (name, norm, pct) in enumerate(zip(channel_names, channel_importance, channel_importance_pct)):
    if pct > 16:  # 平均是14.3%，高於16%算重要
        rating = "⭐⭐⭐ 重要"
    elif pct > 12:
        rating = "⭐⭐ 中等"
    else:
        rating = "⭐ 較低"
    
    print(f"{name:<20} {norm:<15.4f} {pct:<15.2f} {rating}")

print("-" * 80)
print(f"{'總和':<20} {channel_importance.sum():<15.4f} {100.00:<15.2f}")
print("=" * 80)

# 排序（從高到低）
sorted_indices = np.argsort(channel_importance_pct)[::-1]

print("\n" + "=" * 80)
print("重要性排名（從高到低）")
print("=" * 80)
for rank, idx in enumerate(sorted_indices, 1):
    symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    print(f"{symbol} {channel_names[idx]:<20} {channel_importance_pct[idx]:.2f}%")
print("=" * 80)

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 柱狀圖
colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
ax1.bar(range(7), channel_importance_pct, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(7))
ax1.set_xticklabels(channel_names, rotation=45, ha='right')
ax1.set_ylabel('重要性 (%)', fontsize=12, fontweight='bold')
ax1.set_title('7通道重要性分析\n(基於第一層卷積權重)', fontsize=14, fontweight='bold')
ax1.axhline(y=14.29, color='r', linestyle='--', label='平均值 (14.29%)')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 圓餅圖
ax2.pie(channel_importance_pct, labels=channel_names, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('7通道貢獻比例', fontsize=14, fontweight='bold')

plt.tight_layout()
output_path = Path('../TEMP_nnUnet/7ch_channel_importance.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ 圖表已保存: {output_path.absolute()}")

# 建議
print("\n" + "=" * 80)
print("💡 分析建議")
print("=" * 80)

top3 = sorted_indices[:3]
bottom2 = sorted_indices[-2:]

print(f"\n✅ 最重要的3個通道:")
for idx in top3:
    print(f"   - {channel_names[idx]}: {channel_importance_pct[idx]:.2f}%")

print(f"\n⚠️  貢獻較低的2個通道:")
for idx in bottom2:
    print(f"   - {channel_names[idx]}: {channel_importance_pct[idx]:.2f}%")

if channel_importance_pct[bottom2].max() < 10:
    print(f"\n🔍 建議: 如果某通道貢獻<10%，可考慮在未來版本中移除")

print("=" * 80)
