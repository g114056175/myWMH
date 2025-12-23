import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('d:/VSCode/AIOT_E1/Unet_v0.4')
from model import AttentionUNet

print('Loading checkpoint...')
checkpoint = torch.load(
    Path('d:/VSCode/AIOT_E1/Unet_v0.4/checkpoints/best_model.pth'),
    map_location='cpu',
    weights_only=False
)

model = AttentionUNet(in_channels=7, base_channels=64, depth=5, dropout=0.08)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model from Epoch: {checkpoint['epoch']}")
print(f"Val Dice: {checkpoint['val_dice']:.4f}")

# 获取第一层卷积的权重
w = model.encoders[0].conv[0].weight.data  # shape: [64, 7, 3, 3]

# 计算每个输入通道的平均权重（绝对值）
ch_imp = w.abs().mean(dim=[0, 2, 3]).numpy()  # shape: [7]

# 计算百分比
total = ch_imp.sum()
pct = (ch_imp / total) * 100

names = [
    'CLAHE[t-1]',
    'CLAHE[t]', 
    'CLAHE[t+1]',
    'T1',
    'HighPass',
    'Asymmetry',
    'SpatialAtlas'
]

print("\n" + "="*70)
print("7通道权重分析 (Best Model)")
print("="*70)

for i in range(7):
    print(f"{i+1}. {names[i]:<20} Weight: {ch_imp[i]:.6f}  ({pct[i]:6.2f}%)")

print("="*70)

# 排序
sorted_idx = np.argsort(pct)[::-1]
print("\n按重要性排序:")
for j, i in enumerate(sorted_idx):
    print(f"#{j+1}. {names[i]:<20} {pct[i]:6.2f}%")

print("="*70)
print(f"\n统计:")
print(f"  最高: {names[sorted_idx[0]]:<20} {pct[sorted_idx[0]]:6.2f}%")
print(f"  最低: {names[sorted_idx[-1]]:<20} {pct[sorted_idx[-1]]:6.2f}%")
print(f"  比例: {pct[sorted_idx[0]]/pct[sorted_idx[-1]]:6.2f}x")
print(f"  平均: {pct.mean():6.2f}%")

# 检查低权重通道
threshold = 10
weak = [i for i in range(7) if pct[i] < threshold]
if weak:
    print(f"\n权重<{threshold}%的通道:")
    for i in weak:
        print(f"  - {names[i]}: {pct[i]:.2f}%")
else:
    print(f"\n所有通道都>={threshold}%")

print("\n" + "="*70)
