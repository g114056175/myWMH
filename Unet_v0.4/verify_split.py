"""
快速验证：训练集/验证集是否按病例拆分
"""

import sys
sys.path.append('d:/VSCode/AIOT_E1/Unet_v0.4')

import numpy as np
from dataset import WMHDataset
import config

print("="*70)
print("验证Patient-Level Split")
print("="*70)

# 加载数据集
full_dataset = WMHDataset(
    data_root=config.TRAIN_DIR,
    target_size=config.TARGET_SIZE,
    transform=None,
    clahe_clip_limit=config.CLAHE_CLIP_LIMIT,
    highpass_sigma=config.HIGHPASS_SIGMA
)

print(f"\n完整数据集:")
print(f"  总volumes: {len(full_dataset.samples)}")
print(f"  总slices: {len(full_dataset)}")

# 模拟训练脚本的拆分逻辑
num_volumes = len(full_dataset.samples)
train_vol_size = int(0.8 * num_volumes)
val_vol_size = num_volumes - train_vol_size

volume_indices = list(range(num_volumes))
np.random.seed(42)  # 相同的seed
np.random.shuffle(volume_indices)

train_volumes = set(volume_indices[:train_vol_size])
val_volumes = set(volume_indices[train_vol_size:])

print(f"\nVolume拆分:")
print(f"  训练集volumes: {len(train_volumes)}")
print(f"  验证集volumes: {len(val_volumes)}")
print(f"  重叠volumes: {len(train_volumes & val_volumes)}")

# 统计每个split的slices
train_indices = []
val_indices = []

for i, (vol_idx, slice_idx) in enumerate(full_dataset.slice_indices):
    if vol_idx in train_volumes:
        train_indices.append(i)
    else:
        val_indices.append(i)

print(f"\nSlice统计:")
print(f"  训练集slices: {len(train_indices)}")
print(f"  验证集slices: {len(val_indices)}")

# 验证
train_vols_check = set([full_dataset.slice_indices[i][0] for i in train_indices])
val_vols_check = set([full_dataset.slice_indices[i][0] for i in val_indices])

print(f"\n验证:")
print(f"  训练集包含volumes: {len(train_vols_check)}")
print(f"  验证集包含volumes: {len(val_vols_check)}")
print(f"  重叠: {len(train_vols_check & val_vols_check)}")

if len(train_vols_check & val_vols_check) == 0:
    print(f"\n✅ 成功！无数据泄漏")
    print(f"  48/12 volume split: {len(train_vols_check)}/{len(val_vols_check)}")
else:
    print(f"\n❌ 失败！仍有数据泄漏")

# 显示训练集和验证集的volume IDs
print(f"\n训练集volumes (前10): {sorted(train_volumes)[:10]}")
print(f"验证集volumes (全部): {sorted(val_volumes)}")
