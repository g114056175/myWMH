"""
检查训练集/验证集拆分方式
"""

import sys
sys.path.append('d:/VSCode/AIOT_E1/newUnet2')

import torch
from torch.utils.data import random_split
from dataset import WMHDataset
import config

print("="*70)
print("训练集/验证集拆分方式检查")
print("="*70)

# 加载完整数据集
full_dataset = WMHDataset(
    data_root=config.TRAIN_DIR,
    target_size=config.TARGET_SIZE,
    transform=None,
    clahe_clip_limit=config.CLAHE_CLIP_LIMIT,
    highpass_sigma=config.HIGHPASS_SIGMA
)

print(f"\n完整数据集信息:")
print(f"  数据路径: {config.TRAIN_DIR}")
print(f"  总volumes: {len(full_dataset.samples)}")
print(f"  总slices: {len(full_dataset)}")

# 检查slice_indices结构
print(f"\nSlice indices结构:")
print(f"  格式: (volume_idx, slice_idx)")
print(f"  前10个: {full_dataset.slice_indices[:10]}")

# 拆分
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"\n拆分结果:")
print(f"  训练集slices: {len(train_dataset)}")
print(f"  验证集slices: {len(val_dataset)}")

# 关键检查：同一volume的slices是否同时出现在train和val
print("\n" + "="*70)
print("数据泄漏检查")
print("="*70)

train_indices = [i for i in train_dataset.indices]
val_indices = [i for i in val_dataset.indices]

# 获取每个split中涉及的volume
train_volumes = set()
for idx in train_indices:
    vol_idx, slice_idx = full_dataset.slice_indices[idx]
    train_volumes.add(vol_idx)

val_volumes = set()
for idx in val_indices:
    vol_idx, slice_idx = full_dataset.slice_indices[idx]
    val_volumes.add(vol_idx)

overlap = train_volumes & val_volumes

print(f"\n训练集volumes: {sorted(train_volumes)[:10]}... (共{len(train_volumes)}个)")
print(f"验证集volumes: {sorted(val_volumes)[:10]}... (共{len(val_volumes)}个)")
print(f"\n重叠volumes: {overlap}")

if len(overlap) > 0:
    print(f"\n⚠️  警告！数据泄漏！")
    print(f"  {len(overlap)}/{len(full_dataset.samples)} volumes同时出现在训练集和验证集")
    print(f"  重叠比例: {len(overlap)/len(full_dataset.samples)*100:.1f}%")
    print(f"\n  这意味着:")
    print(f"    - 同一病例的不同层在train和val中")
    print(f"    - 验证集性能会被高估")
    print(f"    - 这是严重的方法学错误！")
    
    # 计算有多少slices受影响
    affected_slices = 0
    for idx in val_indices:
        vol_idx, _ = full_dataset.slice_indices[idx]
        if vol_idx in train_volumes:
            affected_slices += 1
    
    print(f"\n  受影响的验证集slices: {affected_slices}/{len(val_dataset)} ({affected_slices/len(val_dataset)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("建议修正方案")
    print("="*70)
    print("\n需要修改train.py:")
    print("""
# 错误方式 (当前):
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# → 按slice随机拆分，导致数据泄漏

# 正确方式:
# 1. 先拆分volumes
num_volumes = len(full_dataset.samples)
train_vol_size = int(0.8 * num_volumes)
val_vol_size = num_volumes - train_vol_size

volume_indices = list(range(num_volumes))
np.random.shuffle(volume_indices)

train_volumes = set(volume_indices[:train_vol_size])
val_volumes = set(volume_indices[train_vol_size:])

# 2. 根据volume归属确定每个slice
train_indices = [i for i, (vol_idx, _) in enumerate(full_dataset.slice_indices) 
                 if vol_idx in train_volumes]
val_indices = [i for i, (vol_idx, _) in enumerate(full_dataset.slice_indices) 
               if vol_idx in val_volumes]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
""")
else:
    print(f"\n✓ 无数据泄漏")
    print(f"  训练集和验证集的volumes完全不重叠")
    print(f"  这是正确的拆分方式")

print("\n" + "="*70)
