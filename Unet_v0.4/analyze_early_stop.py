"""
分析V0.3+Dropout的训练曲线和早停问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 加载训练历史
history_path = Path('d:/VSCode/AIOT_E1/Unet_v0.4/logs/training_history.json')
with open(history_path, 'r') as f:
    history = json.load(f)

epochs = len(history['train_dice'])
best_epoch = np.argmax(history['val_dice']) + 1
best_val = max(history['val_dice'])

print("="*70)
print("V0.3+Dropout Early Stop分析")
print("="*70)

print(f"\n训练信息:")
print(f"  Total Epochs: {epochs}")
print(f"  Best Epoch: {best_epoch}")
print(f"  Best Val Dice: {best_val:.4f}")
print(f"  Patience: 20")

# 检查最后20 epoch的Val Dice变化
last_20_val = history['val_dice'][-20:]
last_20_range = max(last_20_val) - min(last_20_val)
last_20_trend = (last_20_val[-1] - last_20_val[0]) / last_20_val[0]

print(f"\n最后20 epochs分析:")
print(f"  Val Dice range: {last_20_range:.4f}")
print(f"  Trend: {last_20_trend*100:+.2f}%")

# 检查是否还在学习
if last_20_range > 0.02:
    print(f"  ⚠️  Range >0.02，说明还在波动，可能未充分收敛")
elif abs(last_20_trend) > 0.01:
    print(f"  ⚠️  Trend显著，说明还在变化")
else:
    print(f"  ✓ 已基本plateau")

# 检查训练集是否还在提升
train_last_10 = history['train_dice'][-10:]
train_trend = (train_last_10[-1] - train_last_10[0]) / train_last_10[0]

print(f"\n训练集趋势 (最后10 epochs):")
print(f"  Train Dice: {train_last_10[0]:.4f} → {train_last_10[-1]:.4f}")
print(f"  Trend: {train_trend*100:+.2f}%")

if train_trend > 0.01:
    print(f"  ⚠️  训练集还在提升，验证集可能滞后")

# 建议
print("\n" + "="*70)
print("建议")
print("="*70)

if last_20_range > 0.02 or abs(last_20_trend) > 0.01:
    print("\n⚠️  Early stop可能过早:")
    print(f"  - Val Dice还在波动/变化")
    print(f"  - 建议延长patience到30-40")
    print(f"  - 或降低min_delta到0.001")
    
    print("\n修改建议:")
    print("""
config.py:
  EARLY_STOP_PATIENCE = 30  # 从20提高
  EARLY_STOP_MIN_DELTA = 0.001  # 从0.002降低
""")
else:
    print("\n✓ Early stop timing合理")
    print(f"  - Val Dice已plateau")
    print(f"  - 继续训练可能过拟合")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Val Dice
axes[0, 0].plot(history['val_dice'], 'b-', linewidth=2)
axes[0, 0].axvline(best_epoch, color='r', linestyle='--', label=f'Best ({best_epoch})')
axes[0, 0].axhline(best_val, color='g', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Val Dice')
axes[0, 0].set_title('Validation Dice Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 最后30 epochs detail
start = max(0, epochs - 30)
axes[0, 1].plot(range(start, epochs), history['val_dice'][start:], 'b-', linewidth=2, marker='o')
axes[0, 1].axvline(best_epoch, color='r', linestyle='--', label=f'Best ({best_epoch})')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Val Dice')
axes[0, 1].set_title('Last 30 Epochs Detail')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Train vs Val
axes[1, 0].plot(history['train_dice'], 'b-', label='Train', linewidth=2)
axes[1, 0].plot(history['val_dice'], 'r-', label='Val', linewidth=2)
axes[1, 0].axvline(best_epoch, color='g', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Dice')
axes[1, 0].set_title('Train vs Val Dice')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# LR
axes[1, 1].plot(history['lr'], 'g-', linewidth=2)
axes[1, 1].axvline(best_epoch, color='r', linestyle='--', label=f'Best ({best_epoch})')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d:/VSCode/AIOT_E1/TEMP2/early_stop_analysis.png', dpi=150)
print(f"\n✓ Saved: d:/VSCode/AIOT_E1/TEMP2/early_stop_analysis.png")
