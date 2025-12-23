"""
重新检查训练曲线和学习率
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

history_path = Path('d:/VSCode/AIOT_E1/Unet_v0.4/logs/training_history.json')
with open(history_path, 'r') as f:
    history = json.load(f)

epochs = np.arange(1, len(history['val_dice']) + 1)
val_dice = np.array(history['val_dice'])
train_dice = np.array(history['train_dice'])
lr = np.array(history['lr'])

best_epoch = np.argmax(val_dice) + 1
best_val = val_dice.max()

print("="*70)
print("重新检查训练曲线")
print("="*70)

print(f"\nBest Epoch: {best_epoch}")
print(f"Best Val Dice: {best_val:.4f}")

# 检查epoch 32后的表现
post_32 = val_dice[31:]  # epoch 32 onwards
post_32_epochs = epochs[31:]

print(f"\nEpoch 32之后的Val Dice:")
print(f"  Mean: {post_32.mean():.4f}")
print(f"  Std:  {post_32.std():.4f}")
print(f"  Max:  {post_32.max():.4f} (Epoch {post_32_epochs[post_32.argmax()]})")
print(f"  Min:  {post_32.min():.4f} (Epoch {post_32_epochs[post_32.argmin()]})")

# 是否在震荡
if post_32.std() < 0.01:
    print(f"  ✓ 稳定震荡 (std < 0.01)")
else:
    print(f"  ⚠ 波动较大 (std >= 0.01)")

# 检查LR在epoch 32后
post_32_lr = lr[31:]
print(f"\nEpoch 32之后的Learning Rate:")
print(f"  Mean: {post_32_lr.mean():.6f}")
print(f"  Min:  {post_32_lr.min():.6f}")
print(f"  Max:  {post_32_lr.max():.6f}")

# cosine restart at epoch 50
if len(epochs) > 50:
    print(f"\nEpoch 50 (Cosine Restart):")
    print(f"  LR before: {lr[49]:.6f}")
    print(f"  LR at 50:  {lr[50]:.6f}")
    if len(lr) > 51:
        print(f"  LR after:  {lr[51]:.6f}")

# 关键判断
print("\n" + "="*70)
print("结论")
print("="*70)

# 用户观察正确吗
print(f"\n用户观察分析:")
print(f"  'Epoch 32后在0.75震荡': ")
post_32_around_75 = post_32[(post_32 >= 0.74) & (post_32 <= 0.76)]
print(f"    在[0.74, 0.76]范围: {len(post_32_around_75)}/{len(post_32)}个epoch")
if len(post_32_around_75) / len(post_32) > 0.5:
    print(f"    ✓ 确实，超过一半时间在这个范围")
else:
    print(f"    部分正确")

# LR是否太低
print(f"\n  'LR太低':")
mean_lr_post32 = post_32_lr.mean()
if mean_lr_post32 < 3e-5:
    print(f"    ✓ 确实，平均LR {mean_lr_post32:.6f} < 3e-5")
    print(f"    学习能力有限")
else:
    print(f"    平均LR {mean_lr_post32:.6f}，尚可")

# 建议
print("\n" + "="*70)
print("修正建议")
print("="*70)

if post_32.std() < 0.01 and mean_lr_post32 < 3e-5:
    print("\n✓ 用户观察正确:")
    print("  - Epoch 32后确实plateau")
    print("  - LR太低，无法继续改善")
    print("\n建议:")
    print("  1. 接受当前模型 (Best Epoch 32)")
    print("  2. 或者重新训练，调整LR schedule:")
    print("     - 使用更长的T_0 (100 instead of 50)")
    print("     - 或使用ReduceLROnPlateau但设置更高的min_lr")
else:
    print("\n需要进一步观察")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Val Dice全程
axes[0, 0].plot(epochs, val_dice, 'b-', linewidth=2, label='Val Dice')
axes[0, 0].axvline(32, color='r', linestyle='--', label='Epoch 32')
axes[0, 0].axhline(best_val, color='g', linestyle='--', alpha=0.5, label=f'Best={best_val:.4f}')
axes[0, 0].axhline(0.75, color='orange', linestyle=':', alpha=0.5, label='0.75')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Val Dice', fontsize=12)
axes[0, 0].set_title('Validation Dice - Full Training', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Epoch 32后detail
axes[0, 1].plot(post_32_epochs, post_32, 'b-', linewidth=2, marker='o', markersize=4)
axes[0, 1].axhline(best_val, color='g', linestyle='--', alpha=0.5, label=f'Best={best_val:.4f}')
axes[0, 1].axhline(0.75, color='orange', linestyle=':', alpha=0.5, label='0.75')
axes[0, 1].axhline(post_32.mean(), color='purple', linestyle='--', alpha=0.5, label=f'Mean={post_32.mean():.4f}')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Val Dice', fontsize=12)
axes[0, 1].set_title('Post-Epoch 32 Detail', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# LR schedule
axes[1, 0].plot(epochs, lr, 'g-', linewidth=2)
axes[1, 0].axvline(32, color='r', linestyle='--', label='Epoch 32')
axes[1, 0].axvline(50, color='purple', linestyle='--', label='Cosine Restart (50)')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Train vs Val
axes[1, 1].plot(epochs, train_dice, 'b-', label='Train', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs, val_dice, 'r-', label='Val', linewidth=2)
axes[1, 1].axvline(32, color='g', linestyle='--', alpha=0.5, label='Best Epoch')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Dice Score', fontsize=12)
axes[1, 1].set_title('Train vs Val Dice', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path('d:/VSCode/AIOT_E1/TEMP2/training_curve_recheck.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_path}")
