"""
Plot V0.3 Training History
Visualize loss, dice, sensitivity, and learning rate curves
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training history
history_path = Path('d:/VSCode/AIOT_E1/newUnet2/logs/training_history.json')

with open(history_path, 'r') as f:
    history = json.load(f)

# Extract data
epochs = list(range(1, len(history['train_loss']) + 1))
train_loss = history['train_loss']
train_dice = history['train_dice']
train_sens = history['train_sens']
val_loss = history['val_loss']
val_dice = history['val_dice']
val_sens = history['val_sens']
val_prec = history['val_prec']
lr = history['lr']

# Find best epoch
best_epoch = np.argmax(val_dice) + 1
best_dice = max(val_dice)

print("="*70)
print("V0.3 Training Summary")
print("="*70)
print(f"Total Epochs: {len(epochs)}")
print(f"Best Epoch: {best_epoch}")
print(f"Best Val Dice: {best_dice:.4f}")
print(f"Final Train Dice: {train_dice[-1]:.4f}")
print(f"Final Val Dice: {val_dice[-1]:.4f}")
print(f"Final LR: {lr[-1]:.6f}")
print("="*70)

# Create comprehensive training curve plot
fig = plt.figure(figsize=(20, 12))

# 1. Loss curves
ax1 = plt.subplot(2, 3, 1)
ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Dice curves
ax2 = plt.subplot(2, 3, 2)
ax2.plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
ax2.plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
ax2.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
ax2.axhline(best_dice, color='g', linestyle=':', alpha=0.5, label=f'Best Dice ({best_dice:.4f})')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Dice Score', fontsize=12)
ax2.set_title('Dice Score Curves', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Sensitivity curves
ax3 = plt.subplot(2, 3, 3)
ax3.plot(epochs, train_sens, 'b-', label='Train Sensitivity', linewidth=2)
ax3.plot(epochs, val_sens, 'r-', label='Val Sensitivity', linewidth=2)
ax3.axvline(best_epoch, color='g', linestyle='--', alpha=0.5)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Sensitivity', fontsize=12)
ax3.set_title('Sensitivity Curves', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Learning Rate
ax4 = plt.subplot(2, 3, 4)
ax4.plot(epochs, lr, 'purple', linewidth=2)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Learning Rate', fontsize=12)
ax4.set_title('Learning Rate Schedule (CosineAnnealing)', fontsize=14, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
# Mark restarts
for i in [50, 100]:
    if i < len(epochs):
        ax4.axvline(i, color='red', linestyle='--', alpha=0.3)

# 5. Precision curve
ax5 = plt.subplot(2, 3, 5)
ax5.plot(epochs, val_prec, 'orange', label='Val Precision', linewidth=2)
ax5.axvline(best_epoch, color='g', linestyle='--', alpha=0.5)
ax5.set_xlabel('Epoch', fontsize=12)
ax5.set_ylabel('Precision', fontsize=12)
ax5.set_title('Validation Precision', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Train vs Val Dice (overfitting check)
ax6 = plt.subplot(2, 3, 6)
gap = [t - v for t, v in zip(train_dice, val_dice)]
ax6.plot(epochs, gap, 'purple', linewidth=2)
ax6.axhline(0, color='k', linestyle='-', alpha=0.3)
ax6.axvline(best_epoch, color='g', linestyle='--', alpha=0.5)
ax6.set_xlabel('Epoch', fontsize=12)
ax6.set_ylabel('Train Dice - Val Dice', fontsize=12)
ax6.set_title('Overfitting Check (Gap)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.fill_between(epochs, 0, gap, where=[g > 0 for g in gap], alpha=0.3, color='red', label='Overfitting')

plt.suptitle('V0.3 Training History (150 Epochs)', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v03_training_curves.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved to: {output_path}")

# Additional analysis
print("\n" + "="*70)
print("Training Analysis")
print("="*70)

# Overfitting analysis
final_gap = train_dice[-1] - val_dice[-1]
print(f"\nOverfitting Gap (Final):")
print(f"  Train Dice: {train_dice[-1]:.4f}")
print(f"  Val Dice:   {val_dice[-1]:.4f}")
print(f"  Gap:        {final_gap:.4f} ({final_gap/val_dice[-1]*100:.1f}%)")

if final_gap > 0.15:
    print("  ⚠️ WARNING: Significant overfitting detected!")
elif final_gap > 0.10:
    print("  ⚠️ Moderate overfitting")
else:
    print("  ✓ Acceptable gap")

# Learning rate restarts
print(f"\nLearning Rate Restarts (CosineAnnealing T_0=50):")
for epoch in [1, 50, 100, 150]:
    if epoch <= len(lr):
        print(f"  Epoch {epoch:3d}: LR = {lr[epoch-1]:.6f}")

# Best performance window
window = 10
best_window_start = max(0, best_epoch - window)
best_window_end = min(len(val_dice), best_epoch + window)
avg_dice_around_best = np.mean(val_dice[best_window_start:best_window_end])
print(f"\nPerformance around best epoch:")
print(f"  Best Epoch {best_epoch}: {best_dice:.4f}")
print(f"  Average (±{window} epochs): {avg_dice_around_best:.4f}")

print("\n✅ Training curve visualization complete!")
