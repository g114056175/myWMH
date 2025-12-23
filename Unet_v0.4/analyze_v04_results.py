"""
Analyze V0.4 training results and compare with V0.3
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load V0.4 history
v04_history_path = Path('d:/VSCode/AIOT_E1/Unet_v0.4/logs/training_history.json')
with open(v04_history_path, 'r') as f:
    v04_history = json.load(f)

# Load V0.3 history
v03_history_path = Path('d:/VSCode/AIOT_E1/newUnet2/logs/training_history.json')
with open(v03_history_path, 'r') as f:
    v03_history = json.load(f)

# Extract data
v04_epochs = list(range(1, len(v04_history['train_dice']) + 1))
v03_epochs = list(range(1, len(v03_history['train_dice']) + 1))

# Find best epochs
v04_best_epoch = np.argmax(v04_history['val_dice']) + 1
v04_best_dice = max(v04_history['val_dice'])

v03_best_epoch = np.argmax(v03_history['val_dice']) + 1
v03_best_dice = max(v03_history['val_dice'])

print("="*70)
print("V0.4 vs V0.3 Training Comparison")
print("="*70)

print(f"\nV0.3:")
print(f"  Best Epoch: {v03_best_epoch}")
print(f"  Best Val Dice: {v03_best_dice:.4f}")
print(f"  Final Train Dice: {v03_history['train_dice'][-1]:.4f}")
print(f"  Total Epochs: {len(v03_epochs)}")

print(f"\nV0.4:")
print(f"  Best Epoch: {v04_best_epoch}")
print(f"  Best Val Dice: {v04_best_dice:.4f}")
print(f"  Final Train Dice: {v04_history['train_dice'][-1]:.4f}")
print(f"  Total Epochs: {len(v04_epochs)} (early stopped)")

print(f"\nDifference:")
print(f"  Val Dice: {v04_best_dice - v03_best_dice:.4f} ({(v04_best_dice/v03_best_dice - 1)*100:+.2f}%)")

# Analysis
print("\n" + "="*70)
print("Detailed Analysis")
print("="*70)

# Check if V0.4 converged
v04_last_10_val = v04_history['val_dice'][-10:]
v04_plateau = max(v04_last_10_val) - min(v04_last_10_val)
print(f"\nV0.4 Convergence:")
print(f"  Last 10 epochs Val Dice range: {v04_plateau:.4f}")
print(f"  Plateau: {'Yes' if v04_plateau < 0.01 else 'No'}")

# Check overfitting
v04_gap = v04_history['train_dice'][v04_best_epoch-1] - v04_best_dice
v03_gap = v03_history['train_dice'][v03_best_epoch-1] - v03_best_dice

print(f"\nTrain-Val Gap at best epoch:")
print(f"  V0.3: {v03_gap:.4f} ({v03_gap/v03_history['val_dice'][v03_best_epoch-1]*100:.1f}%)")
print(f"  V0.4: {v04_gap:.4f} ({v04_gap/v04_best_dice*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Val Dice comparison
axes[0, 0].plot(v03_epochs, v03_history['val_dice'], 'b-', label='V0.3', linewidth=2)
axes[0, 0].plot(v04_epochs, v04_history['val_dice'], 'r-', label='V0.4', linewidth=2)
axes[0, 0].axhline(v03_best_dice, color='b', linestyle='--', alpha=0.5)
axes[0, 0].axhline(v04_best_dice, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Val Dice')
axes[0, 0].set_title('Validation Dice Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Train Dice comparison
axes[0, 1].plot(v03_epochs, v03_history['train_dice'], 'b-', label='V0.3', linewidth=2)
axes[0, 1].plot(v04_epochs, v04_history['train_dice'], 'r-', label='V0.4', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Train Dice')
axes[0, 1].set_title('Training Dice Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Train-Val Gap
v03_gap_history = [t - v for t, v in zip(v03_history['train_dice'], v03_history['val_dice'])]
v04_gap_history = [t - v for t, v in zip(v04_history['train_dice'], v04_history['val_dice'])]

axes[0, 2].plot(v03_epochs, v03_gap_history, 'b-', label='V0.3', linewidth=2)
axes[0, 2].plot(v04_epochs, v04_gap_history, 'r-', label='V0.4', linewidth=2)
axes[0, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Train Dice - Val Dice')
axes[0, 2].set_title('Overfitting Gap')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Learning rate
axes[1, 0].plot(v03_epochs, v03_history['lr'], 'b-', label='V0.3', linewidth=2)
axes[1, 0].plot(v04_epochs, v04_history['lr'], 'r-', label='V0.4', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. V0.4 detailed (first 64 epochs)
axes[1, 1].plot(v04_epochs, v04_history['train_dice'], 'b-', label='Train', linewidth=2)
axes[1, 1].plot(v04_epochs, v04_history['val_dice'], 'r-', label='Val', linewidth=2)
axes[1, 1].axvline(v04_best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best ({v04_best_epoch})')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Dice Score')
axes[1, 1].set_title('V0.4 Training Progress')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Summary text
axes[1, 2].axis('off')
summary_text = f"""
V0.4 Training Summary

Configuration:
  BASE_CHANNELS: 64 → 48
  DEPTH: 5 → 4
  Dropout: 0 → 0.15
  Parameters: 31.4M → ~8.5M (-73%)

Results:
  Best Val Dice: {v04_best_dice:.4f}
  V0.3 Val Dice: {v03_best_dice:.4f}
  Difference: {v04_best_dice - v03_best_dice:.4f} ({(v04_best_dice/v03_best_dice - 1)*100:+.2f}%)

Stopped at epoch {len(v04_epochs)} (patience=20)
Best at epoch {v04_best_epoch}

Diagnosis:
  ⚠ Val Dice降低4.6%
  {'✓' if v04_gap < v03_gap else '⚠'} Train-Val Gap: {v04_gap:.4f} vs {v03_gap:.4f}
  {'✓' if len(v04_epochs) > 50 else '⚠'} Training: {len(v04_epochs)} epochs
"""

axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('V0.4 vs V0.3 Training Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()

output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v04_vs_v03_training_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved to: {output_path}")
