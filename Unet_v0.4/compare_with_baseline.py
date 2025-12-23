"""
V0.3+Dropout vs V0.3 baseline comparison
"""

import json
import numpy as np
from pathlib import Path

# Load V0.3+Dropout history
v03_dropout_path = Path('d:/VSCode/AIOT_E1/Unet_v0.4/logs/training_history.json')
with open(v03_dropout_path, 'r') as f:
    v03_dropout = json.load(f)

# Load V0.3 baseline
v03_baseline_path = Path('d:/VSCode/AIOT_E1/newUnet2/logs/training_history.json')
with open(v03_baseline_path, 'r') as f:
    v03_baseline = json.load(f)

# Find best epochs
dropout_best_epoch = np.argmax(v03_dropout['val_dice']) + 1
dropout_best_dice = max(v03_dropout['val_dice'])

baseline_best_epoch = np.argmax(v03_baseline['val_dice']) + 1
baseline_best_dice = max(v03_baseline['val_dice'])

print("="*70)
print("V0.3+Dropout vs V0.3 Baseline Comparison")
print("="*70)

print(f"\nV0.3 Baseline:")
print(f"  Architecture: 64ch/D5, No Dropout")
print(f"  Best Epoch: {baseline_best_epoch}")
print(f"  Best Val Dice: {baseline_best_dice:.4f}")
print(f"  Final Train Dice: {v03_baseline['train_dice'][baseline_best_epoch-1]:.4f}")
print(f"  Train-Val Gap: {(v03_baseline['train_dice'][baseline_best_epoch-1] - baseline_best_dice):.4f} ({(v03_baseline['train_dice'][baseline_best_epoch-1] - baseline_best_dice)/baseline_best_dice*100:.1f}%)")
print(f"  Total Epochs: {len(v03_baseline['train_dice'])}")

print(f"\nV0.3+Dropout:")
print(f"  Architecture: 64ch/D5, Dropout 0.08")
print(f"  Best Epoch: {dropout_best_epoch}")
print(f"  Best Val Dice: {dropout_best_dice:.4f}")
print(f"  Final Train Dice: {v03_dropout['train_dice'][dropout_best_epoch-1]:.4f}")
print(f"  Train-Val Gap: {(v03_dropout['train_dice'][dropout_best_epoch-1] - dropout_best_dice):.4f} ({(v03_dropout['train_dice'][dropout_best_epoch-1] - dropout_best_dice)/dropout_best_dice*100:.1f}%)")
print(f"  Total Epochs: {len(v03_dropout['train_dice'])} (early stopped)")

print(f"\n" + "="*70)
print("Improvement Analysis")
print("="*70)

val_improvement = dropout_best_dice - baseline_best_dice
gap_baseline = v03_baseline['train_dice'][baseline_best_epoch-1] - baseline_best_dice
gap_dropout = v03_dropout['train_dice'][dropout_best_epoch-1] - dropout_best_dice

print(f"\nVal Dice:")
print(f"  Baseline: {baseline_best_dice:.4f}")
print(f"  +Dropout: {dropout_best_dice:.4f}")
print(f"  Change: {val_improvement:+.4f} ({val_improvement/baseline_best_dice*100:+.2f}%)")

print(f"\nTrain-Val Gap:")
print(f"  Baseline: {gap_baseline:.4f} ({gap_baseline/baseline_best_dice*100:.1f}%)")
print(f"  +Dropout: {gap_dropout:.4f} ({gap_dropout/dropout_best_dice*100:.1f}%)")
print(f"  Reduction: {gap_baseline - gap_dropout:.4f} ({(gap_baseline - gap_dropout)/gap_baseline*100:.1f}%)")

print(f"\nTraining Efficiency:")
print(f"  Baseline: Best at epoch {baseline_best_epoch}, ran {len(v03_baseline['train_dice'])} epochs")
print(f"  +Dropout: Best at epoch {dropout_best_epoch}, ran {len(v03_dropout['train_dice'])} epochs (early stop)")
print(f"  Time saved: ~{(len(v03_baseline['train_dice']) - len(v03_dropout['train_dice'])) * 2.5 / 60:.1f} hours")

print("\n" + "="*70)
print("✓ Analysis complete")
print("="*70)
