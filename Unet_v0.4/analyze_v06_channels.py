"""Quick channel weight analysis for V0.6"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from model import AttentionUNet

# Load best model
checkpoint_path = Path('checkpoints/best_model.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Create model
model = AttentionUNet(in_channels=7, base_channels=64, depth=5, dropout=0.15)
model.load_state_dict(checkpoint['model_state_dict'])

# Get first conv weights
first_conv_weight = model.encoders[0].conv[0].weight.data  # [64, 7, 3, 3]

# Calculate importance (mean absolute weight per input channel)
channel_importance = first_conv_weight.abs().mean(dim=[0, 2, 3]).numpy()

# To percentage
total = channel_importance.sum()
percentages = (channel_importance / total) * 100

# Channel names
names = ['CLAHE[t-1]', 'CLAHE[t]', 'CLAHE[t+1]', 'T1', 'HighPass', 'Asymmetry', 'SpatialAtlas']

print("="*60)
print("V0.6 Deep Supervision - Channel Weight Analysis")
print("="*60)
print(f"\nModel: Epoch {checkpoint['epoch']}, Val Dice {checkpoint['val_dice']:.4f}\n")

print(f"{'Rank':<6} {'Channel':<18} {'Weight %':<10} {'Rating'}")
print("-"*60)

# Sort by importance
sorted_idx = np.argsort(percentages)[::-1]
for rank, idx in enumerate(sorted_idx):
    pct = percentages[idx]
    if pct > 15:
        rating = "⭐⭐⭐ Critical"
    elif pct > 13:
        rating = "⭐⭐ Important" 
    elif pct > 11:
        rating = "⭐ Useful"
    else:
        rating = "⚠️ Weak"
    
    print(f"#{rank+1:<5} {names[idx]:<18} {pct:>6.2f}%    {rating}")

print("\n" + "="*60)
print("Statistics")
print("="*60)
print(f"Mean:   {percentages.mean():.2f}%")
print(f"Std:    {percentages.std():.2f}%")
print(f"Range:  {percentages.min():.2f}% - {percentages.max():.2f}%")
print(f"Ratio:  {percentages.max()/percentages.min():.2f}x (highest/lowest)")

# Find weak channels
threshold = percentages.mean() - 0.5 * percentages.std()
weak = [i for i in range(7) if percentages[i] < threshold]

print(f"\n" + "="*60)
print("Recommendation")
print("="*60)

if len(weak) > 0:
    print(f"\nWeak channel(s) below {threshold:.2f}%:")
    for idx in weak:
        print(f"  • {names[idx]}: {percentages[idx]:.2f}%")
    print(f"\n→ Consider replacing with stronger features")
else:
    print(f"\n✓ All channels relatively balanced")
    print(f"  Lowest: {names[percentages.argmin()]} ({percentages.min():.2f}%)")
    print(f"  → Still above threshold ({threshold:.2f}%)")
    print(f"\n→ All 7 channels contributing significantly")

print("\n" + "="*60)
