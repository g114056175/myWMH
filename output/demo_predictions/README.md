# Demo Predictions - 7CH v2 Model

## Model Information
- **Model**: 7-Channel v2 with Butterworth High-Pass Filter
- **Checkpoint**: checkpoint_best.pth (Epoch 70)
- **Overall Performance**: Dice 0.8575 on 110 test cases

## Demo Cases

1. **Amsterdam_GE1T5_150** - Volume: 968 voxels, Dice: 0.7157
2. **Amsterdam_GE3T_117** - Volume: 13342 voxels, Dice: 0.9007
3. **Singapore_70** - Volume: 2456 voxels, Dice: 0.7316
4. **Singapore_95** - Volume: 6007 voxels, Dice: 0.9021
5. **Utrecht_32** - Volume: 5012 voxels, Dice: 0.7375

## Visualization Guide

Each image shows:
- **Left**: FLAIR Original MRI image
- **Middle**: Ground Truth (Red = WMH lesions)
- **Right**: Model Prediction (Green = Predicted lesions)

## Files
- `demo_1_Amsterdam_GE1T5_150.png`
- `demo_2_Amsterdam_GE3T_117.png`
- `demo_3_Singapore_70.png`
- `demo_4_Singapore_95.png`
- `demo_5_Utrecht_32.png`

## Notes
- These are slice-level predictions for demonstration
- Full 3D evaluation achieves higher Dice scores
- Green overlaps with Red indicate correct predictions
