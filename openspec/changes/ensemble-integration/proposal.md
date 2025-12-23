# Change: Ensemble Model Integration and Comprehensive 3D Evaluation

## Why

Three enhanced WMH segmentation models have been successfully trained following Plan C strategy:
1. **Model 1** (7-slice): Standard 7-slice 2.5D model with champion augmentations
2. **Model 2** (9-slice): Enhanced 9-slice model for increased context
3. **Model 3** (7-slice stable): Stable generalization variant

Current state:
- All three models have been trained with "triple fix" (Focal Tversky Loss, dynamic empty_ratio, 100 epochs)
- Individual model checkpoints exist and are ready for integration
- Need comprehensive 3D volume-level evaluation to measure ensemble performance
- Target metrics: Sensitivity 70%+, Dice 0.77-0.80

This change implements the final ensemble integration and comprehensive evaluation phase.

## What Changes

- **Ensemble Integration**: Combine predictions from three models using uncertainty-guided averaging
- **3D Evaluation**: Perform volume-level evaluation on test set (not 2D slice-wise)
- **Post-processing**: Apply 3D morphological operations and connected component filtering
- **Comprehensive Metrics**: Calculate Dice, Sensitivity, Specificity, Precision at 3D volume level
- **Results Documentation**: Generate detailed analysis comparing ensemble vs individual models

## Impact

- Affected specs: `wmh-ensemble-evaluation` (new capability spec)
- Affected code:
  - Execute `ensemble_3d.py`: Run comprehensive 3D ensemble evaluation
  - Generate results in `predictions_ensemble_3d/`: NIfTI predictions and JSON metrics
- **Deliverables**: 
  - Ensemble 3D evaluation metrics
  - Per-case prediction volumes
  - Comprehensive comparison report
