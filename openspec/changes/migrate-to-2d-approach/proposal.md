# Change: Migrate to Pure 2D Segmentation Approach

## Why

The 2.5D/3D ensemble approach yielded critically poor performance:
- **3D Dice: 0.047** (target: 0.77) - only 6% of target achieved
- **False Positive Rate**: 50-500× higher than True Positives
- **3D Precision: 0.028** - 97% of predictions were incorrect

Root cause analysis revealed:
1. **Spatial context introduced noise**: WMH lesions are primarily 2D bright spots in individual slices
2. **Over-reliance on adjacent slices**: 2.5D models may have learned spurious spatial correlations
3. **Challenge winner used pure 2D**: The PGS team achieved top performance using per-slice 2D prediction

**PGS Strategy (Challenge Winner)**:
- Pure 2D U-Net per slice
- Aggressive TTA: 4 transforms per prediction
- Multi-model ensemble: 5 models × 4 TTA = 20 predictions averaged
- 3D reconstruction via stacking
- Advanced image preprocessing (CLAHE, edge enhancement)

## What Changes

### Architectural Shift
- **FROM**: 2.5D Attention U-Net (7/9-slice context)
- **TO**: Pure 2D U-Net (single slice input)

### New Components
- **2D Model Architecture**: Lightweight U-Net for single-slice segmentation
- **TTA Pipeline**: 4-way augmentation (original, v-flip, h-flip, both flips)
- **Multi-Model Ensemble**: Train 5 diverse models with different loss functions
- **Image Preprocessing**: CLAHE contrast enhancement, edge detection, morphological filtering
- **3D Reconstruction**: Stack 2D predictions back to 3D volumes with post-processing

### Training Strategy
- 5 independent 2D models with:
  - Different loss functions (BCE, Focal, Dice, Tversky, Combo)
  - Different architectures (standard/deep/lightweight U-Net)
  - Different augmentation strategies
  - Different random seeds

### Inference Pipeline
```
For each test volume:
  For each 2D slice:
    predictions = []
    For each of 5 models:
      For each of 4 TTA transforms:
        pred = model(transform(slice))
        predictions.append(inverse_transform(pred))
    final_pred = mean(predictions)  # 20 predictions averaged
  Stack 2D predictions → 3D volume
  Apply 3D post-processing
  Calculate 3D metrics
```

## Impact

- **Affected specs**: 
  - `wmh-2d-segmentation` (NEW capability)
  - `wmh-ensemble-evaluation` (ARCHIVED - replaced by 2D approach)
  
- **Deleted code**:
  - All 2.5D training scripts (train*.py)
  - All 2.5D models (model.py, checkpoints/)
  - All 2.5D datasets (dataset*.py)
  - Ensemble 3D evaluation (ensemble_3d.py)
  - Results and predictions (~3-4GB of data)

- **New code**:
  - `model_2d.py`: Pure 2D U-Net architecture
  - `dataset_2d.py`: 2D slice dataset loader
  - `train_2d.py`: 2D training script
  - `preprocessing.py`: Image enhancement (CLAHE, edge enhancement)
  - `tta_inference.py`: TTA prediction pipeline
  - `reconstruct_3d.py`: 2D to 3D stacking with post-processing
  - `evaluate_2d.py`: Complete evaluation pipeline

- **BREAKING**: Complete architectural change - no backward compatibility with 2.5D models

## Expected Performance

Conservative estimates based on PGS approach:
- **3D Dice**: 0.70 - 0.78 (vs current 0.047)
- **3D Sensitivity**: 0.65 - 0.75 (vs current 0.543)
- **3D Precision**: 0.50 - 0.70 (vs current 0.028) - **major improvement**
- **3D Specificity**: >0.98 (maintain current 0.977)

Key improvement: Dramatically reduced false positive rate through:
- Focused 2D lesion detection
- 20× prediction averaging (5 models × 4 TTA)
- Advanced image preprocessing
- Conservative 3D post-processing
