## ADDED Requirements

### Requirement: Pure 2D Slice Segmentation
The system MUST perform WMH segmentation on individual 2D FLAIR slices without using adjacent slice context.

#### Scenario: Single Slice Prediction
- **WHEN** a 2D FLAIR slice is provided to the model
- **THEN** the model outputs a 2D probability map of the same dimensions
- **AND** prediction does not depend on adjacent slices

#### Scenario: Model Architecture
- **WHEN** defining the 2D U-Net architecture
- **THEN** input shape is (B, 1, H, W) for single-channel 2D slices
- **AND** output shape is (B, 1, H, W) for binary segmentation

### Requirement: Test-Time Augmentation (TTA)
The system MUST apply 4-way test-time augmentation to improve prediction robustness.

#### Scenario: Four-Way TTA Transform
- **WHEN** predicting a single slice
- **THEN** the system applies 4 transforms: original, vertical flip, horizontal flip, and both flips
- **AND** predictions from all 4 transforms are averaged after inverse transformation

#### Scenario: TTA Consistency
- **WHEN** the same slice is predicted multiple times with TTA
- **THEN** prediction variance is significantly lower than without TTA
- **AND** confidence in lesion detection increases

### Requirement: Multi-Model Ensemble
The system MUST train and ensemble at least 5 independent 2D models with diverse configurations.

#### Scenario: Model Diversity
- **WHEN** training 5 models for ensemble
- **THEN** each model uses a different loss function (BCE, Focal, Dice, Tversky, Combo)
- **AND** models use different random seeds and augmentation strategies

#### Scenario: Ensemble Prediction
- **WHEN** predicting with ensemble
- **THEN** all 5 models generate predictions for the slice
- **AND** each model applies 4-way TTA
- **AND** final prediction is the mean of 20 predictions (5 models × 4 TTA)

### Requirement: Advanced Image Preprocessing
The system MUST apply CLAHE contrast enhancement and optional edge enhancement to improve lesion visibility.

#### Scenario: CLAHE Enhancement
- **WHEN** preprocessing a FLAIR slice
- **THEN** CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied
- **AND** clip limit and tile grid size are configurable
- **AND** local contrast is enhanced while avoiding over-amplification

#### Scenario: Edge Enhancement
- **WHEN** edge enhancement is enabled
- **THEN** Laplacian edge detection is applied
- **AND** edges are weighted and added back to the original image
- **AND** lesion boundaries become more prominent

### Requirement: 3D Volume Reconstruction
The system MUST reconstruct 3D volumes from 2D slice predictions and apply 3D post-processing.

#### Scenario: 2D to 3D Stacking
- **WHEN** all slices in a volume have been predicted
- **THEN** 2D predictions are stacked by slice index to form 3D volume
- **AND** output NIfTI file preserves original affine and header

#### Scenario: 3D Connected Component Filtering
- **WHEN** 3D volume is reconstructed
- **THEN** 3D connected component analysis identifies all regions
- **AND** regions smaller than min_volume threshold are removed
- **AND** morphological opening and closing are applied for smoothing

### Requirement: Performance Target Achievement
The system MUST achieve 3D Dice ≥0.70 and 3D Precision ≥0.50 on the test set.

#### Scenario: Evaluate Against Performance Targets
- **WHEN** evaluation completes on the test set
- **THEN** average 3D Dice is calculated across all test cases
- **AND** average 3D Precision is calculated
- **AND** results are compared against targets (Dice ≥0.70, Precision ≥0.50)
- **AND** performance metrics show significant improvement over 2.5D baseline (Dice 0.047, Precision 0.028)
