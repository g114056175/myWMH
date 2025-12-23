## ADDED Requirements

### Requirement: Three-Model Ensemble Integration
The system MUST integrate predictions from three trained 2.5D WMH segmentation models (Model 1: 7-slice, Model 2: 9-slice, Model 3: 7-slice-stable) using simple averaging ensemble strategy.

#### Scenario: Load Three Models Successfully
- **WHEN** loading models from checkpoints at model1_path, model2_path, and model3_path
- **THEN** all three models load without errors and are set to evaluation mode
- **AND** models are configured with correct n_slices (7, 9, 7 respectively)

### Requirement: 3D Volume-Level Prediction
The system MUST perform predictions at 3D volume level by processing each slice with appropriate context and assembling results into full volume predictions.

#### Scenario: Generate Full 3D Predictions
- **WHEN** a test case FLAIR volume is provided to predict_volume_2_5d()
- **THEN** the function generates a full 3D probability volume (H, W, D)
- **AND** boundary slices are handled by copying nearest valid predictions

#### Scenario: Ensemble Probability Averaging
- **WHEN** three models have generated probability volumes for the same case
- **THEN** simple_ensemble() computes element-wise average of the three volumes
- **AND** output probabilities remain in [0, 1] range

### Requirement: 3D Post-Processing Pipeline
The system MUST apply 3D morphological operations and connected component filtering to convert probability volumes to clean binary masks.

#### Scenario: Apply Morphological Smoothing
- **WHEN** postprocess_3d() is called with ensemble probability volume
- **THEN** threshold is applied (default 0.45)
- **AND** 3D connected component labeling identifies all regions
- **AND** small components (\u003cmin_size pixels) are removed
- **AND** binary closing and opening operations smooth the mask

### Requirement: 3D Volume-Level Metrics
The system MUST calculate Dice coefficient, Sensitivity, Specificity, and Precision at 3D volume level for clinically meaningful evaluation.

#### Scenario: Calculate 3D Metrics
- **WHEN** evaluate_3d_volume() receives binary prediction and ground truth masks
- **THEN** Dice, Sensitivity, Specificity, and Precision are computed on flattened 3D volumes
- **AND** metrics are returned as float values with epsilon handling to avoid division by zero

#### Scenario: Aggregate Results Across Test Set
- **WHEN** evaluate_ensemble() processes all test cases
- **THEN** per-case metrics are saved in individual_results list
- **AND** average metrics with standard deviations are calculated
- **AND** results are saved to JSON file with configuration parameters

### Requirement: Target Performance Achievement
The ensemble MUST be evaluated against target metrics of Sensitivity ≥70% and Dice ≥0.77 across the test set.

#### Scenario: Evaluate Against Performance Targets
- **WHEN** ensemble evaluation completes on the full test set
- **THEN** average 3D Dice and Sensitivity metrics are calculated
- **AND** results are compared against targets (Sensitivity ≥70%, Dice ≥0.77)
- **AND** comparison shows whether targets are met

