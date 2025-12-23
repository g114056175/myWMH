# Project Context

## Purpose
Medical imaging AI project focused on **White Matter Hyperintensity (WMH)** detection and analysis. This project aims to develop AI models for automated segmentation and analysis of white matter lesions in brain MRI scans.

## Tech Stack
- **Language**: Python 3.8+
- **Deep Learning**: PyTorch 2.0+ with torchvision 0.15+
- **Medical Imaging**: nibabel 5.1.0+, SimpleITK 2.2.1+
- **Data Augmentation**: albumentations 1.3.0+
- **Scientific Computing**: NumPy 1.24+, SciPy 1.10+
- **Visualization**: matplotlib 3.7+, tqdm for progress bars, ITK-SNAP 4.4.0 for medical image viewing
- **Data Format**: NIfTI (.nii.gz) for brain MRI scans
- **Dataset**: WMH Challenge dataset (Amsterdam, Singapore, Utrecht cohorts)
- **Hardware Target**: NVIDIA RTX 2060S (8GB VRAM)
- **Advanced Frameworks**: nnU-Net for threshold optimization and medical imaging workflows

## Project Conventions

### Code Style
- **Language**: Python with type hints encouraged
- **Naming**: Use descriptive variable names for anatomical structures (e.g., `flair_image`, `wmh_mask`)
- **Modalities**: FLAIR, T1, 3DT1 consistently named in uppercase
- **Medical Imaging**: Follow nnU-Net conventions where applicable
- **File Organization**: Separate concerns (model, dataset, metrics, training)
- **Comments**: Chinese comments acceptable for domain-specific notes (已使用項目中)

### Architecture Patterns
- **Model Architecture**: 2.5D Attention U-Net (see `model_2d.py`)
  - 2.5D input strategy: 3 consecutive slices for spatial context
  - Attention gates on skip connections
  - Lightweight design: ~20M parameters, optimized for 8GB VRAM
- **Data Pipeline**: 
  - `dataset_2d.py`: 2.5D slice extraction with z-score normalization
  - `augmentations.py`: 11 champion-level augmentations
  - `metrics.py`: Dice Loss, Focal Tversky Loss, BCE Loss combinations
- **Training Strategy**: 
  - Mixed precision (AMP) for memory efficiency
  - Gradient accumulation for effective larger batch sizes
  - Early stopping with validation monitoring
- **Modular Design**: 
  - Model definition (`model_2d.py`)
  - Dataset loaders (`dataset_2d.py`)
  - Loss/metrics (`losses.py`, `metrics.py`)
  - Training scripts (`train_2d.py`, `train_all_models.bat`)
  - Evaluation (`evaluate_ensemble_2d.py`)

### Testing Strategy
- **Data Validation**: 
  - NIfTI file integrity checks (`check_training_data.py`)
  - Verify image dimensions and modality consistency
  - Check mask/label alignment and value ranges
- **Model Testing**: 
  - Test set: `data/wmh/test/` for final evaluation
  - Validation split: 20% of training data for model selection
- **Metrics**: 
  - **Primary**: Dice Coefficient (target: 0.78-0.82), Sensitivity/Recall (target: 70%+)
  - **Secondary**: IoU, Specificity, Precision
  - **Loss Functions**: Focal Tversky Loss (sample-wise), combined Dice+BCE Loss
- **Evaluation Scripts**: 
  - 2D slice evaluation: `evaluate_ensemble_2d.py`
  - Report generation: `nnUnet/generate_report_*.py`
  - Test-time augmentation: `tta_inference.py`
- **Performance Benchmarks**: 
  - Training time target: ≤12 hours on RTX 2060S
  - VRAM usage: ≤8GB with mixed precision

### Git Workflow
- **Branch Strategy**: Feature branches recommended for new models/experiments
- **Data Management**: 
  - Large data files in `data/` (should be gitignored)
  - Keep only dataset metadata (e.g., `data/wmh/dataset.json`)
- **Model Weights**: 
  - Checkpoints in `checkpoints_*` directories (gitignored)
  - Version important checkpoints with descriptive names
  - Store training history as JSON for reproducibility
- **Experiment Tracking**: 
  - Document hyperparameters in training scripts
  - Save results to JSON files (e.g., `final_optimized_results.json`)
  - Keep batch scripts (`.bat`) for reproducible training pipelines

## Domain Context

### Medical Imaging Background
- **White Matter Hyperintensities (WMH)**: Bright regions on T2-FLAIR MRI indicating potential small vessel disease, stroke risk, or aging
- **Clinical Importance**: WMH volume and location correlate with cognitive decline and stroke risk
- **Imaging Modalities**: Typically T1, T2, and FLAIR sequences
- **Segmentation Challenge**: Variable appearance, small lesions, class imbalance

### Dataset Structure
- `data/wmh/training/`: Training images and segmentation masks
- `data/wmh/test/`: Test images for validation
- `data/wmh/additional_annotations/`: Extra annotated data
- `data/wmh/dataset.json`: Dataset metadata and splits

## Important Constraints
- **Medical Data Privacy**: HIPAA/GDPR compliance if using patient data
- **File Size**: Medical images are large; optimize memory usage
- **Clinical Validation**: Results should be validated by medical professionals
- **Reproducibility**: Random seeds, versioning for medical AI submissions

## External Dependencies
- **ITK-SNAP**: Medical image segmentation and annotation tool
- **Medical Imaging Libraries**: nibabel, SimpleITK, or similar
- **Deep Learning**: PyTorch or TensorFlow for model development
- **Compute Resources**: GPU required for training medical imaging models
