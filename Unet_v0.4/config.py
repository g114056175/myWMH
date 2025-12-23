"""
V0.3 Configuration - 7-channel with Dual Attention
Enhanced from V0.2 with Asymmetry and Spatial Atlas features
"""

import os

# ===== Version Info =====
VERSION = "v0.3"
VERSION_NOTES = "7-channel: CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas + Dual Attention"

# ===== Paths =====
DATA_ROOT = r'd:/VSCode/AIOT_E1/data/wmh'
TRAIN_DIR = os.path.join(DATA_ROOT, 'training')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
CHECKPOINT_DIR = r'd:/VSCode/AIOT_E1/Unet_v0.4/checkpoints'  # V0.4
LOG_DIR = r'd:/VSCode/AIOT_E1/Unet_v0.4/logs'  # V0.4

# ===== Model Architecture =====
IN_CHANNELS = 7  # CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas
OUT_CHANNELS = 1
BASE_CHANNELS = 64  # V0.3 proven capacity
DEPTH = 5  # V0.3 depth
DROPOUT = 0.20  # V0.7: Stronger regularization to combat overfitting

# ===== Training Hyperparameters =====
NUM_EPOCHS = 200  # Keep higher epochs
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ===== Loss Function (Focal Tversky Loss) =====
# New: Reduce FP by increasing α (FP weight)
TVERSKY_ALPHA = 0.35  # FP weight (from 0.25, β/α=1.86)
TVERSKY_BETA = 0.65   # FN weight (from 0.75, more balanced)
FOCAL_GAMMA = 1.33

# ===== Early Stopping =====
EARLY_STOP_PATIENCE = 20  # V0.4: Reduced from 50 (more aggressive)
EARLY_STOP_MIN_DELTA = 0.002  # V0.4: Increased from 0.0002

# ===== Learning Rate Scheduler (CosineAnnealingWarmRestarts) =====
LR_COSINE_T0 = 100  # First restart period (from 50, keep LR higher longer)
LR_SCHEDULER_PATIENCE = 12  # (not used with Cosine)
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-7

# ===== Image Processing (CRITICAL - CH4 PURE HIGHPASS) =====
TARGET_SIZE = 224
CLAHE_CLIP_LIMIT = 2.0
HIGHPASS_SIGMA = 2.0  # Balanced edge detection (changed from 1.5)

# ===== Data Augmentation (AGGRESSIVE 3-5x) =====
# Geometric
AUG_HORIZONTAL_FLIP = 0.8
AUG_VERTICAL_FLIP = 0.6
AUG_ROTATE_PROB = 0.9
AUG_ROTATE_LIMIT = 25
AUG_RANDOM_SCALE = 0.7
AUG_SCALE_LIMIT = 0.15
AUG_SHIFT_PROB = 0.6
AUG_SHIFT_LIMIT = 0.1
AUG_ELASTIC_PROB = 0.6
AUG_ELASTIC_ALPHA = 80
AUG_ELASTIC_SIGMA = 8

# Intensity
AUG_BRIGHTNESS_CONTRAST = 0.9
AUG_BC_BRIGHTNESS_LIMIT = 0.3
AUG_BC_CONTRAST_LIMIT = 0.3
AUG_GAMMA_PROB = 0.7
AUG_GAMMA_LIMIT = (70, 130)
AUG_GAUSS_NOISE_PROB = 0.4
AUG_GAUSS_NOISE_VAR = (5, 20)

# Blur/Sharpen
AUG_BLUR_SHARPEN_PROB = 0.4
AUG_BLUR_LIMIT = 5

# Dropout
AUG_COARSE_DROPOUT_PROB = 0.4
AUG_DROPOUT_HOLES = 10
AUG_DROPOUT_SIZE = 20

# ===== Training Settings =====
USE_AMP = True
GRADIENT_CLIP_VALUE = 1.0
GRADIENT_ACCUMULATION_STEPS = 2

# ===== Hardware =====
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = 'cuda'

# ===== Validation =====
VAL_SPLIT = 0.2

# ===== Monitoring =====
SAVE_FREQUENCY = 5
LOG_FREQUENCY = 10

if __name__ == '__main__':
    print("="*60)
    print(f"Configuration V2 - {VERSION_NOTES}")
    print("="*60)
    print(f"Version: {VERSION}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Log Dir: {LOG_DIR}")
    print(f"\nKEY CHANGES FROM V1:")
    print(f"  ✓ ch4: Orig + {DETAIL_ENHANCEMENT_STRENGTH}*HighPass(σ={HIGHPASS_SIGMA})")
    print(f"  ✓ Tversky: α={TVERSKY_ALPHA}, β={TVERSKY_BETA} (β/α={TVERSKY_BETA/TVERSKY_ALPHA:.2f})")
    print(f"  ✓ Data Aug: 3-5x aggressive")
    print(f"  ✓ Early Stop: patience={EARLY_STOP_PATIENCE}")
    print("="*60)
