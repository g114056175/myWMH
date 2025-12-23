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
CHECKPOINT_DIR = r'd:/VSCode/AIOT_E1/newUnet2/checkpoints'
LOG_DIR = r'd:/VSCode/AIOT_E1/newUnet2/logs'

# ===== Model Architecture =====
IN_CHANNELS = 7  # CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas
OUT_CHANNELS = 1
BASE_CHANNELS = 64
DEPTH = 5

# ===== Training Hyperparameters =====
NUM_EPOCHS = 150
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ===== Loss Function (Focal Tversky Loss) =====
# Balanced approach: α=0.25, β=0.75 (β/α = 3.0)
TVERSKY_ALPHA = 0.25  # FP weight (increased from 0.2)
TVERSKY_BETA = 0.75   # FN weight (decreased from 0.8)
FOCAL_GAMMA = 1.33

# ===== Early Stopping =====
EARLY_STOP_PATIENCE = 50  # Increased from 25
EARLY_STOP_MIN_DELTA = 0.0002

# ===== Learning Rate Scheduler =====
LR_SCHEDULER_PATIENCE = 12
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
