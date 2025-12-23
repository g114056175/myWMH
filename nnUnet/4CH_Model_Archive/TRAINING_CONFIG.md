# 訓練配置詳細說明

## nnU-Net自動配置

nnU-Net通過分析訓練數據自動生成以下配置：

### 數據預處理

```json
{
  "target_spacing": [0.976599931716919, 1.0],
  "median_image_size": [236.0, 192.0],
  "normalization_schemes": [
    "ZScoreNormalization",  // Channel 0: FLAIR[t-1]
    "ZScoreNormalization",  // Channel 1: FLAIR[t]
    "ZScoreNormalization",  // Channel 2: FLAIR[t+1]
    "ZScoreNormalization"   // Channel 3: T1
  ],
  "resampling": {
    "data": "order 3 (cubic)",
    "seg": "order 1 (nearest)"
  }
}
```

### 網絡架構

```python
Architecture: PlainConvUNet (2D)
Stages: 6
Feature channels: [32, 64, 128, 256, 512, 512]
Kernel sizes: [[3,3], [3,3], [3,3], [3,3], [3,3], [3,3]]
Strides: [[1,1], [2,2], [2,2], [2,2], [2,2], [2,2]]
Conv per stage (encoder): [2, 2, 2, 2, 2, 2]
Conv per stage (decoder): [2, 2, 2, 2, 2]
Normalization: InstanceNorm2d
Activation: LeakyReLU
```

**總參數量**: ~31M parameters

### 訓練超參數

```python
# Optimizer
optimizer = "SGD"
momentum = 0.99
nesterov = True
weight_decay = 3e-5

# Learning Rate
initial_lr = 0.01
lr_scheduler = "PolyLRScheduler"  # Polynomial decay
total_epochs = 1000 (實際訓練141 epochs後手動停止)

# Loss
loss = "DC_and_CE_loss"  # Dice + Cross Entropy
dice_weight = 1.0
ce_weight = 1.0

# Batch Size
batch_size = 66  # 自動根據GPU記憶體調整
patch_size = [256, 192]
```

### 數據增強

nnU-Net默認數據增強pipeline：

```python
augmentations = [
    # Spatial
    "SpatialTransform": {
        "rotation": [-30, 30] degrees,
        "scaling": [0.7, 1.4],
        "elastic_deformation": True
    },
    
    # Intensity
    "GaussianNoise": p=0.1,
    "GaussianBlur": p=0.2,
    "BrightnessMultiplicative": [0.75, 1.25],
    "ContrastAugmentation": p=0.15,
    "GammaTransform": [0.7, 1.5],
    
    # Simulation
    "SimulateLowResolution": p=0.25,
    "MirrorTransform": axes=[0, 1]
}
```

## 訓練過程

### 硬體配置
- **GPU**: NVIDIA RTX 2060 Super (8GB VRAM)
- **RAM**: 32GB
- **Storage**: SSD

### 記憶體使用
- **Training**: ~6GB VRAM
- **Preprocessing**: ~8GB RAM
- **Peak RAM**: ~15GB

### 訓練時間
- **Preprocessing**: ~2分鐘
- **Training**: ~5小時 (141 epochs)
  - Epoch 0-50: ~2小時
  - Epoch 51-141: ~3小時
- **Average per epoch**: ~2分鐘

### 收斂分析

```
Epoch    Pseudo Dice  Val Loss   Learning Rate
----------------------------------------------
0        0.7079       -0.6035    0.0100
30       0.8609       -0.8600    0.0091
50       0.8750       -0.8700    0.0085
100      0.8780       -0.8720    0.0070
141      0.8761       -0.8643    0.0087

狀態: Epoch 100+進入平台期，Dice波動在0.874-0.878
```

## 交叉驗證

當前模型使用 **Fold 0**（5-fold split）：

```
Total: 3340 training slices
Fold 0: 2672 train / 668 validation
```

如需完整5-fold評估：

```powershell
# 訓練其他folds
nnUNetv2_train 002 2d 1 --npz  # Fold 1
nnUNetv2_train 002 2d 2 --npz  # Fold 2
nnUNetv2_train 002 2d 3 --npz  # Fold 3
nnUNetv2_train 002 2d 4 --npz  # Fold 4

# Ensemble預測
nnUNetv2_predict -i input -o output -d 002 -c 2d --folds 0 1 2 3 4
```

## 復現訓練

完整復現此模型的步驟：

### 1. 環境準備
```powershell
pip install nnunetv2
cd d:\VSCode\AIOT_E1\nnUnet
.\setup_env.bat
```

### 2. 數據準備
```powershell
python prepare_4channel_25D.py
```

### 3. 預處理
```powershell
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity
```

### 4. 訓練
```powershell
nnUNetv2_train 002 2d 0 --npz
```

### 5. 評估
```powershell
python evaluate_3d_fast.py
```

---

**注意**: nnU-Net的自動配置保證了高度的可復現性。使用相同的數據和seed應該能得到非常接近的結果。
