# nnU-Net 7通道v1模型（CLAHE版）

**版本**: v1.0  
**訓練日期**: 2025-12-22  
**訓練時長**: 9小時 (Epoch 282)  
**Pseudo Dice**: 0.8855

---

## 📋 模型概述

### 通道配置
此模型使用**7通道**輸入：

| 通道 | 內容 | 目的 |
|------|------|------|
| 0 | FLAIR[t-1] | 時序上下文 |
| 1 | FLAIR[t] | 主要病灶 |
| 2 | FLAIR[t+1] | 時序下下文 |
| 3 | T1 | 解剖結構 |
| 4 | CLAHE | 對比度增強 |
| 5 | Top-hat | 小亮點檢測 |
| 6 | HighPass σ=2 | 邊緣特徵 |

### 性能表現

| 指標 | 值 |
|------|-----|
| **訓練Pseudo Dice** | **0.8855** |
| 訓練Epochs | 282 |
| 訓練時長 | 9小時 |
| **vs 4通道** | **+1.1%** (4CH: 0.8761) |

**通道重要性分析**:
- HighPass σ=2: 22.92% (最重要)
- FLAIR[t]: 19.59%
- T1: 16.33%
- Top-hat: 12.51%
- CLAHE: 10.28% (最低)

---

## 🚀 使用方法

### 1. 環境設置

```powershell
cd d:\VSCode\AIOT_E1\nnUnet
.\setup_env.bat
```

### 2. 數據準備

使用`prepare_7channel_features.py`準備測試數據：

```powershell
python prepare_7channel_features.py
```

**輸出**: Dataset003_WMH_7CH格式數據

### 3. 推理

```powershell
nnUNetv2_predict `
  -i nnUNet_raw/Dataset003_WMH_7CH/imagesTs `
  -o predictions `
  -d 003 `
  -c 2d `
  -f 0 `
  -chk 7CH_v1_Archive/checkpoint_best.pth
```

---

## ⚙️ 配置詳情

### nnU-Net自動配置

```json
{
  "batch_size": 49,
  "patch_size": [256, 256],
  "target_spacing": [0.977, 1.0],
  "normalization": "ZScoreNormalization" (per-channel),
  "architecture": {
    "n_stages": 7,
    "features_per_stage": [32, 64, 128, 256, 512, 512, 512]
  }
}
```

### 訓練參數

- Optimizer: SGD (momentum 0.99, Nesterov)
- Initial LR: 0.01
- Final LR: 0.0074 (Epoch 282)
- Loss: Dice + CE
- Data Augmentation: nnU-Net default

---

## 📂 歸檔內容

```
7CH_v1_Archive/
├── README.md                      # 本文檔
├── checkpoint_best.pth            # 最佳模型 (~157MB)
├── checkpoint_latest.pth          # 最新模型 (Epoch 282)
├── nnUNetPlans.json               # 配置文件
├── dataset.json                   # 數據集定義
├── prepare_7channel_features.py  # 數據準備腳本
└── TRAINING_LOG.md                # 訓練記錄
```

---

## 🔍 特徵實現細節

### CLAHE (對比度增強)
```python
cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
```

### Top-hat (形態學)
```python
grey_opening(image, size=(15, 15))
tophat = image - opened
```

### HighPass (高通濾波)
```python
blurred = gaussian_filter(image, sigma=2.0)
highpass = image - blurred
```

---

## 📊 與其他版本對比

| 模型 | 通道 | Pseudo Dice | 改進 |
|------|------|-------------|------|
| 4通道 | FLAIR×3 + T1 | 0.8761 | baseline |
| **7通道v1** | **+CLAHE+Top-hat+HighPass** | **0.8855** | **+1.1%** |
| 7通道v2 (計劃) | +Butterworth替代CLAHE | 待訓練 | ? |

---

## ⚠️ 已知限制

1. **CLAHE貢獻較低** (10.28%)
   - 低於預期
   - 可能與Z-score normalization重疊

2. **時序資訊價值有限**
   - FLAIR[t±1]合計18.37%
   - 低於單個FLAIR[t] (19.59%)

3. **未完成3D評估**
   - 僅有Pseudo Dice (slice-level)
   - 需要完整測試集3D評估

---

## 🔄 後續版本

### v2改進計劃
- ✅ 用Butterworth(cutoff=15)替代CLAHE
- ✅ 專攻小病灶檢測 (5-15 pixels)
- ✅ 精確頻率控制，無Gibbs振鈴

---

## 📝 引用

```
Dataset: WMH Challenge (Amsterdam, Singapore, Utrecht)
Framework: nnU-Net v2
Architecture: PlainConvUNet (2D)
```

---

**創建日期**: 2025-12-22  
**狀態**: 已歸檔，準備訓練v2
