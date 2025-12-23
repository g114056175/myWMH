# nnU-Net 7通道v2模型（Butterworth版）

**版本**: v2.0  
**訓練日期**: 2025-12-22  
**訓練時長**: ~3.5小時 (Epoch 102, 提前停止)  
**Test Overall Dice**: 0.8575

---

## 📋 模型概述

### 通道配置
此模型使用**7通道**輸入，相比v1用**Butterworth高通**替代CLAHE：

| 通道 | 內容 | 目的 |
|------|------|------|
| 0 | FLAIR[t-1] | 時序上下文 |
| 1 | FLAIR[t] | 主要病灶 |
| 2 | FLAIR[t+1] | 時序下下文 |
| 3 | T1 | 解剖結構 |
| 4 | **Butterworth(cutoff=15)** | **小病灶邊緣** ⭐ NEW |
| 5 | HighPass σ=2 | 中等邊緣 |
| 6 | Top-hat | 小亮點檢測 |

### 關鍵改進
**v1 → v2 變更**:
- ❌ 移除: CLAHE (v1貢獻僅10.28%)
- ✅ 新增: Butterworth高通 (cutoff=15, order=2)
- 🎯 目標: 改善中小型病灶檢測 (100-1k voxels)

---

## 🎯 性能表現

### 測試集結果 (110 cases)

| 指標 | v2 (Butterworth) | v1 (CLAHE) | 改進 |
|------|------------------|------------|------|
| **Overall Dice** | **0.8575** | 0.8549 | **+0.26%** ✅ |
| **Mean Dice** | **0.7992** | 0.7954 | **+0.38%** ✅ |
| Training Time | 3.5h (102 ep) | 9h (282 ep) | **-61%** ⚡ |

### vs 4通道 Baseline

| 指標 | 4CH | 7CH v2 | 改進 |
|------|-----|--------|------|
| Overall Dice | 0.8331 | **0.8575** | **+2.93%** |
| Mean Dice | 0.7983 | **0.7992** | **+0.11%** |

---

## 🚀 使用方法

### 1. 環境設置

```powershell
cd d:\VSCode\AIOT_E1\nnUnet
.\setup_env.bat
```

### 2. 數據準備

使用`prepare_7channel_v2_butterworth.py`準備測試數據：

```powershell
python 7CH_v2_Archive/prepare_7channel_v2_butterworth.py
```

**輸出**: Dataset004_WMH_7CH_v2格式數據

### 3. 推理

```powershell
nnUNetv2_predict `
  -i nnUNet_raw/Dataset004_WMH_7CH_v2/imagesTs `
  -o predictions `
  -d 004 `
  -c 2d `
  -f 0 `
  -chk 7CH_v2_Archive/checkpoint_best.pth
```

---

## ⚙️ 配置詳情

### nnU-Net自動配置

```json
{
  "batch_size": 49,
  "patch_size": [256, 256],
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
- Final LR: 0.00908 (Epoch 102)
- Loss: Dice + CE
- **提前停止**: Epoch 102 (原計劃1000)

---

## 📂 歸檔內容

```
7CH_v2_Archive/
├── README.md                                # 本文檔
├── checkpoint_best.pth                      # 最佳模型 (~256MB)
├── checkpoint_latest.pth                    # 最新模型 (Epoch 102)
├── nnUNetPlans.json                         # 配置文件
├── dataset.json                             # 數據集定義
├── prepare_7channel_v2_butterworth.py      # 數據準備腳本
├── training_curves.png                      # 訓練曲線
├── evaluation_results.png                   # 評估結果
├── evaluation_metrics.json                  # 詳細metrics
├── TRAINING_LOG.md                          # 訓練記錄
└── EVALUATION_REPORT.md                     # 評估報告
```

---

## 🔍 Butterworth特徵實現

### 技術細節
```python
def butterworth_highpass(image, cutoff=15, order=2):
    # 頻域濾波器
    H(u,v) = 1 / (1 + (cutoff/D)^(2*order))
    
    # cutoff=15: 專門捕捉5-15 pixel的小病灶邊緣
    # order=2: 平滑過渡，無Gibbs振鈴
    # 比簡單高通(σ=2)更精確的頻率控制
```

### 優勢
1. **精確頻率控制** - cutoff=15專攻小病灶
2. **無偽影** - 無Gibbs振鈴（vs FFT理想高通）
3. **互補** - 與HighPass σ=2互補（小vs中病灶）

---

## 📊 訓練詳情

| Metric | 值 |
|--------|-----|
| 訓練Epochs | 102 |
| 最佳Pseudo Dice | 0.8763 (Epoch 70) |
| 最終Pseudo Dice | 0.8702 (Epoch 102) |
| 訓練時長 | 3.5小時 |

---

## 💡 使用建議

### 適用場景
✅ 中小型WMH病灶檢測  
✅ 需要高精度邊緣檢測  
✅ 多站點數據泛化

### 限制
⚠️ 訓練時間較短（102 epochs），可能還有提升空間  
⚠️ vs v1改進有限（+0.26-0.38%）  
⚠️ 仍需改進中型病灶檢測

---

## 📝 引用

```
Dataset: WMH Challenge (Amsterdam, Singapore, Utrecht)
Framework: nnU-Net v2
Architecture: PlainConvUNet (2D)
Feature: Butterworth Highpass (cutoff=15, order=2)
```

---

**創建日期**: 2025-12-22  
**狀態**: 已完成測試集評估，可用於推理
