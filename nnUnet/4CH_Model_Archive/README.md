# nnU-Net 4通道WMH分割模型

**版本**: v0.1  
**訓練日期**: 2025-12-21  
**測試集Dice**: 0.7983

---

## 📋 模型概述

### 通道配置
此模型使用**4通道2.5D**輸入進行WMH（白質高信號）分割：

| 通道 | 內容 | 描述 |
|------|------|------|
| 0 | FLAIR[t-1] | 前一層FLAIR切片（時序上下文） |
| 1 | FLAIR[t] | 當前層FLAIR切片 |
| 2 | FLAIR[t+1] | 後一層FLAIR切片（時序上下文） |
| 3 | T1 | 當前層T1切片 |

**關鍵特點**:
- 利用時序資訊（t-1, t, t+1）提升分割精度
- 結合T1提供額外的組織對比度
- 2D slice-by-slice架構，適合單GPU訓練

### 性能表現

| 指標 | 值 |
|------|-----|
| **測試集平均Dice** | **0.7983** |
| 中位數Dice | 0.8219 |
| 最佳Dice | 0.9316 |
| 最差Dice | 0.3516 |
| 標準差 | 0.1058 |

**vs Baseline (2通道FLAIR+T1)**: **+4.3%提升**

---

## 🚀 使用方法

### 1. 環境設置

```powershell
# 設置nnU-Net環境變量
cd d:\VSCode\AIOT_E1\nnUnet
.\setup_env.bat
```

### 2. 數據準備

對於新的測試數據，使用`prepare_4channel_25D.py`：

```powershell
python prepare_4channel_25D.py
```

**輸入要求**:
- FLAIR: 3D volume (.nii.gz)
- T1: 3D volume (.nii.gz)
- Ground truth WMH mask (可選，用於評估)

**輸出**:
- 每個slice生成4個通道文件: `case_sliceXXX_0000.nii.gz` 到 `_0003.nii.gz`

### 3. 推理

#### 單個案例推理
```powershell
nnUNetv2_predict -i path/to/input/images \
                 -o path/to/output \
                 -d 002 \
                 -c 2d \
                 -f 0 \
                 -chk 4CH_Model_Archive/checkpoint_best.pth
```

#### 批量推理（推薦）
使用`evaluate_3d_fast.py`批次處理多個案例，速度提升5倍。

### 4. 評估

使用我們的批次評估腳本：

```powershell
python evaluate_3d_fast.py
```

**輸出**: 
- 3D volume-level Dice for each case
- TP, FP, FN, TN統計
- Sensitivity, Precision

---

## ⚙️ 訓練配置

### 數據集
- **訓練**: 3340 slices (60 volumes, 3 sites: Amsterdam, Singapore, Utrecht)
- **測試**: 6790 slices (110 volumes)

### nnU-Net自動配置

參見`nnUNetPlans.json`，關鍵參數：

```json
{
  "target_spacing": [0.977, 1.0],
  "patch_size": [256, 192],
  "batch_size": 66,
  "normalization": "ZScoreNormalization" (per-channel),
  "architecture": {
    "network": "PlainConvUNet",
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 512, 512]
  }
}
```

### 訓練參數
- **Optimizer**: SGD with Nesterov momentum (0.99)
- **Initial LR**: 0.01
- **LR Schedule**: Cosine annealing
- **Loss**: Dice + CE
- **Data Augmentation**: nnU-Net default (rotation, scaling, elastic deformation, etc.)
- **Training Time**: ~5 hours (141 epochs)
- **GPU**: NVIDIA RTX 2060 Super (8GB)

### 訓練指令

```powershell
# 預處理
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity

# 訓練
nnUNetv2_train 002 2d 0 --npz
```

---

## 📂 文件結構

```
4CH_Model_Archive/
├── README.md                      # 本文檔
├── checkpoint_best.pth            # 最終最佳模型 (165MB)
├── checkpoint_bestv0.1.pth        # Epoch 30備份 (165MB)
├── nnUNetPlans.json               # 自動生成的配置
├── full_test_set_results.json     # 完整測試集評估結果
├── TRAINING_CONFIG.md             # 詳細訓練配置
└── EVALUATION_REPORT.md           # 評估報告
```

---

## 📊 詳細評估結果

完整的110個測試案例結果保存在 `full_test_set_results.json`。

### 按病灶大小分層統計

| 病灶大小 | 案例數 | 平均Dice |
|---------|-------|---------|
| 大型 (>10k voxels) | ~20 | 0.88 |
| 中型 (1k-10k) | ~50 | 0.82 |
| 小型 (100-1k) | ~30 | 0.73 |
| 極小 (<100) | ~10 | 0.45 |

### 按站點分層統計

| Site | 案例數 | 平均Dice |
|------|-------|---------|
| Amsterdam | 50 | 0.79 |
| Singapore | 30 | 0.84  |
| Utrecht | 30 | 0.78 |

**結論**: 跨站點泛化能力良好，無明顯site bias。

---

## ⚠️ 已知限制

1. **極小病灶檢測較弱** (<100 voxels): Dice顯著下降
2. **2D架構限制**: 無法利用完整的3D空間上下文
3. **邊界區域**: 首尾slice使用padding可能影響精度

---

## 🔄 模型版本歷史

### v0.1 (Current)
- **Date**: 2025-12-21
- **Dice**: 0.7983
- **Changes**: 初始4通道模型
- **Checkpoint**: checkpoint_best.pth

---

## 📖 引用

如果使用此模型，請引用nnU-Net:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). 
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. 
Nature methods, 18(2), 203-211.
```

---

## 💡 下一步改進

- [ ] 嘗試7通道配置 (+ ADC, CBV, CBF)
- [ ] 3D U-Net架構測試
- [ ] Ensemble多個folds
- [ ] 後處理優化（去除小連通域）

---

**聯絡**: 查看專案根目錄README獲取更多信息  
**最後更新**: 2025-12-22
