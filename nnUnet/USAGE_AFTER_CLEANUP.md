# nnUnet 資料夾使用說明（清理後）

## 📁 資料夾結構（清理後）

```
nnUnet/
├── 4CH_Model_Archive/              # 4通道模型（baseline）
│   ├── checkpoint_bestv0.1.pth     # 最佳權重
│   ├── README.md                   # 使用說明
│   └── EVALUATION_REPORT.md        # 評估報告
│
├── 7CH_v1_Archive/                 # 7通道v1（CLAHE版）
│   ├── checkpoint_best.pth         # 最佳權重
│   ├── README.md                   # 使用說明
│   └── EVALUATION_REPORT.md        # 評估報告
│
├── 7CH_v2_Archive/                 # 7通道v2（Butterworth版）⭐
│   ├── checkpoint_best.pth         # 最佳權重
│   ├── README.md                   # 使用說明
│   ├── EVALUATION_REPORT.md        # 評估報告
│   └── CHANNEL_IMPORTANCE.md       # 通道重要性分析
│
├── prepare_4channel_25D.py         # 4通道數據準備
├── prepare_7channel_features.py    # 7通道v1數據準備
├── prepare_7channel_v2_butterworth.py  # 7通道v2數據準備⭐
│
└── setup_env.bat                   # 環境設置腳本
```

---

## ❌ 已刪除內容

### 大型數據文件夾（已移除）
- `nnUNet_preprocessed/` (~6.3 GB) - 預處理切片數據
- `nnUNet_raw/` - 原始訓練數據副本
- `temp_v2_eval/` - 臨時評估資料夾
- `batch_eval_7ch_v2/` - 批量評估中間文件

### 訓練中間文件（已移除）
- 訓練日誌 (training_log_*.txt)
- 進度圖表 (progress.png)
- debug.json

### 一次性腳本（已移除）
- `evaluate_7ch_v2_quick.py`
- `plot_7ch_v2_eval.py`
- `create_v2_confusion_matrix.py`
- 其他臨時評估腳本

---

## ✅ 如何重新生成預處理數據

### 1. 準備原始數據

確保你有WMH資料集：
```
data/wmh/
├── training/
│   ├── Amsterdam/
│   ├── Singapore/
│   └── Utrecht/
└── test/
```

### 2. 選擇模型版本並準備數據

#### 方案A: 使用7CH v2（推薦）

```powershell
cd d:\VSCode\AIOT_E1\nnUnet

# 1. 設置環境
.\setup_env.bat

# 2. 準備7通道v2數據（Butterworth）
python prepare_7channel_v2_butterworth.py

# 3. nnU-Net預處理
nnUNetv2_plan_and_preprocess -d 004 -c 2d

# 4. 推理
nnUNetv2_predict `
  -i nnUNet_raw/Dataset004_WMH_7CH_v2/imagesTs `
  -o predictions `
  -d 004 `
  -c 2d `
  -f 0 `
  -chk 7CH_v2_Archive/checkpoint_best.pth
```

#### 方案B: 使用7CH v1（CLAHE）

```powershell
# 準備7通道v1數據
python prepare_7channel_features.py

# 預處理
nnUNetv2_plan_and_preprocess -d 003 -c 2d

# 推理
nnUNetv2_predict `
  -i nnUNet_raw/Dataset003_WMH_7CH/imagesTs `
  -o predictions `
  -d 003 `
  -c 2d `
  -f 0 `
  -chk 7CH_v1_Archive/checkpoint_best.pth
```

#### 方案C: 使用4CH Baseline

```powershell
# 準備4通道數據
python prepare_4channel_25D.py

# 預處理
nnUNetv2_plan_and_preprocess -d 002 -c 2d

# 推理
nnUNetv2_predict `
  -i nnUNet_raw/Dataset002_WMH_4CH/imagesTs `
  -o predictions `
  -d 002 `
  -c 2d `
  -f 0 `
  -chk 4CH_Model_Archive/checkpoint_bestv0.1.pth
```

---

## 📊 三個模型版本對比

| 模型 | Dice | 訓練時間 | 特色 | 推薦度 |
|------|------|----------|------|--------|
| **7CH v2** | **0.8575** | 3.5h | Butterworth高通，最高Dice | ⭐⭐⭐⭐⭐ |
| 7CH v1 | 0.8549 | 9h | CLAHE增強 | ⭐⭐⭐⭐ |
| 4CH | 0.8331 | 8h | Baseline，簡單 | ⭐⭐⭐ |

---

## 💾 儲存空間

### 清理前: ~7 GB
- nnUNet_preprocessed: 6.3 GB
- temp folders: 0.5 GB
- checkpoints: 0.8 GB

### 清理後: ~0.8 GB
- checkpoints only: 0.8 GB
- Documentation: <10 MB
- Scripts: <5 MB

### 重新生成預處理: +6 GB
預處理數據會重新生成到 `nnUNet_preprocessed/`

---

## 📝 重要說明

### 1. Checkpoint已保留
所有3個模型的最佳checkpoint都已保存在各自Archive資料夾，無需擔心遺失。

### 2. 預處理可重新生成
運行 `nnUNetv2_plan_and_preprocess` 會自動重新創建以下資料夾：
- `nnUNet_preprocessed/`
- `nnUNet_results/Dataset00X/fold_0/`

### 3. 原始數據需自備
你需要自行下載WMH Challenge數據集並放置在 `data/wmh/`

### 4. 詳細使用說明
查看各Archive的README.md獲取詳細使用方法：
- `4CH_Model_Archive/README.md`
- `7CH_v1_Archive/README.md`
- `7CH_v2_Archive/README.md`

---

## 🔄 完整訓練流程（如需重新訓練）

```powershell
# 1. 準備數據
python prepare_7channel_v2_butterworth.py

# 2. 預處理
nnUNetv2_plan_and_preprocess -d 004 -c 2d

# 3. 訓練
nnUNetv2_train 004 2d 0

# 4. 推理
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 004 -c 2d -f 0
```

---

**最後更新**: 2025-12-23  
**清理腳本**: `cleanup_nnunet.ps1`  
**總節省空間**: ~6.3 GB
