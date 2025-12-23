# 專案結構整理計劃

## 當前問題

1. 根目錄混亂（太多測試腳本、臨時文件）
2. 多個實驗版本資料夾（Unet_v0.4, newUnet, newUnet2）
3. 臨時資料夾（TEMP, TEMP2, TEMP_nnUnet）
4. 清理腳本散落各處

## 建議結構

```
AIOT_E1/
│
├── README.md                    # 主要說明文件 ⭐
├── PROJECT_REPORT.md            # 完整專案報告 ⭐
├── requirements.txt             # Python依賴
│
├── models/                      # nnU-Net模型（核心）⭐
│   ├── README.md
│   ├── 4CH_Model_Archive/
│   ├── 7CH_v1_Archive/
│   ├── 7CH_v2_Archive/         # 最佳模型
│   ├── prepare_4channel_25D.py
│   ├── prepare_7channel_features.py
│   ├── prepare_7channel_v2_butterworth.py
│   └── setup_env.bat
│
├── output/                      # Demo輸出 ⭐
│   └── demo_predictions/
│       ├── README.md
│       ├── demo_1_*.png
│       ├── demo_2_*.png
│       └── ...
│
├── experiments/                 # 實驗代碼（可選）
│   ├── baseline_unet/          # 原Unet_v0.4
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── dataset.py
│   │   └── train.py
│   └── multi_channel/          # 原newUnet
│       ├── README.md
│       ├── model.py
│       └── ...
│
├── docs/                        # 文檔
│   ├── GITHUB_UPLOAD_GUIDE.md
│   ├── DATA_README.md
│   └── USAGE_AFTER_CLEANUP.md
│
└── .gitignore                   # Git忽略文件 ⭐
```

## 整理動作

### 1. 重命名 nnUnet → models
```powershell
Rename-Item "nnUnet" "models"
```

### 2. 創建 experiments 資料夾
```powershell
New-Item -ItemType Directory "experiments"
Move-Item "Unet_v0.4" "experiments/baseline_unet"
Move-Item "newUnet" "experiments/multi_channel_v1"
# 刪除 newUnet2（已過時）
Remove-Item "newUnet2" -Recurse
```

### 3. 創建 docs 資料夾
```powershell
New-Item -ItemType Directory "docs"
Move-Item "GITHUB_UPLOAD_GUIDE.md" "docs/"
Move-Item "DATA_README.md" "docs/"
Move-Item "models/USAGE_AFTER_CLEANUP.md" "docs/"
```

### 4. 刪除臨時資料夾
```powershell
Remove-Item "TEMP" -Recurse
Remove-Item "TEMP2" -Recurse
Remove-Item "TEMP_nnUnet" -Recurse
```

### 5. 刪除根目錄測試腳本
```powershell
Remove-Item "*_demo.py"
Remove-Item "test_*.py"
Remove-Item "visualize_*.py"
Remove-Item "cleanup_*.ps1"  # 清理完成後刪除
```

### 6. 刪除舊的不需要的資料夾
```powershell
Remove-Item "baseline_2d_ensemble" -Recurse
Remove-Item "nnUnet2" -Recurse
Remove-Item "openspec" -Recurse  # 如果不需要
Remove-Item ".agent" -Recurse
```

## 最終預期結構

```
AIOT_E1/
├── .gitignore
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
│
├── models/                      # 1.3 GB
│   ├── 4CH_Model_Archive/
│   ├── 7CH_v1_Archive/
│   ├── 7CH_v2_Archive/
│   └── prepare_*.py
│
├── output/                      # 5 MB
│   └── demo_predictions/
│
├── experiments/                 # 10 MB
│   ├── baseline_unet/
│   └── multi_channel_v1/
│
└── docs/                        # 1 MB
    ├── GITHUB_UPLOAD_GUIDE.md
    └── ...
```

**總大小**: ~1.32 GB（主要是模型checkpoint）

## 優點

1. ✅ 清晰的結構
2. ✅ 核心模型在 models/
3. ✅ Demo在 output/
4. ✅ 實驗代碼隔離在 experiments/
5. ✅ 文檔集中在 docs/
6. ✅ 根目錄簡潔

## 執行

運行 `restructure_project.ps1` 自動整理
