# 專案清理建議報告

## 🗑️ 可安全刪除的檔案

### 1. 測試/開發用腳本（根目錄）

這些是開發過程中創建的測試腳本，功能已整合或不再需要：

```
d:\VSCode\AIOT_E1\
├── find_3d_cases.py                    # 已完成，案例已找到
├── visualize_sample_results.py         # 測試用，可刪
├── visualize_features_comprehensive.py # 已完成分析
├── visualize_butterworth.py            # 已生成圖片
├── plot_7ch_v1_curves.py              # 已生成圖片
└── training_data_check.py             # 檢查用，可刪
```

**預估節省**: ~50-100 KB

---

### 2. 臨時評估腳本（nnUnet目錄外）

這些評估腳本已有更新版本或已完成：

```
nnUnet/
├── evaluate_sample.py                 # 舊版評估
├── evaluate_3d_complete.py            # 舊版評估
├── evaluate_3d_fast.py                # 已完成4CH評估
├── evaluate_7ch_v1.py                 # 未完成版本
├── evaluate_7ch_v1_full.py            # 未完成版本
├── evaluate_7ch_v1_complete.py        # ✅ 保留（v1最終版本）
├── evaluate_7ch_v2_quick.py           # v2評估用，保留
├── generate_all_cases.py              # 已完成
├── show_results.py                    # 簡單工具，保留
└── analyze_channel_importance.py      # ✅ 保留（重要分析）
```

**建議**:
- 刪除: evaluate_sample.py, evaluate_3d_complete.py, evaluate_7ch_v1.py, evaluate_7ch_v1_full.py
- 保留: evaluate_7ch_v1_complete.py, evaluate_7ch_v2_quick.py, analyze_channel_importance.py

**預估節省**: ~50 KB

---

### 3. 數據準備腳本（已完成任務）

```
nnUnet/
├── prepare_4channel_25D.py            # ✅ 已歸檔到4CH_Model_Archive/
├── prepare_7channel_features.py       # ✅ 已歸檔到7CH_v1_Archive/
└── prepare_7channel_v2_butterworth.py # ✅ 保留（v2用）
```

**建議**: 可刪除根目錄的副本（archive已保存）

---

### 4. 臨時預測結果（大檔案！）

```
nnUnet/
├── predictions_7ch_v1/     # ~0.89 MB, 3118 files
├── predictions_7ch_v2/     # 可能很大（如果有）
├── batch_eval/             # 臨時資料夾（如果存在）
└── batch_eval_7ch/         # 臨時資料夾（如果存在）
```

**建議**: 
- ✅ **可以刪除predictions_7ch_v1/**（評估已完成，結果已保存）
- ⚠️ predictions_7ch_v2/如果存在且未完成評估，暫時保留

**預估節省**: ~1-5 MB

---

### 5. 舊模型檢查點（如果有）

**請檢查這些位置是否有舊的或未使用的模型**:

```powershell
# 檢查命令
Get-ChildItem "nnUnet\nnUNet_results" -Recurse -Filter "*.pth" | 
  Select-Object Directory, Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}
```

可能的清理目標：
- Dataset001, Dataset002 如果不再需要
- 舊的checkpoint_*.pth（非best或latest）

**預估節省**: 可能100-500 MB+

---

## 📋 清理腳本

### 安全清理（不動nnUnet核心）

```powershell
# 刪除測試腳本
Remove-Item "d:\VSCode\AIOT_E1\find_3d_cases.py"
Remove-Item "d:\VSCode\AIOT_E1\visualize_sample_results.py"
Remove-Item "d:\VSCode\AIOT_E1\visualize_features_comprehensive.py"
Remove-Item "d:\VSCode\AIOT_E1\visualize_butterworth.py"
Remove-Item "d:\VSCode\AIOT_E1\plot_7ch_v1_curves.py"

# 刪除舊評估腳本
Remove-Item "d:\VSCode\AIOT_E1\nnUnet\evaluate_sample.py"
Remove-Item "d:\VSCode\AIOT_E1\nnUnet\evaluate_3d_complete.py"
Remove-Item "d:\VSCode\AIOT_E1\nnUnet\evaluate_7ch_v1.py"
Remove-Item "d:\VSCode\AIOT_E1\nnUnet\evaluate_7ch_v1_full.py"
Remove-Item "d:\VSCode\AIOT_E1\nnUnet\generate_all_cases.py"

# 刪除臨時預測結果（v1已評估完成）
Remove-Item "d:\VSCode\AIOT_E1\nnUnet\predictions_7ch_v1" -Recurse -Force

# 刪除臨時資料夾（如果存在）
if (Test-Path "d:\VSCode\AIOT_E1\nnUnet\batch_eval") {
    Remove-Item "d:\VSCode\AIOT_E1\nnUnet\batch_eval" -Recurse -Force
}
```

**總預估節省**: ~5-10 MB（主要是predictions）

---

## ⚠️ 保留建議

**必須保留**:
- `nnUnet/4CH_Model_Archive/` - 4通道模型歸檔
- `nnUnet/7CH_v1_Archive/` - 7通道v1歸檔
- `nnUnet/nnUNet_results/Dataset003*/` - 7CH v1訓練結果
- `nnUnet/nnUNet_results/Dataset004*/` - 7CH v2訓練結果
- `TEMP_nnUnet/` - 分析報告和可視化

**可選保留**:
- `nnUnet/nnUNet_results/Dataset002*/` - 4CH訓練結果（已歸檔，可刪節省~500MB）

---

**建議操作順序**:
1. 先刪除predictions_7ch_v1/（~1MB）
2. 再刪除測試腳本
3. 最後考慮刪除Dataset002的checkpoint（如果空間緊張）
