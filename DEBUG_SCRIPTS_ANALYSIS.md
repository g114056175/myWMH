# 測試/Debug腳本清理分析

## 🔍 掃描結果

以下腳本按類別分類，標註是否應該刪除。

### ❌ 應刪除 - 測試/驗證腳本

**根目錄**:
- `visualize_butterworth.py` - Butterworth測試可視化
- `visualize_features.py` - 特徵可視化測試
- `find_3d_cases.py` - 尋找3D案例（已完成）
- `test_butterworth.py` - Butterworth測試
- `add_metrics_to_demo.py` - Demo metrics添加（已完成）
- `generate_demo_output.py` - Demo生成（已完成）

**Unet_v0.4**:
- `analyze_*.py` (6個) - 分析腳本
- `check_*.py` - 檢查腳本
- `compare_*.py` - 對比腳本  
- `demo_*.py` - Demo腳本
- `evaluate_*.py` (多個) - 評估腳本
- `get_*.py` - 獲取數據腳本
- `recheck_*.py` - 重新檢查
- `test_*.py` - 測試腳本
- `verify_*.py` - 驗證腳本
- `visualize_*.py` - 可視化腳本

**newUnet**:
- `analyze_*.py`
- `check_*.py`
- `evaluate_*.py`
- `generate_*.py`
- `quick_*.py` - 快速測試
- `test_*.py`
- `validate_*.py`
- `visualize_*.py`

**nnUnet**:
- `analyze_channel_importance.py` - 通道分析（已完成）
- `evaluate_3d_fast.py` - 3D快速評估（已完成）
- `evaluate_7ch_v1_complete.py` - v1完整評估（已完成）
- `evaluate_7ch_v2_complete.py` - v2完整評估（已完成）
- `evaluate_7ch_v2_full.py` - v2完整評估（已完成）
- `show_results.py` - 顯示結果

**TEMP_nnUnet**:
- `visualize_feature_samples.py` - 特徵樣本可視化

### ✅ 應保留 - 核心功能腳本

**根目錄**:
- `dataset.py` - 數據集載入器
- `metrics.py` - 評估指標
- `model.py` - 模型定義
- `train.py` - 訓練腳本

**Unet_v0.4**:
- `config.py`
- `dataset.py`
- `losses.py`
- `model.py`
- `train.py`

**newUnet**:
- `config.py`
- `dataset.py`
- `losses.py`
- `model.py`
- `train.py`
- `augmentation.py`

**nnUnet**:
- `prepare_4channel_25D.py` ⭐
- `prepare_7channel_features.py` ⭐
- `prepare_7channel_v2_butterworth.py` ⭐
- `prepare_data_2channel.py`
- `setup_env.bat`

---

## 📋 刪除清單

### 根目錄
```
visualize_butterworth.py
visualize_features.py
find_3d_cases.py
test_butterworth.py
add_metrics_to_demo.py
generate_demo_output.py
```

### Unet_v0.4/
```
analyze_channel_weights_v04.py
analyze_early_stop.py
analyze_error_types.py
analyze_v04_results.py
analyze_v06_channels.py
check_data_split.py
compare_with_baseline.py
demo_flip_types.py
evaluate_test_3d.py
evaluate_tta.py
get_channel_weights.py
recheck_training.py
test_threshold_optimization.py
verify_split.py
visualize_mixup_demo.py
```

### newUnet/
```
analyze_channel_importance.py
analyze_spatial_map.py
check_ckpt.py
evaluate.py
evaluate_test_3d.py
evaluate_tta.py
generate_complete_report.py
generate_final_report.py
quick_sanity_check.py
quick_test_10samples.py
quick_test_check.py
quick_val.py
quick_validate.py
save_checkpoint.py
test_5samples.py
test_thresholds.py
test_validation.py
update_snapshot.py
validate_components.py
visualize_basic.py
visualize_enhancement_methods.py
visualize_final_methods.py
visualize_poor_samples.py
visualize_predictions.py
```

### nnUnet/
```
analyze_channel_importance.py
evaluate_3d_fast.py
evaluate_7ch_v1_complete.py
evaluate_7ch_v2_complete.py
evaluate_7ch_v2_full.py
show_results.py
```

### TEMP_nnUnet/
```
visualize_feature_samples.py
```

---

## 📊 統計

- **總計待刪除**: ~50個腳本
- **預估節省空間**: ~500 KB
- **主要類型**:
  - 分析腳本 (analyze_*): 10個
  - 評估腳本 (evaluate_*, test_*): 15個
  - 可視化腳本 (visualize_*): 12個
  - 驗證腳本 (check_*, verify_*, validate_*): 8個
  - 其他測試工具: 5個

---

## ⚠️ 注意事項

1. **已保留核心腳本**: train.py, model.py, dataset.py 等
2. **已保留數據準備**: prepare_*.py 腳本
3. **Archive不受影響**: 所有Archive資料夾中的文件保留
4. **可重新生成**: 這些腳本的功能結果已保存在報告中

---

**建議**: 執行自動清理腳本或手動確認後刪除
