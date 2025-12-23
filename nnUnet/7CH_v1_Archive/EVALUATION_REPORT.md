# 7通道v1完整評估報告

**評估日期**: 2025-12-22  
**測試案例**: 110 cases  
**模型**: Dataset003_WMH_7CH (checkpoint_best.pth)

---

## 📊 整體性能指標

| Metric | Score |
|--------|-------|
| **Overall Dice** | **0.8549** |
| **Mean Dice (per-case)** | **0.7954** |
| Median Dice | 0.8242 |
| Std Dice | 0.1048 |
| Sensitivity | 0.8267 |
| Precision | 0.8851 |
| Specificity | 0.9998 |

---

## 🔍 與baseline對比

| 模型 | Overall Dice | Mean Dice | 改進 |
|------|-------------|-----------|------|
| **4通道** | 0.8331* | 0.7983 | baseline |
| **7通道v1** | **0.8549** | **0.7954** | **+2.18% (overall)** |

*基於4CH評估

**關鍵發現**: 
- Overall Dice提升明顯 (+2.18%)
- Mean Dice略低 (-0.29%)，說明對某些案例性能下降

---

## 📈 按病灶大小分類

| 類別 | 案例數 | Mean Dice | 表現 |
|------|--------|-----------|------|
| Small (<100 vx) | 0 | N/A | - |
| **Medium (100-1k vx)** | 29 | **0.6750** | ⚠️ 需改進 |
| Large (>1k vx) | 81 | 0.8385 | ✅ 良好 |

**關鍵問題**: 中型病灶(100-1k)表現較差，Dice僅0.675

---

## 🎯 混淆矩陣 (Voxel-level)

```
              Predicted Neg  Predicted Pos
GT Neg         41,589,392         8,289
GT Pos            104,394       497,241
```

**分析**:
- TP: 497,241
- FP: 8,2 89 (非常低，Precision 0.8851)
- FN: 104,394 (較高，Sensitivity 0.8267)
- TN: 41,589,392

**結論**: 模型非常保守，假陽性極低但漏檢較多

---

## 📉 Dice分布統計

- Min: 0.3766
- 25th percentile: ~0.74
- Median: 0.8242
- 75th percentile: ~0.87
- Max: 0.9324

**變異性**: Std 0.1048，說明不同案例間差異較大

---

## 🆚 vs 訓練Pseudo Dice

| Metric | Value |
|--------|-------|
| Training Pseudo Dice | 0.8855 |
| **Test Mean Dice** | **0.7954** |
| **差距** | **-0.0901 (-10.2%)** |

**分析**: 
- Pseudo Dice (slice-level) 過於樂觀
- 實際3D評估顯示性能下降
- 正常的train-test gap

---

## 💡 關鍵發現

### ✅ 優勢
1. **Overall Dice優秀** (0.8549)
2. **Precision高** (0.8851) - 假陽性少
3. **大病灶表現好** (0.8385)

### ⚠️ 劣勢
1. **中型病灶弱** (100-1k: 0.6750)
2. **Sensitivity偏低** (0.8267) - 漏檢多
3. **Mean Dice略低於4CH** (-0.29%)

### 🔍 原因分析
- CLAHE貢獻低(10.28%)可能是性能瓶頸
- 中型病灶檢測需要改進
- 7通道增加複雜度但未充分利用

---

## 🚀 v2改進方向

**基於評估結果的建議**:

1. ✅ **用Butterworth替代CLAHE**
   - CLAHE貢獻最低且與中型病灶表現差相關
   - Butterworth(cutoff=15)專攻中小型病灶

2. ✅ **優化中型病灶檢測**
   - 100-1k voxels是關鍵弱點
   - Butterworth頻率調整可能有幫助

3. ⚠️ **考慮簡化通道**
   - 7通道可能過於複雜
   - 可能5-6通道更optimal

---

## 📁 輸出檔案

- `training_curves.png` - 訓練曲線
- `evaluation_results.png` - 評估結果可視化
- `evaluation_metrics.json` - 詳細數據
- `EVALUATION_REPORT.md` - 本報告

---

**結論**: 7CH v1整體性能優秀，但中型病灶檢測需改進。v2使用Butterworth有望解決此問題。
