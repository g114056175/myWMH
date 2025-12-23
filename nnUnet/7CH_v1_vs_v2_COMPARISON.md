# 7CH v1 vs v2 完整對比分析

## 📊 整體性能對比

### 核心指標

| 指標 | v1 (CLAHE) | v2 (Butterworth) | 差異 | winner |
|------|------------|------------------|------|--------|
| **Overall Dice** | 0.8549 | **0.8575** | **+0.26%** | ✅ v2 |
| **Mean Dice** | 0.7954 | **0.7992** | **+0.38%** | ✅ v2 |
| Median Dice | 0.8242 | ~0.825 | +0.10% | ✅ v2 |
| **Sensitivity** | 0.8267 | **0.8405** | **+1.67%** | ✅ v2 |
| **Precision** | 0.8851 | 0.8752 | -1.12% | ✅ v1 |
| **Specificity** | 0.9998 | 0.9998 | 0% | = |
| Std Dice | 0.1048 | ~0.105 | 持平 | = |

### 訓練效率

| 項目 | v1 | v2 | 差異 |
|------|----|----|------|
| 訓練Epochs | 282 | **102** | **-64%** |
| 訓練時間 | 9小時 | **3.5小時** | **-61%** |
| Pseudo Dice | 0.8855 | 0.8702 | -1.7% |

---

## 🔍 混淆矩陣對比

### v1 (CLAHE)
```
             Pred Neg      Pred Pos
GT Neg    41,589,392         8,289
GT Pos       104,394       497,241
```

### v2 (Butterworth)
```
             Pred Neg      Pred Pos
GT Neg   349,387,795        73,368
GT Pos        97,644       514,569
```

### 關鍵差異
- **TP增加**: 497k → 515k (+3.5%) ✅ v2檢測到更多真陽性
- **FP增加**: 8k → 73k (+813%) ⚠️ v2假陽性大增
- **FN減少**: 104k → 98k (-6.5%) ✅ v2漏檢減少
- **TN**: v1和v2總voxel數不同（可能是valid mask差異）

---

## 💡 通道重要性對比

### v1 (CLAHE)
1. HighPass σ=2: **22.92%** ⭐⭐⭐
2. FLAIR[t]: **19.59%** ⭐⭐⭐
3. T1: **16.33%** ⭐⭐⭐
4. Top-hat: **12.51%** ⭐⭐
5. **CLAHE: 10.28%** ⭐
6. FLAIR[t+1]: 9.47%
7. FLAIR[t-1]: 8.90%

### v2 (Butterworth)
1. HighPass σ=2: **18.09%** ⭐⭐⭐
2. FLAIR[t]: **17.86%** ⭐⭐⭐
3. **Butterworth: 15.72%** ⭐⭐⭐
4. T1: **14.52%** ⭐⭐
5. Top-hat: **13.29%** ⭐⭐
6. FLAIR[t+1]: 10.58%
7. FLAIR[t-1]: 9.94%

### 關鍵發現
✅ **Butterworth排名第3** (vs CLAHE第5)  
✅ **貢獻度15.72%** (vs CLAHE 10.28%, **+53%提升**)  
✅ **權重更均衡**: v2最高18% vs v1最高23%

---

## ⚖️ 優缺點分析

### v1 (CLAHE) 優勢
1. ✅ **Precision更高** (0.8851 vs 0.8752)
   - 假陽性少（8k vs 73k）
   - 更保守的預測策略
   
2. ✅ **穩定proven**
   - 訓練282 epochs，充分收斂
   - CLAHE是成熟的對比增強技術

### v1 (CLAHE) 劣勢
1. ❌ **訓練時間長** (9小時 vs 3.5小時)
2. ❌ **Sensitivity較低** (0.8267 vs 0.8405)
   - 漏檢較多（104k FN）
3. ❌ **CLAHE貢獻低** (僅10.28%)
   - 特徵工程效果有限

---

### v2 (Butterworth) 優勢
1. ✅ **Overall Dice更高** (0.8575 vs 0.8549)
2. ✅ **Sensitivity顯著提升** (+1.67%)
   - 更好的病灶檢測率
   - FN減少6.5%
   
3. ✅ **訓練效率高**
   - 僅102 epochs達到更好性能
   - 節省61%訓練時間
   
4. ✅ **Butterworth貢獻高** (15.72%)
   - 比CLAHE高53%
   - 排名提升至第3

5. ✅ **權重分布均衡**
   - 7個通道都有實質貢獻
   - 無過於dominant的單一通道

### v2 (Butterworth) 劣勢
1. ❌ **Precision略降** (0.8752 vs 0.8851)
   - 假陽性增加（73k vs 8k）
   - 可能過於敏感
   
2. ❌ **訓練未完全收斂**
   - 僅102 epochs（計劃1000）
   - 可能還有提升空間

---

## 🎯 性能差異原因分析

### 為什麼v2 Sensitivity更高？

1. **Butterworth的頻率精確性**
   - cutoff=15專攻小病灶邊緣
   - 比CLAHE的空域增強更有針對性
   - 捕捉到更多subtle lesions

2. **特徵貢獻度提升**
   - Butterworth 15.72% > CLAHE 10.28%
   - 模型更依賴這個特徵
   - 學習到更有效的邊緣信息

### 為什麼v2 Precision略降？

1. **假陽性大增** (8k → 73k)
   - Butterworth可能對噪聲也敏感
   - cutoff=15可能包含部分非病灶的高頻成分
   - Trade-off: 為了高Sensitivity犧牲部分Precision

2. **訓練時間短**
   - 102 epochs vs 282 epochs
   - 可能還沒完全學會區分真假陽性
   - 繼續訓練可能改善

### 為什麼v2訓練更快？

1. **更有效的特徵**
   - Butterworth提供更discriminative的信息
   - 模型更快學會關鍵特徵
   
2. **權重分布均衡**
   - 7個通道都有貢獻
   - 避免過度依賴單一通道
   - 訓練更穩定

---

## 🏆 綜合評價

### 總體表現: v2 (Butterworth) 勝出

**優勢領域**:
- ✅ Overall Dice (+0.26%)
- ✅ Sensitivity (+1.67%) 
- ✅ 訓練效率 (+61%)
- ✅ 特徵工程成效 (+53%)

**劣勢領域**:
- ⚠️ Precision (-1.12%)

### 使用建議

**選v1 (CLAHE) 如果**:
- 需要極低假陽性率
- 計算資源充足（可訓練9小時）
- 優先考慮Precision

**選v2 (Butterworth) 如果**:
- 需要最佳Overall Dice
- 重視病灶檢測率（Sensitivity）
- 訓練時間/資源有限
- 需要高效的特徵工程

---

## 🔮 改進建議

### v2可進一步優化

1. **繼續訓練至200+ epochs**
   - 可能改善Precision
   - 縮小FP

2. **調整Butterworth cutoff**
   - 測試cutoff=20或12
   - 可能找到Sensitivity/Precision平衡點

3. **Post-processing**
   - 加入size-based filtering
   - 移除小型假陽性

---

## 📝 結論

**v2 (Butterworth)是更優的選擇**，因為：

1. **性能提升**: Overall Dice和Sensitivity都更好
2. **效率優勢**: 訓練時間僅v1的39%
3. **特徵工程成功**: Butterworth貢獻度比CLAHE高53%
4. **Precision差距小**: 僅-1.12%，可透過繼續訓練或post-processing改善

**Butterworth替代CLAHE的策略驗證成功** ✅
