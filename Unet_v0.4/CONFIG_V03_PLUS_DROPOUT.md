# V0.4 修正版配置

**修改日期**: 2025-12-20  
**策略**: V0.3架构 + 轻度Dropout + 修正HFlip

---

## 🎯 最终配置

### 模型架构
```python
BASE_CHANNELS = 64  # V0.3 proven capacity
DEPTH = 5           # V0.3 depth
IN_CHANNELS = 7     # 7-channel input
Dropout = 0.08      # Light regularization
```

**参数量**: ~31M (与V0.3相同)

---

### 为什么选择这个配置？

**V0.3的优势**:
- ✅ 证明有效 (Val Dice 0.8239)
- ✅ 充足容量 (Train Dice 0.9766)
- ✅ 可以学到细节

**V0.3的问题**:
- ❌ 过拟合 (Train-Val Gap 16.8%)
- ❌ HFlip bug

**V0.4修正版**:
- ✅ 保持V0.3容量
- ✅ 添加Dropout 0.08 (轻度正则化)
- ✅ 修正HFlip bug
- ✅ 修正数据增强

---

## 📊 Dropout选择: 0.08

### 为什么是0.08？

**Dropout与模型大小的关系**:
```
小模型 (8M):   Dropout 0.15 太强 → 容量不足
中模型 (15M):  Dropout 0.10-0.12 适中
大模型 (30M):  Dropout 0.05-0.10 轻度
```

**0.08的理由**:
1. **足够轻**: 不影响31M模型的学习能力
2. **有效果**: 预期降低过拟合 3-5%
3. **经验值**: 大模型常用0.05-0.10

**预期影响**:
```
V0.3 (无Dropout):
  Train-Val Gap: 16.8%
  
V0.4修正版 (Dropout 0.08):
  Train-Val Gap: 12-14% (改善20-30%)
```

---

## 🎯 预期性能

### 保守估计

```
Val Dice:   0.82-0.84  (vs V0.3: 0.8239)
Test Dice:  0.78-0.80  (vs V0.3: 0.7664)
Train Dice: 0.93-0.95  (vs V0.3: 0.9766)

Train-Val Gap: 12-14% (vs V0.3: 16.8%)
```

### 最佳情况

```
Val Dice:   0.84-0.85
Test Dice:  0.80-0.82
Train-Val Gap: 10-12%
```

**成功标准**: Test Dice ≥ 0.78

---

## ✅ 改进汇总

与V0.3相比的修正:

1. **✅ 添加Dropout 0.08**: 轻度正则化
2. **✅ 修正HFlip**: 移除与Asymmetry冲突的HFlip
3. **✅ 保守数据增强**: Brightness/Gamma ±10%
4. **✅ 早停参数**: Patience=20 (更激进)
5. **✅ 延长训练**: 200 epochs

---

## 🚀 训练监控

### Epoch 10
- Val Dice应达: 0.65-0.70
- VRAM使用: 6-7GB (与V0.3相同)

### Epoch 50
- Val Dice应达: 0.78-0.82
- Train-Val Gap应<15%

### Epoch 100-150
- 寻找最佳点
- 预期Best Epoch: 120-160

### 最终
- Val Dice目标: 0.82+
- Test Dice目标: 0.78+

---

**配置完成**: 2025-12-20  
**准备训练**: ✓  
**预期时间**: 8-10小时
