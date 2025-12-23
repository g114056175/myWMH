# V0.4 修正版训练配置

**启动时间**: 2025-12-20 18:32  
**版本**: V0.4 Corrected

---

## 🔧 关键修正

### 1. 数据泄漏修正 ✅

```
之前: 按slice随机拆分
  → 60个volumes 100%重叠
  → 严重数据泄漏

现在: 按volume拆分
  → 48 volumes训练
  → 12 volumes验证
  → 0重叠
```

### 2. Tversky参数调整 ✅

```
之前: α=0.25, β=0.75 (β/α=3.0)
  → 过度惩罚FN
  → Precision低 (0.7335)
  → FP过多

现在: α=0.35, β=0.65 (β/α=1.86)
  → 更平衡
  → 增加FP惩罚
  → 预期Precision提升到0.77-0.80
```

### 3. LR Schedule优化 ✅

```
之前: T_0=50
  → Best在epoch 32时LR已很低
  → 无法继续优化

现在: T_0=100
  → LR下降更慢
  → 在50-80 epochs时仍有学习能力
  → 预期Best epoch: 60-80
```

---

## 📊 预期性能

### 训练过程

```
Epoch 1-30: 快速学习
  Val Dice: 0 → 0.75

Epoch 30-60: 稳定提升
  Val Dice: 0.75 → 0.80

Epoch 60-100: 精细优化
  Val Dice: 0.80 → 0.82-0.83

Best Epoch: 预期在70-90
```

### 最终性能

```
Val Dice: 0.80-0.82
  (修正数据泄漏后，比之前的0.8242略低是正常的)

Test Dice: 0.80-0.82
  (这才是真实性能)

Precision: 0.78-0.82
  (Tversky调整的主要收益)

Sensitivity: 0.80-0.82
  (保持合理水平)

Train-Val Gap: <10%
  (更好的泛化)
```

---

## ⚙️ 配置总结

```python
# 模型
BASE_CHANNELS = 64
DEPTH = 5
Dropout = 0.08

# 数据
Train: 48 volumes (~2700 slices)
Val: 12 volumes (~640 slices)
No data leakage ✓

# 损失函数
TVERSKY_ALPHA = 0.35
TVERSKY_BETA = 0.65
FOCAL_GAMMA = 1.33

# 训练
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4
T_0 = 100

# 早停
PATIENCE = 20
MIN_DELTA = 0.002
```

---

## 📈 监控要点

### Epoch 10
- Val Dice应达: 0.65-0.70
- Precision应达: 0.70-0.75

### Epoch 30
- Val Dice应达: 0.75-0.78
- Precision应达: 0.73-0.77

### Epoch 50
- Val Dice应达: 0.78-0.80
- Precision应达: 0.76-0.80

### Epoch 70-90 (预期Best)
- Val Dice应达: 0.80-0.82
- Precision应达: 0.78-0.82

### 如果异常
- Val Dice <0.70 at epoch 30 → LR可能太高
- Precision未提升 → Tversky调整效果不明显
- Train-Val gap >15% → 仍在过拟合

---

## ✅ 成功标准

训练成功，如果满足 **4/5** 条件：

1. ✅ Test Dice ≥ 0.78
2. ✅ Precision ≥ 0.77
3. ✅ Train-Val Gap < 12%
4. ✅ 训练稳定无NaN
5. ✅ Precision比之前(0.7335)提升≥3%

---

## 🎯 下一步

**训练期间** (8-10小时):
- 自动保存checkpoints
- 自动记录training history

**训练完成后**:
1. 评估Test set
2. 对比Precision提升
3. 如需要，测试阈值0.75
4. 实现后处理优化

**预期Timeline**:
- 启动: 2025-12-20 18:30
- 完成: 2025-12-21 03:00-05:00
- 评估: 2025-12-21 上午

---

**配置完成**: ✓  
**数据泄漏修正**: ✓  
**Tversky优化**: ✓  
**LR Schedule改进**: ✓  

**准备启动训练！** 🚀
