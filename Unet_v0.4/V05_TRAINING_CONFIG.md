# V0.5 Training Configuration

**版本**: V0.5  
**启动时间**: 2025-12-21 00:00  
**基于**: V0.4修正版 (patient-level split + Tversky调整)

---

## 🎯 V0.5核心改进

### 1. Dropout提升 ⭐⭐⭐⭐⭐

```python
V0.4: Dropout 0.08
V0.5: Dropout 0.12 (+50%)

目标: 减少过拟合
预期: Gap 18% → 10-12%
```

### 2. 数据增强加强 ⭐⭐⭐⭐⭐

**新增**:
```python
Scaling: 0.9-1.1倍
→ 模拟不同脑大小
→ 参考nnU-Net等顶级方法
```

**提升**:
```python
Rotate: 20° → 25°
ElasticTransform: alpha 30→45, p 0.2→0.3
Gamma: (90,110) → (85,115)
```

**保持**:
```python
VerticalFlip: 50%
GridDistortion: 30%
Brightness/Contrast: ±10% (保守)
```

---

## 📊 完整配置

### 模型

```python
Architecture: Attention U-Net
BASE_CHANNELS: 64
DEPTH: 5
Dropout: 0.12  # ← V0.5提升

Parameters: ~31M
Input: 7 channels
```

### 数据

```python
Patient-level split:
  Train: 48 volumes (2672 slices)
  Val: 12 volumes (668 slices)
  No data leakage ✓

Test: 110 samples (separate)
```

### 损失函数

```python
Focal Tversky Loss:
  TVERSKY_ALPHA: 0.35  # FP权重
  TVERSKY_BETA: 0.65   # FN权重
  FOCAL_GAMMA: 1.33
  
目标: 降低FP，提高Precision
```

### 训练

```python
EPOCHS: 200
BATCH_SIZE: 8
LEARNING_RATE: 1e-4

LR Scheduler: CosineAnnealingWarmRestarts
  T_0: 100  # 更长周期
  eta_min: 1e-6

Early Stopping:
  PATIENCE: 20
  MIN_DELTA: 0.002
```

---

## 🎯 预期性能

### V0.4 (修正版) Baseline

```
Best Epoch: 63
Val Dice: 0.7644
Test Dice: 0.7590

Train-Val Gap: ~15-18%
主要问题: 过拟合
```

### V0.5 预期

```
Best Epoch: 预计70-90

Val Dice: 0.78-0.80 (+1.5-3.5%)
Test Dice: 0.78-0.80 (+2-4%)

Train-Val Gap: 10-12% (健康)

改进来源:
1. Dropout 0.12: +1.5-2.5%
2. 数据增强: +1.0-1.5%
```

---

## 📈 训练监控指标

### Epoch 30检查点

```
期望:
Val Dice: ≥0.72
Train Dice: 0.82-0.85
Gap: <15%

如果Val Dice <0.70:
→ 可能欠拟合，考虑降低Dropout
```

### Epoch 50检查点

```
期望:
Val Dice: ≥0.76
Train Dice: 0.85-0.88
Gap: 10-12%

如果Train Dice <0.82:
→ 确认欠拟合，降低Dropout到0.10
```

### Epoch 70-90 (预期Best)

```
期望:
Val Dice: 0.78-0.80
Train Dice: 0.88-0.90
Gap: 10-12%
```

---

## ⚠️ 风险评估

### 主要风险

**1. Dropout 0.12可能过强?**
```
概率: 30%
影响: Train Dice <0.85
缓解: 监控epoch 50，必要时重训练0.10
```

**2. 数据增强可能破坏细节?**
```
概率: 20%
影响: 小WMH检测下降
缓解: 保守参数设置，避免过强变换
```

**3. 收敛可能变慢?**
```
概率: 60%
影响: Best epoch延后到80-100
缓解: 已设置T_0=100，足够epoch数
```

### 成功标准

**必须达到 (3/4)**:
1. ✅ Test Dice ≥ 0.78
2. ✅ Train-Val Gap < 13%
3. ✅ Val Dice提升 ≥1.5%
4. ✅ Precision ≥ 0.77

---

## 🔄 对比历史版本

| 版本 | Dropout | 数据Split | Tversky | Test Dice | Gap | 主要问题 |
|------|---------|-----------|---------|-----------|-----|----------|
| V0.3 | 0 | Slice-level | 0.25/0.75 | 0.7664 | 21% | 数据泄漏 |
| V0.4-small | 0.08 | Patient | 0.25/0.75 | 0.7464 | 8% | 容量不足 |
| V0.4-修正 | 0.08 | Patient | 0.35/0.65 | 0.7590 | 18% | 过拟合 |
| **V0.5** | **0.12** | Patient | 0.35/0.65 | **0.78-0.80** | **10-12%** | - |

---

## 📋 V0.5 vs V0.4变化总结

### 修改文件

**model.py**:
```python
Line 138-142: Dropout 0.08 → 0.12
```

**dataset.py**:
```python
Line 275-283: 新增ShiftScaleRotate (含Scaling)
Line 287-291: ElasticTransform加强
Line 295: Gamma范围放宽
```

**其他保持不变**:
- config.py: 保持
- train.py: 保持
- Patient-level split: 保持
- Tversky参数: 保持

---

## ⏱️ 预计训练时间

```
硬件: RTX 2060 SUPER 8GB

单epoch: ~3.5分钟
  (Dropout 0.12略慢于0.08)

预计Best epoch: 80
总时间: 280分钟 ≈ 4.5-5小时

完成时间: 2025-12-21 05:00左右
```

---

## ✅ 训练前检查清单

- [x] Dropout更新到0.12
- [x] 数据增强配置更新
- [x] Patient-level split确认
- [x] Tversky参数确认
- [x] checkpoints目录清空
- [x] 配置文档完成

**准备就绪！可以启动训练**

---

## 🎯 V0.5目标

**核心目标**:
- Test Dice: 0.78-0.80 (vs 当前0.759)
- Train-Val Gap: <12% (vs 当前18%)
- 方法学正确 ✓
- 可复现性 ✓

**如果成功**:
- 达到原定0.78-0.80目标
- 为后续Deep Supervision等改进打下基础

**如果需要**:
- 准备Phase 2: Deep Supervision
- 目标: 0.80-0.82

**启动训练**: `python train.py`
