# V0.7 Training Configuration

**Version**: V0.7  
**Date**: 2025-12-21  
**Target**: Reduce overfitting from 20% to <14%

---

## 🎯 V0.7 Key Changes

### Change 1: Dropout 0.15 → 0.20 ⭐⭐⭐⭐⭐

```python
DROPOUT = 0.20  # +33% regularization strength

Expected:
- Train Dice: 0.88-0.92 (vs 0.96)
- Val Dice: 0.77-0.79
- Gap: 13-15% (vs 20%)
```

### Change 2: Enhanced Deep Supervision Weights ⭐⭐⭐⭐

```python
V0.6: weights = [0.3, 0.5, 0.7, 1.0]
V0.7: weights = [0.5, 0.7, 0.9, 1.0]

Effect:
- Give auxiliary outputs more pressure
- Stronger multi-scale regularization
- Expected Gap reduction: -2%
```

---

## 📊 Baseline (V0.6)

```
Best Val Dice: 0.7738 (Epoch 70)
Test Dice: 0.7656
Train-Val Gap: 20%
Precision: 0.7262

Problem: Severe overfitting
```

---

## 🎯 V0.7 Targets

```
Test Dice: 0.76-0.78 (maintain)
Gap: <14% (improve)
Precision: 0.73-0.75 (maintain)
Recall: 0.81-0.83 (maintain)
```

---

## ⚠️ Risk Assessment

**Dropout 0.2**:
- Underfitting risk: 15-20%
- Monitor: Train Dice at Epoch 30 should >0.82

**Enhanced DS Weights**:
- Risk: Very low
- Possible slower convergence: +10 epochs

---

## 📈 Monitoring Plan

**Epoch 30**:
```
Check Train Dice:
- >0.90: Still overfitting, maybe try 0.22
- 0.84-0.88: Perfect
- <0.82: Too strong, consider 0.18
```

**Epoch 50**:
```
Expected:
- Train: 0.86-0.88
- Val: 0.76-0.78
- Gap: <13%
```

**Epoch 80-100 (Best)**:
```
Target:
- Train: 0.88-0.90
- Val: 0.77-0.79
- Gap: 12-14%
```

---

## ✅ Complete Configuration

```python
# Model
Dropout: 0.20
BASE_CHANNELS: 64
DEPTH: 5
Deep Supervision: True
DS Weights: [0.5, 0.7, 0.9, 1.0]

# Data
Channels: 7
Patient-level split: 48/12
Augmentation: V0.5 enhanced

# Loss
Tversky: α=0.35, β=0.65
Focal: γ=1.33

# Training
Epochs: 200
Batch: 8
LR: 1e-4
LR Schedule: CosineAnnealingWarmRestarts (T_0=100)
Early Stop: patience=20
```

---

## 🔄 vs Previous Versions

| Version | Dropout | DS Weights | Test Dice | Gap |
|---------|---------|------------|-----------|-----|
| V0.4 | 0 | No DS | 0.7590 | 18% |
| V0.5 | 0 | No DS | - | - |
| V0.6 | 0.15 | [0.3,0.5,0.7,1.0] | 0.7656 | 20% |
| **V0.7** | **0.20** | **[0.5,0.7,0.9,1.0]** | **0.76-0.78** | **12-14%** |

---

**Expected Training Time**: 4-5 hours  
**Best Epoch**: 80-100 (estimated)
