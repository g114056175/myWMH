# V0.6 Training - Deep Supervision + Dropout 0.15

**启动时间**: 2025-12-21 午后  
**版本**: V0.6 (Deep Supervision)  
**基于**: V0.4/V0.5分析结果

---

## 🎯 V0.6关键创新

### Deep Supervision ⭐⭐⭐⭐⭐

**3层辅助输出**:
```
Decoder Layer 1 (1/8 resolution) → Aux Output 1 (weight 0.3)
Decoder Layer 2 (1/4 resolution) → Aux Output 2 (weight 0.5)  
Decoder Layer 3 (1/2 resolution) → Aux Output 3 (weight 0.7)
Final Output (full resolution)    → Final Output (weight 1.0)

Total Loss = (0.3*Loss1 + 0.5*Loss2 + 0.7*Loss3 + 1.0*LossFinal) / 2.5
```

**原理**:
- 强制每层学习有意义特征
- 多尺度监督
- 梯度直达中间层

### Dropout 0.15 ⭐⭐⭐⭐

**从V0.4的0.08提升到0.15**:
```
参数/样本比: ~11,600
需要强正则化

Dropout 0.15效果:
- 训练时15%神经元随机关闭
- 有效容量: 31M → ~26M
- 防止过度co-adaptation
```

---

## 📊 预期效果

### V0.5 Baseline (失败)

```
Best Val Dice: 0.7583  
Train-Val Gap: 16%
问题: 数据增强无效，仍过拟合
```

### V0.6 预期

```
Best Epoch: 80-100 (Deep Supervision收敛慢)

Train Dice: 0.86-0.88 (从0.91下降，正常)
Val Dice: 0.78-0.80 (+2-4%)
Test Dice: 0.78-0.80 (+2-4%)

Train-Val Gap: 8-10% (从16%大幅改善)
```

---

## 🔧 技术细节

### 模型架构

```python
AttentionUNet(
    in_channels=7,
    base_channels=64,
    depth=5,
    dropout=0.15  # V0.6
)

新增组件:
- aux_head1: Conv2d(512, 1) 
- aux_head2: Conv2d(256, 1)
- aux_head3: Conv2d(128, 1)
- final_conv: Conv2d(64, 1)

总参数: ~31M (略增，辅助头很轻)
```

### 训练Loss

```python
# 训练时 (deep_supervision=True)
outputs = model(x, deep_supervision=True)
# 返回: [aux1, aux2, aux3, final]

loss = 0.0
weights = [0.3, 0.5, 0.7, 1.0]
for out, w in zip(outputs, weights):
    loss += w * criterion(out, masks)
loss = loss / sum(weights)  # Normalize

# 验证/测试时 (deep_supervision=False)
output = model(x, deep_supervision=False)
# 返回: final only
```

---

## 📈 训练监控

### Epoch 30

```
期望:
Train Dice: 0.80-0.83
Val Dice: 0.72-0.75
Gap: 8-12%

如果Train Dice <0.78:
→ Dropout 0.15可能过强
```

### Epoch 50

```
期望:
Train Dice: 0.84-0.86
Val Dice: 0.76-0.78
Gap: 8-10%
```

### Epoch 80-100 (Best)

```
期望:
Train Dice: 0.86-0.88
Val Dice: 0.78-0.80
Test Dice: 0.78-0.80
Gap: 8-10%
```

---

## ⚠️ 风险管理

### 主要风险

**1. Dropout 0.15欠拟合 (8-12%概率)**
```
症状: Train Dice <0.83 at epoch 50
应对: 考虑降到0.12重训练
```

**2. 收敛变慢 (高概率)**
```
症状: Best epoch在90-120
应对: 已设置200 epochs，不用担心
```

**3. 细节学习能力下降 (中等风险)**
```
症状: 小WMH Dice显著下降
应对: 评估后考虑调整
```

### 成功标准 (4/5)

1. Test Dice ≥ 0.78
2. Train-Val Gap < 12%
3. Val Dice提升 ≥ 2%
4. 无严重欠拟合 (Train Dice ≥0.85)
5. 稳定收敛 (无NaN，无震荡)

---

## 🔄 与历史版本对比

| 版本 | Key Feature | Test Dice | Gap | 问题 |
|------|-------------|-----------|-----|------|
| V0.4 | Patient split + Tversky | 0.7590 | 18% | 过拟合 |
| V0.5 | +Enhanced Aug | 0.7583 | 16% | 无效 |
| **V0.6** | **+Deep Sup + Drop 0.15** | **0.78-0.80** | **8-10%** | - |

---

## ⏱️ 预计训练时间

```
RTX 2060 SUPER 8GB

单Epoch: ~4分钟 (Deep Supervision略慢)
预计Best: Epoch 90
总时间: ~6小时

完成时间: 2025-12-21 晚上
```

---

## ✅ V0.6配置总结

```python
# 模型
Dropout: 0.15  # ← 强正则化
Deep Supervision: 3 aux outputs  # ← 多尺度监督

# 数据 (保持V0.4/V0.5)
Patient-level split: 48/12
Augmentation: Scaling 0.9-1.1, Rotate ±25°, etc.

# 损失 (保持)
Tversky: α=0.35, β=0.65

# 训练 (保持)
Epochs: 200
LR Schedule: T_0=100
Early Stop: patience=20
```

**核心突破**: Deep Supervision解决过拟合  
**预期结果**: 达到0.78-0.80目标

准备启动V0.6训练！
