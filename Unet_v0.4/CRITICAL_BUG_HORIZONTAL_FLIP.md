# V0.3 数据增强 - 紧急修正

**日期**: 2025-12-20 11:36  
**问题**: HorizontalFlip与Asymmetry特征冲突

---

## ❌ 发现的严重问题

### 1. **HorizontalFlip与Asymmetry冲突**

**Asymmetry特征计算**:
```python
asymmetry = np.abs(flair_t - np.flipud(flair_t))  # 上下翻转
```

**数据增强中的HorizontalFlip**:
```python
A.HorizontalFlip(p=0.9)  # 左右翻转！
```

**冲突**:
- Asymmetry: 检测**左右**不对称（通过上下翻转实现）
- HorizontalFlip: **左右**翻转图像
- 结果: **HorizontalFlip会破坏Asymmetry特征！**

**例子**:
```
原始图像: 左侧有WMH，右侧正常
Asymmetry值: 高（因为左右不同）

经过HorizontalFlip后:
翻转图像: 右侧有WMH，左侧正常
Asymmetry值: 仍然高（但现在是反向的）

模型会学到错乱的Asymmetry模式！
```

---

### 2. **亮度调整太强**

**之前**: ±30% brightness
- 会显著改变WMH与正常组织的信号强度比
- 可能破坏CLAHE增强的效果

**修正**: ±15% brightness
- 更保守，保留对比度特征

---

### 3. **Elastic太强**

**之前**: alpha=40
**修正**: alpha=30
- 更轻微的变形
- 避免扭曲WMH边界

---

## ✅ 修正后的配置

```python
A.VerticalFlip(p=0.5)              # 50% - 保留（不影响Asymmetry）
A.Rotate(limit=20, p=0.7)          # ±20°, 70%
A.ElasticTransform(alpha=30, sigma=6, p=0.2)  # alpha=30, 20%
A.GridDistortion(p=0.3)            # 30%
A.RandomBrightnessContrast(
    brightness_limit=0.15,         # ±15% (降低)
    contrast_limit=0.15,           # ±15% (降低)
    p=0.7
)
A.RandomGamma(gamma_limit=(85, 115), p=0.5)  # 收窄范围
```

---

## 🔍 为什么之前没发现？

1. **术语混淆**:
   - `np.flipud` = "flip up-down" = 上下翻转 = 实现**左右**对称检测
   - `HorizontalFlip` = 水平翻转 = **左右**翻转
   - 两者作用在**同一维度**上

2. **Asymmetry特征被破坏**:
   - 训练时HorizontalFlip随机左右翻转
   - Asymmetry特征失去一致性
   - 可能是V0.3性能未提升的原因之一

---

## 📊 对V0.3性能的影响

**推测**:
- HorizontalFlip破坏了Asymmetry特征的有效性
- Asymmetry从5.46x对比度优势 → 变得混乱
- 这可能解释了为什么V0.3只比V0.2提升0.10%

**应该做**:
- 移除HorizontalFlip
- 重新训练V0.3b
- 预期Asymmetry特征会发挥作用

---

## ✅ 立即行动

1. ✅ 修正dataset.py - 移除HorizontalFlip
2. ✅ 降低Brightness/Contrast到±15%
3. ✅ 降低Elastic到alpha=30
4. ⏳ **建议**: 用修正后的配置重新训练V0.3b

---

**修正时间**: 2025-12-20 11:36  
**严重程度**: 高（破坏关键特征）  
**建议**: 重新训练
