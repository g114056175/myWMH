# Unet_v0.4 - WMH Segmentation Model

**创建日期**: 2025-12-20  
**基于**: V0.3 (newUnet2) 完成训练与评估  
**目标**: 解决V0.3过拟合问题，提升测试集性能

---

## 🎯 V0.4 核心目标

**解决V0.3的主要问题**:
1. ❌ 严重过拟合 (Train 0.98 → Test 0.77)
2. ❌ HorizontalFlip破坏Asymmetry特征
3. ❌ 模型过大 (31.4M参数 vs 60 volumes)
4. ❌ 未达性能目标 (0.77 vs 0.81-0.83)

**V0.4性能目标**:
- Test Dice: **0.78-0.80** (+1-3% vs V0.3)
- Train-Val Gap: **<12%** (vs V0.3: 16.8%)
- 过拟合程度: **显著降低**

---

## 📊 V0.3 问题分析总结

### 性能表现

```
训练集 Dice:  0.9766  ← 过拟合严重
验证集 Dice:  0.8239  ↓ 15.6%
测试集 Dice:  0.7664  ↓ 21.5% (vs Train)

Aggregate指标 (pixel-level):
  Dice:        0.8290
  Sensitivity: 0.8422
  Precision:   0.8163
```

### 根本原因

**1. 参数/数据比例失衡**
```
参数量: 31.4M
数据量: 60 volumes (2,672 slices)
比例: 11,765 参数/slice

理想比例: 100-1,000
实际: 超标 100-117倍 ❌
```

**2. HorizontalFlip Bug**
```python
# Asymmetry特征
asymmetry = np.abs(flair - np.flipud(flair))  # 检测左右不对称

# 数据增强 (错误)
A.HorizontalFlip(p=0.9)  # ❌ 左右翻转破坏Asymmetry

结果: Asymmetry从5.46x对比度优势 → 失效
```

**3. 模型深度过深**
```
Depth = 5:
  最小分辨率: 14×14 (太粗糙)
  空间细节损失严重

Depth = 4:
  最小分辨率: 28×28 (4倍提升)
  更适合WMH小目标
```

---

## 🔧 V0.4 完整改进策略

### 优先级1: 简化模型架构 ⭐⭐⭐⭐⭐

**修改config.py**:
```python
# V0.3配置
BASE_CHANNELS = 64
DEPTH = 5
→ 31.4M参数，过拟合严重

# V0.4配置
BASE_CHANNELS = 48  # ← 降低25%
DEPTH = 4           # ← 减少1层
→ 预期8.5M参数 (-73%)
```

**效果预期**:
- 参数量: 31.4M → 8.5M
- 参数/slice: 11,765 → 3,181 (降低73%)
- 最小分辨率: 14×14 → 28×28 (提升4倍)
- 过拟合风险: 大幅降低

**实施**:
```python
# config.py
BASE_CHANNELS = 48
DEPTH = 4
VERSION = "v0.4"
VERSION_NOTES = "Simplified 48ch/D4 + Fixed HFlip + Dropout"
```

---

### 优先级2: 修正数据增强 ⭐⭐⭐⭐⭐

**问题**: HorizontalFlip与Asymmetry冲突

**解决方案** (已在dataset.py中):
```python
# ❌ 移除
# A.HorizontalFlip(p=0.9)  # 与Asymmetry冲突

# ✅ 保留安全的增强
A.VerticalFlip(p=0.5)              # 上下翻转，安全
A.Rotate(limit=20, p=0.7)          # ±20°旋转
A.ElasticTransform(alpha=30, sigma=6, p=0.2)  # 轻度变形
A.GridDistortion(p=0.3)            # 网格扭曲

# ✅ 保守的强度变换
A.RandomBrightnessContrast(
    brightness_limit=0.1,  # ±10% (非常保守)
    contrast_limit=0.1,
    p=0.7
)
A.RandomGamma(gamma_limit=(90, 110), p=0.5)  # ±10%
```

**可选增强** (如果效果不佳再考虑):
```python
# 更激进的增强 (V0.4b测试用)
A.Rotate(limit=25, p=0.8)  # 提高到±25°, 80%
A.ElasticTransform(alpha=40, p=0.4)  # 提高强度
A.GaussNoise(var_limit=0.002, p=0.2)  # 添加轻微噪声
```

---

### 优先级3: 添加Dropout正则化 ⭐⭐⭐

**修改model.py的ConvBlock**:

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # ← 添加这行
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
```

**位置**: model.py约第75-85行

**效果**: 
- 训练时随机丢弃15%神经元
- 强制模型学习更鲁棒的特征
- 预期降低过拟合5-10%

---

### 优先级4: 优化训练策略 ⭐⭐⭐⭐

**修改train.py或config.py**:

```python
# 训练轮数
EPOCHS = 200  # 从150提高 (小模型需更多epoch)

# 早停参数
EARLY_STOP_PATIENCE = 20  # 从50降低（更激进）
EARLY_STOP_MIN_DELTA = 0.002  # 从0.001提高（更严格）

# 学习率（保持不变，观察效果）
LEARNING_RATE = 1e-4
```

**理由**:
- 小模型收敛慢，需更多epoch
- 更激进早停避免无效训练
- V0.3在133 epoch最佳，150 epoch反而下降

---

## 📈 预期性能对比

| 指标 | V0.3 | V0.4 (预期) | 改善 |
|------|------|-------------|------|
| **Test Dice** | 0.7664 | **0.78-0.80** | +1-3% |
| **Train Dice** | 0.9766 | **0.90-0.92** | 降低过拟合 |
| **Val Dice** | 0.8239 | **0.82-0.84** | 持平或略升 |
| **Train-Val Gap** | 16.8% | **8-12%** | ↓ 减半 |
| **参数量** | 31.4M | **8.5M** | -73% |
| **训练时间/epoch** | 97s | **~40s** | -59% |
| **VRAM使用** | 6-7GB | **4-5GB** | -2GB |

---

## 🧪 可选实验策略

### 实验A: 测试不同模型大小

如果V0.4效果不理想，可以尝试：

```python
V0.4-Small:  base=32, depth=4  (4.0M参数)
V0.4-Medium: base=48, depth=4  (8.5M参数) ← 推荐
V0.4-Large:  base=56, depth=4  (11M参数)

选择标准:
- 如果Train Dice < 0.88: 模型太小 → 尝试Large
- 如果Train-Val Gap > 15%: 模型太大 → 尝试Small
```

### 实验B: Test-Time Augmentation (TTA)

**无需重训练**，评估时应用：

```python
# 测试时使用VerticalFlip增强
predictions = []
for transform in [None, VerticalFlip]:
    pred = model(transform(image))
    predictions.append(inverse_transform(pred))
final = mean(predictions)

预期提升: +1-2% Test Dice
```

---

## 📝 实施步骤

### Step 1: 代码修改 (30分钟)

```bash
# 1. 修改config.py
BASE_CHANNELS = 48
DEPTH = 4
EPOCHS = 200

# 2. 修改model.py
添加Dropout2d(0.15)到ConvBlock

# 3. 检查dataset.py
确认HFlip已移除，增强参数正确
```

### Step 2: 启动训练 (8-10小时)

```bash
cd d:\VSCode\AIOT_E1\Unet_v0.4
python train.py
```

---

## ✅ 成功标准

V0.4成功，如果满足 **4/5** 条件：

1. ✅ Test Dice ≥ 0.78
2. ✅ Train-Val Gap < 12%
3. ✅ Sensitivity ≥ 0.83  
4. ✅ 训练稳定无NaN
5. ✅ 对V0.3的Bottom 5样本有改善

---

## 📚 参考文档

项目内：
- `V0.4_IMPROVEMENT_PLAN.md` - 详细改进计划
- `CRITICAL_BUG_HORIZONTAL_FLIP.md` - HFlip bug说明
- `../newUnet2/V0.3_Final_Evaluation/` - V0.3完整评估

---

**创建**: 2025-12-20  
**状态**: 准备训练  
**预期完成**: 8-10小时
