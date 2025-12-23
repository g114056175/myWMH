# V0.4 训练启动清单

**日期**: 2025-12-20  
**状态**: 准备就绪

---

## ✅ 已确认配置

### 数据增强策略

**✅ 正确**: axis=0翻转 (VerticalFlip)
```python
# dataset.py 中当前配置
A.VerticalFlip(p=0.5)  # np.flipud, axis=0翻转 ✅
# ❌ 已移除 HorizontalFlip (与Asymmetry冲突)
A.Rotate(limit=20, p=0.7)  # 可选：增加到25°
A.ElasticTransform(alpha=30, sigma=6, p=0.2)
A.GridDistortion(p=0.3)
A.RandomBrightnessContrast(brightness_limit=0.1, p=0.7)  # ±10%
A.RandomGamma(gamma_limit=(90, 110), p=0.5)  # ±10%
```

---

## 🔧 需要修改的配置

### 1. config.py

```python
# 当前 (V0.3)
BASE_CHANNELS = 64
DEPTH = 5
EPOCHS = 150
VERSION = "v0.3"

# ↓ 修改为 ↓

# V0.4 配置
BASE_CHANNELS = 48  # ← 改这里
DEPTH = 4           # ← 改这里
EPOCHS = 200        # ← 改这里
VERSION = "v0.4"    # ← 改这里
VERSION_NOTES = "Simplified 48ch/D4 + Fixed HFlip + Dropout"

# 早停参数
EARLY_STOP_PATIENCE = 20  # 从50降低
EARLY_STOP_MIN_DELTA = 0.002  # 从0.001提高
```

---

### 2. model.py - 添加Dropout

**位置**: ConvBlock类，约70-85行

```python
# 找到这个类
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # ← 在这里添加下面这行
            nn.Dropout2d(0.15),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
```

**或者更新构造函数**:
```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.15):  # 添加dropout参数
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # 使用参数
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
```

---

### 3. dataset.py - 已正确 ✅

**当前配置已经正确**，无需修改：
- ✅ 移除了HorizontalFlip
- ✅ 保留VerticalFlip (axis=0翻转)
- ✅ 保守的Brightness/Gamma (±10%)

**可选优化** (如果想更激进):
```python
# 可将Rotate从±20°增加到±25°
A.Rotate(limit=25, p=0.8)  # 从(20, 0.7)提高
```

---

## 📊 V0.4 vs V0.3 完整对比

| 配置项 | V0.3 | V0.4 | 原因 |
|--------|------|------|------|
| **BASE_CHANNELS** | 64 | **48** | 减少过拟合 |
| **DEPTH** | 5 | **4** | 提升空间分辨率 |
| **参数量** | 31.4M | **~8.5M** | -73% |
| **HorizontalFlip** | ✅ 0.9 | **❌ 移除** | 与Asymmetry冲突 |
| **VerticalFlip** | 0.7 | **✅ 0.5** | axis=0翻转，安全 |
| **Brightness** | ±30% | **±10%** | 保护对比度 |
| **Gamma** | 70-130 | **90-110** | 保守 |
| **Dropout** | ❌ 无 | **✅ 0.15** | 正则化 |
| **EPOCHS** | 150 | **200** | 小模型需更多 |
| **Patience** | 50 | **20** | 更激进早停 |

---

## 🎯 预期效果

### 性能指标

```
V0.3:
  Train Dice:  0.9766  (过拟合)
  Val Dice:    0.8239
  Test Dice:   0.7664
  Train-Val Gap: 16.8%
  
V0.4 (预期):
  Train Dice:  0.90-0.92  (降低)
  Val Dice:    0.82-0.84  (持平或略升)
  Test Dice:   0.78-0.80  (+1-3%)
  Train-Val Gap: <12%  (显著降低)
```

### 训练特性

```
VRAM使用:   6-7GB → 4-5GB
训练速度:   97s/epoch → ~40s/epoch
总训练时间: 6.25h (150 ep) → 3.5-4h (200 ep, but faster)
```

---

## 🚀 启动步骤

### Step 1: 修改文件 (5分钟)

```bash
# 1. 编辑 config.py
# BASE_CHANNELS = 48, DEPTH = 4, EPOCHS = 200

# 2. 编辑 model.py
# 在ConvBlock中添加 nn.Dropout2d(0.15)

# 3. 检查 dataset.py
# 确认VerticalFlip存在，HorizontalFlip已移除
```

---

### Step 2: 验证配置 (2分钟)

```bash
cd d:\VSCode\AIOT_E1\Unet_v0.4

# 检查模型参数量
python -c "from model import AttentionUNet; import torch; m = AttentionUNet(in_channels=7, base_channels=48, depth=4); print(f'Parameters: {sum(p.numel() for p in m.parameters())/1e6:.1f}M')"

# 预期输出: Parameters: 8.5M (vs V0.3: 31.4M)

# 检查数据增强
python -c "from dataset import get_train_transform; t = get_train_transform(); print([type(x).__name__ for x in t.transforms.transforms])"

# 预期输出应包含VerticalFlip，不包含HorizontalFlip
```

---

### Step 3: 清空历史 (1分钟)

```bash
# 清空checkpoints和logs
Remove-Item "checkpoints\*" -Force -ErrorAction SilentlyContinue
Remove-Item "logs\*" -Force -ErrorAction SilentlyContinue
```

---

### Step 4: 启动训练 (8-10小时)

```bash
python train.py
```

---

## 📈 训练监控要点

### 第1 Epoch 检查

```
期望看到:
- VRAM: 4-5GB (vs V0.3: 6-7GB) ✅
- Train Loss: ~0.9-1.0
- Train Dice: ~0.1-0.2
- 无错误，无NaN
```

### 第10 Epoch 检查

```
期望看到:
- Train Loss: 0.4-0.5
- Train Dice: 0.6-0.7
- Val Loss: 0.3-0.4
- Val Dice: 0.6-0.7
- Train-Val Gap: <15%
```

### 第50 Epoch 检查

```
期望看到:
- Train Dice: 0.85-0.88
- Val Dice: 0.78-0.82
- Train-Val Gap: <12%
- LR周期性变化 (Cosine Annealing)
```

### 第100-150 Epoch

```
期望看到:
- Val Dice plateau (可能早停)
- 最佳epoch在120-160之间
- Train Dice不超过0.93 (避免过拟合)
```

---

## ⚠️ 潜在问题与应对

### 问题1: CUDA OOM

```
症状: RuntimeError: CUDA out of memory
解决: 降低BATCH_SIZE到6或4
```

### 问题2: Loss变NaN

```
症状: Loss显示nan
原因: 学习率太高或梯度爆炸
解决: 降低LR到5e-5
```

### 问题3: Val Dice不提升

```
症状: 50 epoch后Val Dice仍<0.75
原因: 可能模型太小或数据问题
解决: 
1. 提高base_channels到56
2. 降低Dropout到0.1
3. 检查spatial_atlas是否加载
```

### 问题4: 收敛太慢

```
症状: 100 epoch后Val Dice<0.80
原因: 小模型收敛慢
解决: 
1. 提高LR到2e-4
2. 延长到250 epoch
```

---

## ✅ 成功标准

V0.4训练成功，如果满足 **4/5** 条件：

1. ✅ Test Dice ≥ 0.78
2. ✅ Train-Val Gap < 12%
3. ✅ Sensitivity ≥ 0.83
4. ✅ 训练稳定无NaN
5. ✅ 对V0.3 Bottom 5样本有改善

---

## 📝 训练日志记录

**训练开始时间**: _________  
**预期完成时间**: _________ (约8-10小时)  
**实际完成时间**: _________

**关键Epoch记录**:
- Epoch 10: Val Dice = _______
- Epoch 50: Val Dice = _______
- Epoch 100: Val Dice = _______
- Best Epoch: _______ , Val Dice = _______

**最终结果**:
- Test Dice: _______
- Train-Val Gap: _______%
- 是否成功: ☐ 是 ☐ 否

---

## 🎉 准备就绪！

所有配置已确认：
- ✅ 数据增强正确 (VerticalFlip on axis=0)
- ✅ 模型简化计划明确 (48ch/D4)
- ✅ Dropout策略清晰 (0.15)
- ✅ 训练参数优化 (200 epochs, patience=20)

**下一步**: 修改config.py和model.py后，立即启动训练！

Good luck! 🚀
