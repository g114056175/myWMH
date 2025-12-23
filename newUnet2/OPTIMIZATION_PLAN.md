# newUnet2 (V0.3) 优化实施计划

## 🎯 总目标

**从 Dice 0.7656 → 0.80+**

通过特征工程和架构优化，提升模型性能至WMH Challenge第1-2名水平。

---

## 📋 优化路线图

### ✅ 已完成 (V0.2 Baseline)

- [x] 6通道输入 (FLAIR×3 + CLAHE + HighPass + T1)
- [x] Attention U-Net架构
- [x] Focal Tversky Loss
- [x] 完整数据增强
- [x] 测试集评估 (Dice 0.7656)

---

### 🚀 待实施优化

#### **优化1: 多尺度HighPass特征** ⭐⭐⭐⭐⭐

**目标**: 增强不同尺度病灶的检测能力

**实施**:
```python
# dataset.py 修改
Ch4: HighPass(sigma=1.5)  # 细边缘 - 小病灶
Ch5: HighPass(sigma=3.0)  # 粗边缘 - 大病灶

def apply_multiscale_highpass(image):
    hp_fine = apply_highpass(image, sigma=1.5)
    hp_coarse = apply_highpass(image, sigma=3.0)
    return hp_fine, hp_coarse
```

**预期**: +3-4% Dice  
**难度**: ⭐ (简单)  
**时间**: 1-2小时

---

#### **优化2: Frangi血管增强** ⭐⭐⭐⭐

**目标**: 增强管状/血管周围WMH特征

**实施**:
```python
# dataset.py 添加
from skimage.filters import frangi

Ch6: Frangi(FLAIR[t], sigmas=[1, 2, 3])

def apply_frangi(image):
    # 增强血管/管状结构
    vessel_enhanced = frangi(
        image, 
        sigmas=range(1, 4, 1),
        black_ridges=False
    )
    return normalize_to_01(vessel_enhanced)
```

**预期**: +1-2% Dice  
**难度**: ⭐⭐ (中等)  
**时间**: 2-3小时

---

#### **优化3: Dual Attention机制** ⭐⭐⭐⭐⭐

**目标**: 同时学习空间和通道重要性

**实施**:
```python
# model.py 修改
class DualAttentionGate(nn.Module):
    def __init__(self, F_g, F_l):
        # 空间注意力 (already have)
        self.spatial_att = SpatialAttention(F_g, F_l)
        
        # 通道注意力 (new)
        self.channel_att = ChannelAttention(F_l)
    
    def forward(self, g, x):
        x_spatial = self.spatial_att(g, x)
        x_channel = self.channel_att(x)
        return (x_spatial + x_channel) / 2

# 替换所有AttentionGate → DualAttentionGate
```

**预期**: +2-3% Dice  
**难度**: ⭐⭐⭐ (中等)  
**时间**: 3-4小时

---

#### **优化4: Deep Supervision** ⭐⭐⭐⭐⭐

**目标**: 多层级监督，加速收敛

**实施**:
```python
# model.py 修改
class AttentionUNetDeepSupervision(AttentionUNet):
    def __init__(self, ...):
        super().__init__(...)
        
        # 添加中间层输出
        self.aux_out1 = nn.Conv2d(base*8, 1, 1)
        self.aux_out2 = nn.Conv2d(base*4, 1, 1)
        self.aux_out3 = nn.Conv2d(base*2, 1, 1)
    
    def forward(self, x):
        # ... encoder/decoder
        
        if self.training:
            # 返回多个输出
            return final, aux1, aux2, aux3
        else:
            return final

# train.py 修改loss计算
if training:
    final, aux1, aux2, aux3 = model(x)
    loss = criterion(final, mask) + \
           0.5 * criterion(resize(aux1), mask) + \
           0.3 * criterion(resize(aux2), mask) + \
           0.2 * criterion(resize(aux3), mask)
```

**预期**: +2-3% Dice  
**难度**: ⭐⭐⭐ (中等)  
**时间**: 4-5小时

---

#### **优化5: CosineAnnealingWarmRestarts** ⭐⭐⭐⭐⭐

**目标**: 周期性跳出局部最优，学习率不降到0

**实施**:
```python
# train.py 修改
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,       # 第一个周期50 epochs
    T_mult=1,     # 每次周期相同
    eta_min=1e-6  # 最小学习率
)

# 训练循环
for epoch in range(200):  # 增加到200 epochs
    # ... training
    scheduler.step()  # 每个epoch都step
```

**预期**: +1-2% Dice  
**难度**: ⭐ (简单)  
**时间**: 30分钟

---

#### **优化6: 更强数据增强** ⭐⭐⭐⭐

**目标**: 提升泛化能力

**实施**:
```python
# dataset.py 修改增强概率
HorizontalFlip(p=0.9)        # 0.8 → 0.9
VerticalFlip(p=0.7)          # 0.5 → 0.7
Rotate(limit=30, p=0.95)     # limit=25, p=0.9
ElasticTransform(alpha=120, sigma=8, p=0.8)  # alpha=80, p=0.6
GridDistortion(p=0.5)        # p=0.4
```

**预期**: +1-2% Dice  
**难度**: ⭐ (简单)  
**时间**: 30分钟

---

## 📊 预期性能提升

| 优化项 | 预期Dice提升 | 累计Dice |
|--------|-------------|----------|
| Baseline V0.2 | - | 0.7656 |
| 多尺度HighPass | +3-4% | 0.7956 |
| Frangi血管增强 | +1-2% | 0.8056 |
| Dual Attention | +2-3% | 0.8256 |
| Deep Supervision | +2-3% | 0.8456 |
| CosineWarmRestart | +1-2% | 0.8556 |
| 更强数据增强 | +1-2% | **0.8656** |

**保守估计**: Dice 0.80-0.82  
**乐观估计**: Dice 0.83-0.85

---

## 🗓️ 实施时间表

### 第1天: 特征工程
- [ ] 多尺度HighPass (1-2h)
- [ ] Frangi血管增强 (2-3h)
- [ ] 测试pipeline (1h)
- [ ] **总计**: 4-6小时

### 第2天: 架构优化
- [ ] Dual Attention (3-4h)
- [ ] Deep Supervision (4-5h)
- [ ] **总计**: 7-9小时

### 第3天: 训练配置
- [ ] CosineWarmRestart (30min)
- [ ] 更强数据增强 (30min)
- [ ] 完整测试 (1h)
- [ ] **开始训练** (10-12h)

### 第4天: 评估
- [ ] 等待训练完成
- [ ] 3D评估
- [ ] 性能分析
- [ ] 对比V0.2

**总时间**: 3-4天（包括训练）

---

## 🔧 实施注意事项

### VRAM管理
- 8通道 + Dual Attention ≈ 7-8 GB
- 如果OOM: 降低batch_size到6
- 确保AMP开启

### 训练稳定性
- 监控loss曲线
- 检查梯度是否正常
- Early stopping patience=100

### 验证方式
- 每个优化单独测试
- 对比baseline性能
- 确保改进有效

---

## 📝 开发日志

### 2025-12-19
- [x] 创建newUnet2项目结构
- [x] 复制核心文件
- [x] 撰写优化计划
- [ ] 待开始实施

---

## 🎯 成功标准

**最低目标**: Dice 0.80 (超越V0.2约4%)  
**期望目标**: Dice 0.82 (WMH Challenge第2名水平)  
**理想目标**: Dice 0.83+ (WMH Challenge第1名水平)

---

## 📚 参考文档

- V0.2评估: `../newUnet/final_evaluation_summary.md`
- 排名分析: `ranking_and_lr_analysis.md`
- 特征工程理论: 见优化1-2详细说明
- 架构优化理论: 见优化3-4详细说明
