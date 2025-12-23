# newUnet2 V0.3 - 训练准备策略

**版本**: V0.3  
**目标**: Dice 0.80+ (从V0.2的0.7656提升)  
**最后更新**: 2025-12-20

---

## 🎯 核心架构变更

### 输入通道：从6通道 → 7通道

```python
# V0.2 (6通道)
Ch0: FLAIR[t-1] - Raw      # 上一切片 (原始)
Ch1: FLAIR[t]   - Raw      # 当前切片 (原始)
Ch2: FLAIR[t+1] - Raw      # 下一切片 (原始)
Ch3: CLAHE               # 对比度增强
Ch4: HighPass            # 高通滤波 (σ=2.0)
Ch5: T1[t]      - Raw      # T1加权

# V0.3 (7通道) ⭐ 优化设计
Ch0: FLAIR[t-1] - CLAHE    # ⭐ 改：2.5D前后层用CLAHE一致增强
Ch1: FLAIR[t]   - CLAHE    # ⭐ 改：核心层CLAHE对比度更好
Ch2: FLAIR[t+1] - CLAHE    # ⭐ 改：保持2.5D特征一致性
Ch3: T1[t]      - Raw      # ✅ 保留：独特的多模态信息
Ch4: HighPass[t]           # ✅ 保留：权重最高17.77%
Ch5: Asymmetry             # ⭐ NEW: 左右不对称 (5.46x对比度)
Ch6: Spatial Atlas         # ⭐ NEW: 空间先验 (6.09x对比度)
```

**核心改动**:
- 用CLAHE替代Raw FLAIR (减少冗余，统一增强)
- 保留T1 (唯一非FLAIR模态，深层特征可能利用)
- 新增2个验证有效的全局特征

---

## 📊 数据流处理流程

### 阶段1: 数据加载
```python
# 1. 加载3D医学影像
flair_3d = nib.load('FLAIR.nii.gz').get_fdata()  # Shape: (H, W, D)
t1_3d = nib.load('T1.nii.gz').get_fdata()        # Shape: (H, W, D)
mask_3d = nib.load('wmh.nii.gz').get_fdata()     # Shape: (H, W, D)

# 2. 预处理 (全volume)
flair_3d = center_crop(normalize_slice(flair_3d), 224)
t1_3d = center_crop(normalize_slice(t1_3d), 224)
mask_3d = center_crop(mask_3d, 224)

# 结果: (224, 224, D)
```

### 阶段2: 提取2D切片（在`__getitem__`中）
```python
for z in range(2, depth - 2):  # 跳过前后2层
    # 提取切片
    flair_prev = flair_3d[:, :, z-1]  # (224, 224)
    flair_curr = flair_3d[:, :, z]    # (224, 224)
    flair_next = flair_3d[:, :, z+1]  # (224, 224)
    t1_curr = t1_3d[:, :, z]          # (224, 224)
    mask_curr = mask_3d[:, :, z]      # (224, 224)
```

### 阶段3: 特征工程
```python
# 3.1 CLAHE应用到3个FLAIR切片 ⭐ V0.3改动
clahe_prev = apply_clahe(flair_prev, clip_limit=2.0)  # t-1
clahe_curr = apply_clahe(flair_curr, clip_limit=2.0)  # t
clahe_next = apply_clahe(flair_next, clip_limit=2.0)  # t+1

# 3.2 HighPass (边缘检测) - 仅对当前切片
highpass = apply_highpass(flair_curr, sigma=2.0)

# 3.3 Asymmetry (左右不对称) ⭐ NEW
asymmetry = np.abs(flair_curr - np.flipud(flair_curr))
asymmetry = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)

# 3.4 Spatial Atlas (全局先验) ⭐ NEW
# 加载一次，所有样本共用
spatial_atlas = self.spatial_atlas  # (224, 224), 预加载

# 3.5 T1归一化
t1_curr_norm = to_uint8(t1_curr).astype(np.float32) / 255.0
```

### 阶段4: 通道堆叠
```python
# Stack成7通道 (H, W, 7)
image = np.stack([
    clahe_prev,         # Ch0: FLAIR[t-1] CLAHE ⭐
    clahe_curr,         # Ch1: FLAIR[t] CLAHE ⭐
    clahe_next,         # Ch2: FLAIR[t+1] CLAHE ⭐
    t1_curr_norm,       # Ch3: T1 Raw
    highpass,           # Ch4: HighPass
    asymmetry,          # Ch5: Asymmetry ⭐ NEW
    spatial_atlas,      # Ch6: Spatial Atlas ⭐ NEW
], axis=-1).astype(np.float32)

mask = mask_curr.astype(np.float32)
```

---

## ⚠️ 数据增强的关键问题

### 问题：固定特征在增强时会失效

#### 受影响的通道
```python
Ch0-2: CLAHE×3      # ✅ 自动跟随（albumentations处理）
Ch5: Asymmetry      # ✅ 自动跟随（albumentations处理）
Ch6: Spatial Atlas  # ✅ 自动跟随（albumentations处理）
```

**原因**:
- `Spatial Atlas` 是全局固定的先验图
- 如果图像旋转/翻转，但atlas不动 → **特征错位**
- `Asymmetry` 也基于空间结构，需要同步变换

---

## ✅ 解决方案：同步变换

### 方案A: Albumentations自动处理 ⭐⭐⭐⭐⭐ 推荐

**Albumentations会自动对所有通道应用相同变换！**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 定义增强
transform = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=25, p=0.9),
    A.ElasticTransform(alpha=80, sigma=8, p=0.6),
    A.GridDistortion(p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.GaussNoise(var_limit=0.001, p=0.2),
    ToTensorV2(),
])

# 应用增强
# image: (H, W, 8) - 所有8个通道
# mask: (H, W)
augmented = transform(image=image, mask=mask)

# ✅ albumentations会对image的所有8个通道应用相同的几何变换
# ✅ Spatial Atlas和Asymmetry会自动跟随旋转/翻转
# ✅ 强度变换（Brightness/Gamma等）只影响需要的通道
```

**关键优势**:
- ✅ **自动同步**: 所有通道的几何变换完全一致
- ✅ **无需手动处理**: albumentations内部处理
- ✅ **已验证**: 当前V0.2使用相同方案

---

### 方案B: 手动同步（不推荐，仅作参考）

如果不用albumentations，需要手动确保同步：

```python
# ❌ 错误做法
augmented_image = rotate(image[:, :, :6])  # 只旋转前6通道
# Spatial Atlas没旋转 → 特征错位！

# ✅ 正确做法
augmented_image = rotate(image)  # 旋转所有8通道
```

---

## 🔍 特殊通道的增强策略

### 1. Asymmetry (Ch6)

**特性**: 基于FLAIR计算，与空间位置相关

**数据增强行为**:
```python
# 几何变换 (旋转/翻转/弹性等)
✅ 跟随变换 - albumentations自动处理

# 强度变换 (亮度/对比度/Gamma等)
⚠️ 可能影响 - 但Asymmetry基于差值，相对稳定
```

**验证**: Asymmetry在变换后仍然标记"不对称"区域

---

### 2. Spatial Atlas (Ch7)

**特性**: 全局固定先验，与空间位置强相关

**数据增强行为**:
```python
# 几何变换 (旋转/翻转/弹性等)
✅ 跟随变换 - albumentations自动处理
   - 旋转25° → atlas也旋转25°
   - 水平翻转 → atlas也水平翻转
   - 弹性变形 → atlas也弹性变形

# 强度变换 (亮度/对比度/Gamma等)
✅ 不受影响 - atlas值范围固定[0, 1]
```

**重要**: Spatial Atlas经过多次旋转/变形后仍保持"脑室旁高值"的相对位置

---

## 📝 实现检查清单

### Dataset实现要点

- [ ] 在`__init__`中加载`spatial_atlas_prior.npy`
- [ ] 在`__getitem__`中计算`asymmetry`特征
- [ ] 确保所有8个通道堆叠到`image`
- [ ] 使用albumentations的`Compose`进行增强
- [ ] 验证增强后所有通道维度一致: `(8, 224, 224)`

### Config更新

```python
IN_CHANNELS = 7  # ⭐ 从6改为7
VERSION = "v0.3"
VERSION_NOTES = "7-channel: CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas"
```

### Model更新

```python
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=7, ...):  # ⭐ 默认7通道
```

---

## ⚠️ 潜在风险与缓解

### 风险1: 过度依赖Spatial Atlas

**问题**: 模型过度依赖先验位置，泛化能力下降

**缓解**:
- ✅ 数据增强旋转/变形 → 打破固定位置依赖
- ✅ Atlas是平滑的概率图，非硬约束
- ✅ 训练时监控atlas的权重，确保不过高

### 风险2: Asymmetry在增强后失效

**问题**: 旋转/变形后，"左右"对称性定义改变

**影响**: 有限
- Asymmetry捕捉的是"不对称"这个概念
- 即使旋转，不对称区域仍会有高值
- **预期**: 仍有帮助，但可能不如Spatial Atlas

**监控**: 训练后分析通道权重

### 风险3: 内存/VRAM增加

**问题**: 从6通道→8通道，内存增加33%

**缓解**:
- 监控训练时VRAM使用
- 如需要，减少batch_size (8→6)
- Spatial Atlas预计算，无额外计算成本

---

## 🚀 训练策略

### 学习率调度器

```python
# V0.2问题: ReduceLROnPlateau导致LR降到0
# V0.3方案: CosineAnnealingWarmRestarts

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,        # 第一次重启周期
    T_mult=1,      # 周期倍增
    eta_min=1e-6   # 最小学习率
)
```

### 数据增强概率（保持V0.2）

```python
A.HorizontalFlip(p=0.8)           # 80% 水平翻转 ✅
A.VerticalFlip(p=0.5)             # 50% 垂直翻转 ✅
A.Rotate(limit=25, p=0.9)         # 90% 旋转±25° ✅
A.ElasticTransform(alpha=80, sigma=8, p=0.6)  # 60% 弹性变换 ✅
A.GridDistortion(p=0.4)           # 40% 网格变换 ✅
A.RandomBrightnessContrast(p=0.5) # 50% 亮度/对比度 ✅
A.RandomGamma(p=0.3)              # 30% Gamma ✅
A.GaussNoise(var_limit=0.001, p=0.2)  # 20% 高斯噪声 ✅
```

**所有几何变换会自动同步到8个通道！**

---

## 📊 预期性能

### 通道权重预测

| 通道 | V0.2权重 | V0.3预测 | 备注 |
|------|---------|---------|------|
| Ch4: HighPass | 17.77% | 15-17% | 仍重要 |
| **Ch7: Spatial Atlas** | - | **14-16%** | ⭐ 预测最高 |
| **Ch6: Asymmetry** | - | **12-14%** | ⭐ 新增 |
| Ch1: FLAIR[t] | 16.85% | 14-16% | 稳定 |
| Ch0/2: FLAIR[t±1] | 16.5% | 13-15% | 2.5D上下文 |
| Ch3: CLAHE | 16.30% | 12-14% | 稍降 |
| Ch5: T1 | 16.15% | 10-12% | 最低 |

### Dice提升预测

```
V0.2 Baseline:        0.7656
+ Asymmetry:          +0.01-0.015  → 0.775-0.780
+ Spatial Atlas:      +0.015-0.02  → 0.790-0.800
+ Better LR:          +0.005-0.01  → 0.795-0.810

保守估计: 0.795
目标:     0.80-0.82
乐观:     0.83+
```

---

## 📋 实施步骤

1. **更新config.py** ✅
   - `IN_CHANNELS = 8`
   
2. **更新dataset.py**
   - 加载spatial_atlas
   - 计算asymmetry
   - Stack 8通道
   
3. **更新model.py**
   - `in_channels=8`
   
4. **更新train.py**
   - CosineAnnealingWarmRestarts
   
5. **测试pipeline**
   - 验证8通道输入
   - 验证数据增强同步
   
6. **开始训练**
   - 150-200 epochs
   - 监控通道权重

---

## 🔧 调试验证脚本

```python
# 验证数据增强同步
def test_augmentation_sync():
    dataset = WMHDataset(train_dir, transform=get_train_transform())
    image, mask = dataset[0]
    
    print(f"Image shape: {image.shape}")  # 应该是 (8, 224, 224)
    print(f"Mask shape: {mask.shape}")    # 应该是 (1, 224, 224)
    
    # 检查Spatial Atlas是否随图像变换
    atlas_channel = image[7]  # Ch7
    print(f"Atlas channel range: [{atlas_channel.min():.3f}, {atlas_channel.max():.3f}]")
    
    # 应该仍在[0, 1]范围，但位置可能旋转/变形
```

---

**Status**: 📋 准备就绪  
**Next**: 实施dataset.py更新  
**Risk Level**: 🟡 中等（需验证增强同步）
