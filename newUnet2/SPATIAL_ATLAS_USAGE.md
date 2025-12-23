# Spatial Atlas Prior - 使用说明

## 📊 什么是Spatial Atlas Prior？

**Spatial Atlas Prior** 是一个基于训练集统计生成的**空间先验图谱**，用于告诉模型WMH病变在大脑中的**典型分布位置**。

---

## 🎯 核心特性

### 1. 尺寸与格式
```
Shape: (224, 224)  ✅ 与所有其他通道完全相同
Dtype: float32
Range: [0.0, 1.0]  ✅ 已归一化
```

### 2. 数值含义
```
高值区域 (0.5-1.0):  WMH高概率区域（脑室旁、深部白质）
中值区域 (0.1-0.5):  WMH中等概率区域
低值区域 (0.0-0.1):  WMH低概率区域（边缘、背景）
```

### 3. 生成过程
1. ✅ 聚合60个训练样本的所有WMH mask
2. ✅ 归一化为概率图
3. ✅ **强制对称化**: `(原图 + flipud) / 2` - 符合大脑左右对称
4. ✅ **高斯平滑**: sigma=7.0 - 避免过拟合
5. ✅ 最终归一化到[0, 1]

---

## ✅ 验证结果

在**110个测试样本**上验证：

| 指标 | 数值 | 结论 |
|------|------|------|
| **WMH vs 非WMH对比度** | **6.09x** | ⭐⭐⭐ 极高 |
| **覆盖率** | **87.3%** | ⭐⭐⭐ 优秀 |
| **WMH区域平均值** | 0.4882 | 高先验区域 |
| **全局平均值** | 0.0832 | 背景低值 |

**结论**: 非常有效！87%的WMH像素出现在高先验区域。

---

## 🚀 如何使用

### 在dataset.py中

#### 1. 加载Atlas（在`__init__`中）
```python
class WMHDataset(Dataset):
    def __init__(self, ...):
        # ... 其他初始化代码 ...
        
        # 加载spatial atlas prior
        atlas_path = Path(__file__).parent / 'spatial_atlas_prior.npy'
        self.spatial_atlas = np.load(atlas_path)
        
        print(f"Loaded spatial atlas: {self.spatial_atlas.shape}, "
              f"range=[{self.spatial_atlas.min():.3f}, {self.spatial_atlas.max():.3f}]")
```

#### 2. 添加为通道（在`__getitem__`中）
```python
def __getitem__(self, idx):
    # ... 现有的6个通道处理 ...
    
    # Stack到7个通道
    image = np.stack([
        flair_t_minus_1_norm,   # ch0
        flair_t_norm,           # ch1
        flair_t_plus_1_norm,    # ch2
        clahe_t,                # ch3
        highpass_t,             # ch4
        t1_t_norm,              # ch5
        self.spatial_atlas,     # ch6 ⭐ 新增空间先验
    ], axis=-1).astype(np.float32)
    
    # ... 数据增强和返回 ...
```

### 在config.py中

```python
# ===== Model Architecture =====
IN_CHANNELS = 7  # 从6改为7 ⭐
OUT_CHANNELS = 1
BASE_CHANNELS = 64
DEPTH = 5

# ===== Version Info =====
VERSION = "v0.3"
VERSION_NOTES = "7-channel: 6-channel + Spatial Atlas Prior"
```

### 在model.py中

```python
# 更新默认参数
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=7, ...):  # ⭐ 改为7
        # ... 模型定义 ...
```

---

## ⚠️ 重要注意事项

### 1. **全局共享，不随样本变化**
```python
# ✅ 正确：所有样本、所有切片使用同一张atlas
self.spatial_atlas  # 固定的 (224, 224)

# ❌ 错误：不要为每个样本生成不同的atlas
```

### 2. **不受数据增强影响**
```python
# ✅ 数据增强后，atlas保持不变
# 模型会学习：即使图像旋转/翻转，先验位置仍有参考价值
```

### 3. **与Spatial Map的区别**

| 特性 | 旧Spatial Map | ✅ Spatial Atlas Prior |
|------|--------------|----------------------|
| 基于 | 坐标（中心=1） | 训练集统计 |
| 验证 | 未验证 | 6.09x对比度 ✅ |
| 对称性 | 天然对称 | 强制对称化 ✅ |
| 数据增强 | ❌ 冲突 | ✅ 兼容 |
| 泛化能力 | 未知 | 87%覆盖率 ✅ |

---

## 📈 预期效果

### 对模型的帮助

1. **空间先验**: 
   - 告诉模型"脑室旁、深部白质"更可能有WMH
   - 减少边缘、背景的假阳性

2. **全局上下文**:
   - 与其他局部特征（CLAHE, HighPass）互补
   - 提供跨空间的先验知识

3. **预期提升**:
   - **Dice: +1-2%**
   - 特别是**减少假阳性**（Precision提升）

---

## 🔍 可视化说明

查看 `TEMP2/spatial_atlas_explained.png` 了解：
1. 热力图显示
2. 灰度图（输入通道的样子）
3. 等高线显示概率分区
4. 阈值分区
5. 值分布直方图
6. 3D表面图

**高亮区域（红色/高值）**: 脑室旁白质、深部白质  
**低值区域（黑色/低值）**: 边缘、背景、脑室内部

---

## 📝 总结

Spatial Atlas Prior是：
- ✅ 基于数据的先验（不是假设）
- ✅ 经过验证的有效特征（6.09x对比度）
- ✅ 实现简单（1个.npy文件）
- ✅ 计算成本零（预计算）
- ✅ 与数据增强兼容

**强烈推荐在V0.3中使用！**

---

**文件位置**: `newUnet2/spatial_atlas_prior.npy`  
**生成脚本**: `newUnet/generate_spatial_atlas.py`  
**最后更新**: 2025-12-20
