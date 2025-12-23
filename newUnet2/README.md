# newUnet2 - V0.3 Development

**版本**: V0.3 (开发中)  
**基于**: newUnet V0.2 (Dice 0.7656)  
**目标**: Dice 0.80-0.82 (WMH Challenge 1st-2nd place)

---

## 📁 项目结构

```
newUnet2/  (V0.3 - 开发中)
├── config.py                       # 配置文件
├── model.py                        # Attention U-Net
├── dataset.py                      # 数据加载
├── losses.py                       # 损失函数
├── train.py                        # 训练脚本
├── spatial_atlas_prior.npy         # ⭐ 空间先验图谱
├── OPTIMIZATION_PLAN.md            # 优化计划
├── SPATIAL_ATLAS_USAGE.md          # Spatial Atlas使用说明
├── README.md                       # 本文件
│
├── Feature Testing Scripts/        # 特征测试脚本
│   ├── test_asymmetry_feature.py   # 不对称特征测试
│   ├── diagnose_flip_axis.py       # 翻转轴诊断
│   ├── check_augmentation_flip.py  # 数据增强验证
│   ├── generate_spatial_atlas.py   # 生成spatial atlas
│   ├── visualize_spatial_atlas.py  # 可视化spatial atlas
│   ├── quick_validate_atlas.py     # 快速验证atlas
│   └── check_dimensions.py         # 维度检查
│
├── checkpoints/                    # 模型检查点
└── logs/                          # 训练日志
```

---

## 🎯 V0.3 优化策略

### 已验证的新特征

| 特征 | 对比度 | 覆盖率 | 状态 |
|------|--------|--------|------|
| **Spatial Atlas Prior** | **6.09x** | **87.3%** | ✅ 已生成 |
| **Asymmetry** | **5.46x** | - | ✅ 已验证 |
| HighPass (V0.2) | 2.5-3.0x | - | ✅ 已使用 |

### 计划中的改进

1. **多尺度特征工程** ⭐⭐⭐⭐⭐
   - HighPass σ=1.5 (细边缘)
   - HighPass σ=3.0 (粗边缘)
   - Frangi血管增强
   - Asymmetry不对称特征
   - Spatial Atlas Prior

2. **架构改进** ⭐⭐⭐⭐⭐
   - Dual Attention (Spatial + Channel)
   - Deep Supervision

3. **训练优化** ⭐⭐⭐⭐
   - CosineAnnealingWarmRestarts
   - 更强数据增强

---

## 🚀 快速开始

### 1. 特征验证（已完成）

```bash
# 验证Asymmetry特征
python test_asymmetry_feature.py

# 验证Spatial Atlas
python quick_validate_atlas.py

# 可视化Spatial Atlas
python visualize_spatial_atlas.py
```

### 2. 准备训练

**待实现**:
- [ ] 更新`dataset.py`添加新通道
- [ ] 更新`config.py`设置IN_CHANNELS=9或更多
- [ ] 更新`model.py`实现Dual Attention
- [ ] 更新`train.py`实现Deep Supervision

---

## 📊 预期性能提升

| 改进 | 预期Dice提升 | 优先级 |
|------|-------------|--------|
| 多尺度HighPass | +3-5% | ⭐⭐⭐⭐⭐ |
| Spatial Atlas | +1-2% | ⭐⭐⭐⭐⭐ |
| Asymmetry | +1-2% | ⭐⭐⭐⭐ |
| Dual Attention | +2-3% | ⭐⭐⭐⭐⭐ |
| Deep Supervision | +2-3% | ⭐⭐⭐⭐⭐ |
| Better LR Schedule | +1-2% | ⭐⭐⭐⭐ |

**保守估计**: Dice 0.80-0.82  
**乐观估计**: Dice 0.83-0.85

---

## 📝 开发日志

### 2025-12-20
- ✅ 创建newUnet2项目
- ✅ 从newUnet复制核心文件
- ✅ 生成Spatial Atlas Prior (6.09x对比度, 87.3%覆盖率)
- ✅ 验证Asymmetry特征 (5.46x对比度)
- ✅ 验证数据增强配置正确
- ✅ 移动所有V0.3测试脚本到newUnet2

### 待办事项
- [ ] 实现多尺度HighPass
- [ ] 实现Frangi滤波
- [ ] 实现Asymmetry特征
- [ ] 更新dataset.py集成所有新通道
- [ ] 实现Dual Attention
- [ ] 实现Deep Supervision
- [ ] 开始训练

---

## 📚 参考文档

- `OPTIMIZATION_PLAN.md` - 详细优化计划
- `SPATIAL_ATLAS_USAGE.md` - Spatial Atlas使用说明
- `../newUnet/V0.2_Final_Evaluation/` - V0.2评估报告

---

**项目状态**: 🚧 开发中  
**上一版本**: ../newUnet (V0.2, Dice 0.7656)  
**最后更新**: 2025-12-20
