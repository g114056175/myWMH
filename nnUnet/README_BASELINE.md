# nnU-Net Baseline使用指南

**2通道配置 (FLAIR + T1)**

---

## 🚀 快速开始

### 1. 数据准备

```bash
cd d:\VSCode\AIOT_E1\nnUnet
python prepare_data_2channel.py
```

预期输出:
- 60个训练病例
- 110个测试病例
- Patient-level组织（每个病例是完整3D volume）

### 2. 设置环境

```bash
setup_env.bat
```

### 3. 预处理与规划

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

nnU-Net会自动:
- 分析数据统计
- 决定best spacing
- 决定patch size
- 决定augmentation
- 生成预处理数据

### 4. 训练

```bash
# 2D配置，fold 0
nnUNetv2_train 001 2d 0
```

训练时间: 约6-10小时（取决于GPU）

### 5. 预测

```bash
nnUNetv2_predict -i nnUNet_raw/Dataset001_WMH/imagesTs \
                 -o predictions \
                 -d 001 -c 2d -f 0
```

---

## 📊 预期性能

基于WMH Challenge经验:

```
Baseline (FLAIR + T1, 2D nnU-Net):
- Training Dice: 0.85-0.90
- Test Dice: 0.75-0.80

优势:
+ 完全自动化
+ 经过大量验证
+ 稳定可靠

vs 你的V0.6:
V0.6: Test 0.7656 (手工7通道)
预期: Test 0.75-0.80 (自动2通道)

→ 如果达到0.78+，说明nnU-Net很强
```

---

## ⚠️ 关键注意事项

### Patient-Level Split

```
✓ 数据已按patient组织
✓ nnU-Net会自动做5-fold cross-validation
✓ 每个fold确保patient不重叠

不用担心数据泄漏!
```

### 不能用的增强

nnU-Net默认会用:
- Rotation ✓
- Scaling ✓  
- Elastic ✓
- Mirroring ✓ ← 包括HFlip

但对于2通道baseline这都OK
(只有7通道的Asymmetry才怕HFlip)

---

## 📁 目录结构

```
nnUnet/
├── nnUNet_raw/
│   └── Dataset001_WMH/
│       ├── imagesTr/          # 60 cases × 2 channels = 120 files
│       │   ├── WMH_0001_0000.nii.gz  (FLAIR)
│       │   ├── WMH_0001_0001.nii.gz  (T1)
│       │   └── ...
│       ├── labelsTr/          # 60 files
│       ├── imagesTs/          # 110 cases × 2 = 220 files
│       ├── labelsTs/          # 110 files (ground truth)
│       └── dataset.json
├── nnUNet_preprocessed/       # 自动生成
├── nnUNet_results/            # 训练输出
├── predictions/               # 预测结果
├── prepare_data_2channel.py
├── setup_env.bat
└── README_BASELINE.md
```

---

## 🎯 下一步

**训练完成后**:

1. 评估测试集
2. 对比V0.6 (0.7656)
3. 如果≥0.78:
   - nnU-Net成功!
   - 考虑ensemble或4通道
4. 如果<0.76:
   - 尝试4通道(2.5D)
   - 或7通道(全特征)

---

**预期**: 1-2天完成baseline，为后续改进提供参考
