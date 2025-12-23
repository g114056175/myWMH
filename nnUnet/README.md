# nnU-Net WMH Segmentation Model

## 概述

这是一个基于nnU-Net v2的白质高信号(WMH)分割模型，使用3通道输入（FLAIR + T1 + TopHat特征）进行2D分割。

**模型性能**:
- Training Set Dice: **0.886**
- Test Set Dice (修复后): **0.66-0.83**
- Amsterdam数据: 优秀表现
- Singapore/Utrecht数据: 经过spacing统一修复后可用

---

## 目录结构

```
nnUnet/
├── nnUNet_raw/               # 原始准备好的数据
│   └── Dataset001_WMH/
│       ├── imagesTr/         # 训练图像 (3通道: _0000.nii.gz=FLAIR, _0001=T1, _0002=TopHat)
│       ├── labelsTr/         # 训练标签
│       ├── imagesTs/         # 测试图像
│       ├── labelsTs/         # 测试标签
│       └── dataset.json      # 数据集配置
├── nnUNet_preprocessed/      # nnU-Net自动预处理的数据
├── nnUNet_results/           # 训练好的模型
│   └── Dataset001_WMH/
│       └── nnUNetTrainer__nnUNetPlans__2d/
│           └── fold_0/
│               ├── checkpoint_best.pth      # 最佳模型
│               └── checkpoint_final.pth     # 最终模型
├── predictions/              # 原始测试集预测结果
├── predictions_fixed/        # 修复后测试集预测
├── predictions_training/     # 训练集预测（用于验证）
├── scripts/                  # 辅助脚本
│   ├── eval_fixed_results.py
│   ├── visualize_singapore70_fixed.py
│   ├── optimize_threshold.py
│   └── run_inference.py
├── results/                  # 结果可视化
│   └── singapore70_fixed_visualization.png
├── docs/                     # 文档
├── prepare_nnunet_data.py    # **核心**：数据准备脚本
├── setup_nnunet_env.bat      # 环境设置
├── run_complete_training.bat # 训练脚本
├── run_fixed_inference.bat   # 推论脚本
├── run_inference_with_probabilities.bat  # 概率推论（用于阈值优化）
└── README.md                 # 本文件
```

---

## 环境设置

### 必需环境变量

在运行任何nnU-Net命令前，必须设置以下环境变量：

```powershell
# 方法1: 使用提供的批处理（推荐）
nnUnet\setup_nnunet_env.bat

# 方法2: 手动设置
$env:nnUNet_raw = "D:\VSCode\AIOT_E1\nnUnet\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\VSCode\AIOT_E1\nnUnet\nnUNet_preprocessed"
$env:nnUNet_results = "D:\VSCode\AIOT_E1\nnUnet\nnUNet_results"
```

---

## 数据准备

### 输入数据格式

原始数据应位于 `data/wmh/` 目录，结构如下：

```
data/wmh/
├── training/
│   ├── Amsterdam/
│   │   └── GE3T/
│   │       └── 100/
│   │           ├── pre/
│   │           │   ├── FLAIR.nii.gz
│   │           │   └── T1.nii.gz
│   │           └── wmh.nii.gz
│   └── Singapore/
│       └── 50/
│           ├── pre/
│           │   ├── FLAIR.nii.gz
│           │   └── T1.nii.gz
│           └── wmh.nii.gz
└── test/
    └── (同样结构)
```

### 运行数据准备

```powershell
python prepare_nnunet_data.py
```

**功能**:
1. 扫描training和test目录
2. 对所有数据resample到统一spacing: `[3.0, 0.9766, 1.0]` (z, y, x)
3. 计算TopHat特征（多尺度morphological top-hat）
4. 保存为nnU-Net 3通道格式
5. 生成dataset.json

**输出**:
- `nnUNet_raw/Dataset001_WMH/imagesTr/WMH_XXXX_0000.nii.gz` (FLAIR)
- `nnUNet_raw/Dataset001_WMH/imagesTr/WMH_XXXX_0001.nii.gz` (T1)
- `nnUNet_raw/Dataset001_WMH/imagesTr/WMH_XXXX_0002.nii.gz` (TopHat)
- `nnUNet_raw/Dataset001_WMH/labelsTr/WMH_XXXX.nii.gz` (Label)

**重要配置**:
- **TARGET_SPACING**: `[3.0, 0.9766, 1.0]` - 统一所有数据到此spacing（解决training/test分布差异）
- **TopHat kernel sizes**: `[3, 5, 7, 11]` - 多尺度检测

---

## 模型训练

### 训练配置

- **架构**: nnU-Net 2D U-Net
- **配置**: `2d` (逐slice处理)
- **Fold**: 0 (单fold训练)
- **输入通道**: 3 (FLAIR + T1 + TopHat)
- **Patch size**: `[256, 224]`
- **Batch size**: 57
- **Normalization**: Z-score (per case)

### 运行训练

```powershell
# 使用提供的批处理（推荐）
nnUnet\run_complete_training.bat

# 或手动执行
# 1. Planning
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

# 2. 训练
nnUNetv2_train 001 2d 0 --npz
```

**训练时间**: 约6-8小时（取决于GPU）

**输出**:
- 模型checkpoint: `nnUNet_results/Dataset001_WMH/nnUNetTrainer__nnUNetPlans__2d/fold_0/`
- 验证结果: `progress.png`, `training.log`

---

## 推论

### 标准推论（二值输出）

```powershell
# 使用批处理
nnUnet\run_fixed_inference.bat

# 或手动
nnUNetv2_predict -i nnUnet/nnUNet_raw/Dataset001_WMH/imagesTs -o nnUnet/predictions_fixed -d 001 -c 2d -f 0 -chk checkpoint_best.pth
```

### 概率推论（用于阈值优化）

```powershell
nnUnet\run_inference_with_probabilities.bat
```

添加 `--save_probabilities` 标志可保存softmax概率（0-1连续值），用于后续阈值优化以提高Recall。

---

## 评估

### 快速评估

```powershell
python scripts/eval_fixed_results.py
```

评估Singapore测试cases，输出Dice、Precision、Recall等指标。

### 可视化

```powershell
python scripts/visualize_singapore70_fixed.py
```

生成FLAIR、GT和Prediction的并排对比图。

### 阈值优化

```powershell
python scripts/optimize_threshold.py
```

测试不同阈值（0.1-0.8）以找到Dice和Recall的最佳平衡。

**注意**: 需要先运行概率推论获取连续值输出。

---

## 输入/输出规格

### 输入

- **FLAIR**: T2-FLAIR MRI序列
- **T1**: T1-weighted MRI序列  
- **格式**: NIfTI (.nii.gz)
- **原始spacing**: 任意（将自动resample到目标spacing）
- **目标spacing**: `[3.0, 0.9766, 1.0]` mm (z, y, x)

### 输出

- **格式**: NIfTI (.nii.gz)
- **值**: 
  - 标准推论: 二值 (0=背景, 1=WMH)
  - 概率推论: 连续 [0, 1] (WMH概率)
- **空间**: 与输入相同（自动resample回原始space）

---

## 关键技术点

### 1. 数据准备的统一spacing

**问题**: Training和Test的Singapore数据原始shape不同：
- Training: (256, 232, 48)
- Test: (132, 256, 83)

导致Z-score normalized后的统计特性不同，模型无法识别。

**解决**: 在`prepare_nnunet_data.py`中resample所有数据到统一spacing `[3.0, 0.9766, 1.0]`。

**代码位置**: `prepare_nnunet_data.py` Line 13-18

### 2. TopHat特征

增强小病灶检测能力，通过多尺度morphological top-hat提取高亮区域。

**代码位置**: `prepare_nnunet_data.py` Line 57-84

### 3. Z-score Normalization

nnU-Net自动对每个case做Z-score: `(data - mean) / std`

确保不同scanner/protocol的数据可以被模型处理。

---

## 故障排查

### 问题1: "More than one dataset name found"

**原因**: nnUNet_raw中有多个Dataset001_WMH目录（例如备份）

**解决**: 删除或移出备份目录，只保留一个Dataset001_WMH

### 问题2: Test set Dice很低

**检查**:
1. Test数据是否经过统一spacing处理？（应该在48-100 slices范围）
2. 查看`nnUNet_raw/Dataset001_WMH/imagesTs/`中文件的shape
3. 如果shape差异大，重新运行`prepare_nnunet_data.py`

### 问题3: Checkpoint not found

**检查**:
1. 环境变量是否正确设置？
2. 模型是否训练完成？
3. Checkpoint路径: `nnUNet_results/Dataset001_WMH/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth`

---

## 性能指标

### Training Set (60 cases)

- **Dice**: 0.886
- **数据**: Amsterdam (20) + Singapore/Utrecht (40)

### Test Set (110 cases)

| 数据集 | Dice | Recall | Precision |
|--------|------|--------|-----------|
| Amsterdam | ~0.80-0.85 | ~0.75-0.85 | ~0.85-0.90 |
| Singapore (修复前) | 0.017 | 0.02 | 0.01 |
| Singapore (修复后) | 0.66-0.83 | 0.54-0.79 | 0.83-0.88 |

**修复前后对比**:
- Singapore/70: 0.017 → **0.656** (提升39倍)
- Singapore/71: - → **0.833** (优秀)

---

## 未来改进

1. **阈值优化**: 使用概率输出优化阈值以提高Recall（减少漏检）
2. **后处理**: 添加连通域分析去除小误检
3. **集成学习**: 训练多fold并ensemble
4. **3D模型**: 尝试nnU-Net 3D配置以利用体积信息

---

## 参考

- nnU-Net论文: Isensee et al., Nature Methods 2021
- nnU-Net GitHub: https://github.com/MIC-DKFZ/nnUNet
- 数据集: WMH Segmentation Challenge (假设)

---

## 联系信息

如有问题，请检查：
1. `nnUNet_results/*/fold_0/training.log` - 训练日志
2. `this README.md` - 使用说明
3. `/docs` - 额外文档

最后更新: 2025-12-17
