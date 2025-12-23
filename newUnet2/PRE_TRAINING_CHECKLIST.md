# V0.3 训练前检查清单

**执行前必须检查所有项目！**

---

## ✅ 代码修改检查

### Config.py
- [ ] `IN_CHANNELS = 7` ✅
- [ ] `VERSION = "v0.3"` ✅
- [ ] VERSION_NOTES已更新 ✅

### Dataset.py
- [ ] spatial_atlas已加载 ✅
- [ ] CLAHE应用到3个FLAIR切片 ✅
- [ ] Asymmetry计算正确 ✅
- [ ] 7通道stack顺序正确 ✅
- [ ] 数据增强概率已提高 ✅

### Model.py
- [ ] DualAttentionGate已实现 ✅
- [ ] 所有AttentionGate已替换 ✅
- [ ] `in_channels=7`默认值 ✅

### Train.py
- [ ] CosineAnnealingWarmRestarts已实现 ✅
- [ ] ReduceLROnPlateau已移除 ✅
- [ ] scheduler.step()位置正确 ✅

---

## ✅ 功能验证

### 数据Pipeline
```bash
python -c "
from dataset import WMHDataset, get_train_transform
import config

dataset = WMHDataset(config.TRAIN_DIR, transform=get_train_transform())
image, mask = dataset[0]

print(f'Image shape: {image.shape}')  # 应该是 (7, 224, 224)
print(f'Mask shape: {mask.shape}')    # 应该是 (1, 224, 224)
print(f'Image range: [{image.min():.3f}, {image.max():.3f}]')

assert image.shape[0] == 7, 'Wrong channel count!'
print('✅ Dataset OK!')
"
```

- [ ] Image shape = (7, 224, 224) ✅
- [ ] 无错误输出 ✅

### 模型Forward
```bash
python -c "
import torch
from model import AttentionUNet

model = AttentionUNet(in_channels=7)
x = torch.randn(2, 7, 224, 224)
y = model(x)

print(f'Input: {x.shape}')
print(f'Output: {y.shape}')
assert y.shape == (2, 1, 224, 224), 'Wrong output shape!'
print('✅ Model OK!')
"
```

- [ ] Output shape = (2, 1, 224, 224) ✅
- [ ] 无错误输出 ✅

### Scheduler测试
```bash
python -c "
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

model = torch.nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-6)

lrs = []
for epoch in range(100):
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

print(f'Initial LR: {lrs[0]:.6f}')
print(f'Min LR: {min(lrs):.6f}')
print(f'Epoch 50 LR: {lrs[50]:.6f}')  # 应该重启

assert min(lrs) >= 1e-6, 'LR too low!'
print('✅ Scheduler OK!')
"
```

- [ ] LR不会降到0 ✅
- [ ] Epoch 50重启 ✅

---

## ✅ 文件检查

- [ ] `newUnet2/spatial_atlas_prior.npy` 存在 ✅
- [ ] `newUnet2/checkpoints/` 文件夹存在 ✅
- [ ] `newUnet2/logs/` 文件夹存在 ✅
- [ ] 训练数据路径正确 ✅

---

## ✅ 硬件检查

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

- [ ] CUDA可用 ✅
- [ ] VRAM ≥ 8GB ✅

---

## ✅ 训练参数确认

- [ ] NUM_EPOCHS = 150-200 ✅
- [ ] BATCH_SIZE = 8 (或6如果OOM) ✅
- [ ] LEARNING_RATE = 1e-4 ✅
- [ ] EARLY_STOP_PATIENCE = 50-100 ✅

---

## 🚀 准备启动

**所有检查通过后，执行**:
```bash
cd newUnet2
python train.py
```

**监控重点**:
1. 前3个epoch检查VRAM
2. 前10个epoch检查Loss下降
3. 每20 epoch检查LR变化
4. Val Dice是否稳步提升

---

**检查完成**: ⬜  
**批准训练**: ⬜  
**训练开始时间**: __________
