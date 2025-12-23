# 旋转增强详解与边界处理

**日期**: 2025-12-20

---

## 🔄 术语澄清

### HFlip vs VFlip vs Asymmetry

**用户的定义 (正确理解)**:
```python
# Asymmetry特征 - 检测左右对称性
asymmetry = np.abs(flair - np.flipud(flair))
# np.flipud = flip up-down = 上下翻转
# 效果: 检测图像的左右对称性

用户认为的"水平反转" = np.flipud = 上下翻转图像以检测左右对称
```

**Albumentations术语**:
```python
A.HorizontalFlip()  # 左右翻转 (沿水平轴翻转)
A.VerticalFlip()    # 上下翻转 (沿垂直轴翻转)

# 这与np.flip的术语不同！
np.flipud()  # flip up-down = 上下翻转
np.fliplr()  # flip left-right = 左右翻转
```

**对应关系**:
```
Albumentations.HorizontalFlip() = np.fliplr() = 左右翻转
Albumentations.VerticalFlip() = np.flipud() = 上下翻转

Asymmetry用的是np.flipud() = 上下翻转 = A.VerticalFlip()
```

---

## ✅ 当前V0.4实现状态

### 已移除的增强

```python
# ❌ 已完全移除 HorizontalFlip
# A.HorizontalFlip(p=0.9)  # 这个会左右翻转，与Asymmetry冲突

# ✅ 保留 VerticalFlip (上下翻转)
A.VerticalFlip(p=0.5)  # 这个不影响Asymmetry
```

**为什么VerticalFlip是安全的？**
```python
# 原始图像
asymmetry = np.abs(flair - np.flipud(flair))
# 检测: 上半部分 vs 下半部分的左右对称性

# 应用VerticalFlip后
flair_vflip = np.flipud(flair)  # 整个图像上下翻转
asymmetry_vflip = np.abs(flair_vflip - np.flipud(flair_vflip))
# 检测: 仍然是左右对称性，只是头脚颠倒

# 但Albumentations会对所有7通道同时应用VFlip
# 所以Asymmetry通道也会同步翻转
# 特征与图像保持一致 ✅
```

---

## 🔄 旋转增强的完整机制

### 当前实现验证

```python
# dataset.py中的__getitem__
def __getitem__(self, idx):
    # 1. 先计算所有7通道特征 (在原始方向)
    clahe_t_minus_1 = apply_clahe(flair[t-1])
    clahe_t = apply_clahe(flair[t])
    clahe_t_plus_1 = apply_clahe(flair[t+1])
    t1_norm = normalize(t1[t])
    highpass_t = apply_highpass(flair[t])
    asymmetry = np.abs(flair[t] - np.flipud(flair[t]))  # 原始方向计算
    
    # 2. Stack成7通道
    image = np.stack([
        clahe_t_minus_1, clahe_t, clahe_t_plus_1,
        t1_norm, highpass_t, asymmetry,
        self.spatial_atlas
    ], axis=0)  # (7, 224, 224)
    
    # 3. 转换为Albumentations格式
    image = np.transpose(image, (1, 2, 0))  # (224, 224, 7)
    
    # 4. 应用数据增强 - 所有通道同时变换
    if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']  # 所有7通道一起旋转/翻转
        mask = transformed['mask']    # mask同步变换
    
    # ✅ 结果: 所有特征图保持相对关系
```

---

## 📐 旋转的边界处理

### Albumentations.Rotate的默认行为

```python
A.Rotate(
    limit=20,  # 旋转角度范围 ±20°
    p=0.7,
    interpolation=cv2.INTER_LINEAR,  # 双线性插值
    border_mode=cv2.BORDER_REFLECT_101,  # ← 边界处理模式
    value=None,
    mask_value=None
)
```

### 边界处理模式详解

**当前使用: BORDER_REFLECT_101**

```
原始图像边界: [a b c d | e f g h | i j k l]
                       ↑ 图像边界

旋转后需要填充的区域使用反射:
填充: [d c b a | a b c d ... | l k j i]
                ↑ 反射填充

优点:
- 平滑过渡，无突变
- 适合医学影像
- 保持边界区域的组织特征
```

**其他可选模式**:

1. **BORDER_CONSTANT** (常数填充)
```python
border_mode=cv2.BORDER_CONSTANT
value=0  # 填充值

效果: 旋转后空白区域填充0 (黑色)
问题: 产生人工边界，模型可能学到伪影
```

2. **BORDER_REPLICATE** (复制边界)
```python
border_mode=cv2.BORDER_REPLICATE

效果: [a b c d | d d d d ...]
问题: 产生不自然的重复
```

3. **BORDER_WRAP** (环绕)
```python
border_mode=cv2.BORDER_WRAP

效果: [i j k l | a b c d | e f g h]
问题: 对医学影像无意义
```

**推荐**: 保持**BORDER_REFLECT_101** (当前默认)

---

## 🎯 改进建议

### 选项1: 保持当前实现 (推荐) ⭐⭐⭐⭐⭐

```python
# dataset.py - get_train_transform()
A.VerticalFlip(p=0.5)  # 安全
A.Rotate(
    limit=20,  # 或增加到25-30
    p=0.7,
    border_mode=cv2.BORDER_REFLECT_101  # 默认，最佳
)
```

**优点**:
- Albumentations自动处理边界
- REFLECT_101适合医学影像
- 不需要额外裁剪/padding

---

### 选项2: 更严格的旋转 (可选)

如果担心边界反射引入伪影：

```python
A.Rotate(
    limit=15,  # 减小角度
    p=0.8,     # 提高概率
    border_mode=cv2.BORDER_REFLECT_101
)
```

**理由**:
- 小角度旋转，边界填充区域更小
- 更高概率确保多样性

---

### 选项3: Safe Rotate (最保守)

旋转后裁剪到安全区域：

```python
# 自定义SafeRotate
class SafeRotate:
    def __init__(self, limit=20, p=0.7):
        self.limit = limit
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask
        
        angle = random.uniform(-self.limit, self.limit)
        
        # 旋转
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image_rotated = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        mask_rotated = cv2.warpAffine(
            mask, M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # 计算安全裁剪区域 (避免边界伪影)
        safe_margin = int(h * abs(np.sin(np.radians(angle))) / 2)
        
        if safe_margin > 0:
            image_rotated = image_rotated[
                safe_margin:-safe_margin,
                safe_margin:-safe_margin
            ]
            mask_rotated = mask_rotated[
                safe_margin:-safe_margin,
                safe_margin:-safe_margin
            ]
            
            # Resize回原尺寸
            image_rotated = cv2.resize(image_rotated, (w, h))
            mask_rotated = cv2.resize(mask_rotated, (w, h))
        
        return image_rotated, mask_rotated
```

**缺点**: 复杂度高，可能不必要

---

## 📊 推荐配置

### V0.4最终数据增强

```python
def get_train_transform(
    aug_p_vertical=0.5,
    aug_p_rotate=0.7,
    aug_p_brightness=0.7,
    aug_p_gamma=0.5,
    aug_p_elastic=0.2,
    aug_p_grid=0.3
):
    """
    V0.4 Final - Corrected Augmentation
    
    移除: HorizontalFlip (与Asymmetry冲突)
    保留: VerticalFlip (安全)
    增强: Rotate角度可适度增加
    """
    return A.Compose([
        # 几何增强
        A.VerticalFlip(p=aug_p_vertical),  # 0.5
        A.Rotate(
            limit=25,  # ±25° (从20增加)
            p=aug_p_rotate,  # 0.7
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,  # 反射填充
        ),
        A.ElasticTransform(
            alpha=30, 
            sigma=6, 
            p=aug_p_elastic,  # 0.2
            border_mode=cv2.BORDER_REFLECT_101
        ),
        A.GridDistortion(
            p=aug_p_grid,  # 0.3
            border_mode=cv2.BORDER_REFLECT_101
        ),
        
        # 强度增强
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # ±10%
            contrast_limit=0.1,
            p=aug_p_brightness  # 0.7
        ),
        A.RandomGamma(
            gamma_limit=(90, 110),  # ±10%
            p=aug_p_gamma  # 0.5
        ),
        
        ToTensorV2(),
    ])
```

---

## 🧪 验证脚本

确认所有特征图正确旋转：

```python
# verify_rotation.py
import numpy as np
import matplotlib.pyplot as plt
from dataset import WMHDataset
import config

dataset = WMHDataset(config.TRAIN_DIR, transform=None)
image, mask = dataset[0]  # (7, 224, 224)

# 手动旋转测试
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

transform = A.Compose([
    A.Rotate(limit=(20, 20), p=1.0, border_mode=cv2.BORDER_REFLECT_101),
    ToTensorV2()
])

image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 7)
transformed = transform(image=image_np, mask=mask.cpu().numpy())

image_rotated = transformed['image'].numpy()  # (7, H, W)
mask_rotated = transformed['mask']

# 可视化7通道
fig, axes = plt.subplots(2, 7, figsize=(28, 8))

channel_names = [
    'CLAHE[t-1]', 'CLAHE[t]', 'CLAHE[t+1]',
    'T1', 'HighPass', 'Asymmetry', 'Spatial Atlas'
]

for i in range(7):
    # 原始
    axes[0, i].imshow(image_np[:,:,i], cmap='gray')
    axes[0, i].set_title(f'{channel_names[i]}\n(Original)')
    axes[0, i].axis('off')
    
    # 旋转后
    axes[1, i].imshow(image_rotated[i], cmap='gray')
    axes[1, i].set_title(f'{channel_names[i]}\n(Rotated 20°)')
    axes[1, i].axis('off')

plt.suptitle('Rotation Verification - All 7 Channels', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('d:/VSCode/AIOT_E1/TEMP2/rotation_verification_7ch.png', dpi=150)
print("✅ Saved to TEMP2/rotation_verification_7ch.png")
```

---

## ✅ 总结

**当前实现状态**:
1. ✅ HorizontalFlip已移除
2. ✅ 所有7通道一起旋转（正确）
3. ✅ 边界使用BORDER_REFLECT_101填充（最佳实践）
4. ✅ VerticalFlip是安全的

**无需额外裁剪/padding**:
- Albumentations的REFLECT_101已经很好处理边界
- 医学影像标准做法
- 不会引入明显伪影

**建议**:
- 保持当前实现
- 可考虑将Rotate从±20°增加到±25°
- 概率从0.7提高到0.8

**完全符合用户要求**:
- ❌ 不用HFlip (已移除)
- ✅ 所有特征图一起旋转
- ✅ 边界自动处理（REFLECT）

准备开始训练！
