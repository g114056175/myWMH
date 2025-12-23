# 7通道nnU-Net配置说明

**基于Unet_v0.4验证成功的7通道配置**

---

## 📋 7通道配置详情

### 通道定义

```
Channel 0 (_0000.nii.gz): CLAHE[t-1]  # 前一个slice的CLAHE
Channel 1 (_0001.nii.gz): CLAHE[t]    # 当前slice的CLAHE  
Channel 2 (_0002.nii.gz): CLAHE[t+1]  # 后一个slice的CLAHE
Channel 3 (_0003.nii.gz): T1          # T1-weighted
Channel 4 (_0004.nii.gz): HighPass    # 高通滤波 (sigma=2.0)
Channel 5 (_0005.nii.gz): Asymmetry   # 左右对称性特征
Channel 6 (_0006.nii.gz): SpatialAtlas # 位置先验图
```

### 通道权重 (已验证)

```
基于V0.6模型分析:
HighPass:      19.36%  (最重要)
T1:            14.54%
CLAHE[t]:      14.49%
CLAHE[t-1]:    13.49%
CLAHE[t+1]:    12.97%
SpatialAtlas:  12.86%
Asymmetry:     12.29%  (最低但仍重要)
```

---

## 🔧 特征计算方法

### 1. CLAHE (对比度限制自适应直方图均衡)

```python
import cv2

def apply_clahe(image_slice, clip_limit=2.0):
    """
    Args:
        image_slice: 2D numpy array, normalized to [0, 1]
        clip_limit: 2.0 (verified optimal)
    
    Returns:
        2D array, CLAHE processed
    """
    # Convert to uint8
    image_uint8 = (image_slice * 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image_uint8)
    
    # Back to float [0, 1]
    return clahe_image.astype(np.float32) / 255.0
```

### 2. HighPass Filter

```python
from scipy.ndimage import gaussian_filter

def apply_highpass(image_slice, sigma=2.0):
    """
    Args:
        image_slice: 2D array, normalized
        sigma: 2.0 (verified optimal)
    
    Returns:
        High-pass filtered image
    """
    # Gaussian blur (low-pass)
    blurred = gaussian_filter(image_slice, sigma=sigma)
    
    # High-pass = original - low-pass
    highpass = image_slice - blurred
    
    # Normalize to [0, 1]
    highpass = (highpass - highpass.min()) / (highpass.max() - highpass.min() + 1e-8)
    
    return highpass.astype(np.float32)
```

### 3. Asymmetry Feature

```python
def compute_asymmetry(image_slice):
    """
    Compute left-right asymmetry
    
    注意: 因为此特征，不能使用HorizontalFlip增强!
    """
    # Flip vertically (上下翻转, 保持左右关系)
    flipped = np.flipud(image_slice)
    
    # Absolute difference
    asymmetry = np.abs(image_slice - flipped)
    
    # Normalize
    asymmetry = (asymmetry - asymmetry.min()) / (asymmetry.max() - asymmetry.min() + 1e-8)
    
    return asymmetry.astype(np.float32)
```

### 4. Spatial Atlas

```python
def create_spatial_atlas(shape=(224, 224)):
    """
    Create spatial prior based on distance from center
    
    WMH tends to appear in periventricular and deep white matter (center)
    """
    h, w = shape
    
    # Coordinate grids
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(x, y)
    
    # Distance from center
    center_x, center_y = 0.5, 0.5
    dist = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
    
    # Invert: center=1, edge=0
    spatial_map = 1.0 - (dist / dist.max())
    
    return spatial_map.astype(np.float32)
```

---

## 📐 数据预处理参数

### Size & Normalization

```python
TARGET_SIZE = 224  # Center crop/pad to 224x224

# Normalization per volume
def normalize_slice(volume_3d):
    """
    Normalize entire 3D volume before extracting slices
    """
    p1 = np.percentile(volume_3d, 1)
    p99 = np.percentile(volume_3d, 99)
    volume_clipped = np.clip(volume_3d, p1, p99)
    volume_norm = (volume_clipped - p1) / (p99 - p1 + 1e-8)
    return volume_norm

# Convert to uint8 for CLAHE
def to_uint8(image):
    return np.clip(image * 255, 0, 255).astype(np.uint8)
```

### Center Crop/Pad

```python
def center_crop(image, target_size=224):
    """
    Center crop or pad to target size
    Works on both 2D and 3D
    """
    if image.ndim == 2:
        h, w = image.shape
        # Pad if needed
        if h < target_size or w < target_size:
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            image = np.pad(image, 
                          ((pad_h//2, pad_h - pad_h//2),
                           (pad_w//2, pad_w - pad_w//2)), 
                          mode='constant')
        
        # Crop
        h, w = image.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return image[start_h:start_h+target_size, 
                    start_w:start_w+target_size]
    
    elif image.ndim == 3:
        # 3D version
        h, w, d = image.shape
        # Similar logic for 3D
        # ...
```

---

## 🎯 数据增强注意事项

### 可用增强

```python
✓ VerticalFlip (50%)
✓ Rotate (±25°, 70%)  
✓ Scale (0.9-1.1, 70%)
✓ ElasticTransform (alpha=45, p=30%)
✓ GridDistortion (p=30%)
✓ Brightness/Contrast (±10%, p=70%)
✓ Gamma (85-115, p=50%)
```

### 禁用增强

```python
❌ HorizontalFlip - 会破坏Asymmetry特征!
❌ Rotate >30° - 会破坏Asymmetry的上下关系
❌ GaussNoise - 可能破坏医学图像结构
```

---

## 📊 验证的性能

### Unet_v0.4/v0.6配置 (相同7通道)

```
Architecture: Attention U-Net
BASE_CHANNELS: 64
DEPTH: 5
Dropout: 0.15
Deep Supervision: Yes

Results:
Val Dice: 0.7738
Test Dice: 0.7656
Precision: 0.7262
Sensitivity: 0.8338
```

---

## 🚀 nnU-Net适配目标

使用这些7通道:
1. 保持特征计算方法不变
2. 让nnU-Net自动配置architecture
3. 让nnU-Net自动配置augmentation (但需禁用HFlip)
4. 预期提升到Test Dice 0.78-0.82

---

## 📁 预期nnU-Net数据结构

```
nnUNet_raw/Dataset001_WMH/
  imagesTr/
    WMH_0001_0000.nii.gz  # CLAHE[t-1]
    WMH_0001_0001.nii.gz  # CLAHE[t]
    WMH_0001_0002.nii.gz  # CLAHE[t+1]
    WMH_0001_0003.nii.gz  # T1
    WMH_0001_0004.nii.gz  # HighPass
    WMH_0001_0005.nii.gz  # Asymmetry
    WMH_0001_0006.nii.gz  # SpatialAtlas
    ...
  labelsTr/
    WMH_0001.nii.gz
    ...
  imagesTs/
    (same structure for test)
  dataset.json
```

**注意**: 
- 每个case需要7个文件 (7个通道)
- 对于3D volume, 需要决定: 逐slice处理(2D) 还是 full 3D?
- 建议: 2D (逐slice) 因为我们的特征是2.5D设计

---

**准备好后**: 运行数据转换脚本生成上述结构
