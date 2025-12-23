"""
強化資料增強模組 - PGS風格
包含各種幾何變換、強度變換和醫學影像專用增強
"""
import numpy as np
import torch
import cv2
from scipy.ndimage import rotate, shift
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class RandomHorizontalFlip:
    """水平翻轉"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image = torch.flip(image, dims=[2])  # Width維度
            mask = torch.flip(mask, dims=[2])
        return image, mask


class RandomVerticalFlip:
    """垂直翻轉"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image = torch.flip(image, dims=[1])  # Height維度
            mask = torch.flip(mask, dims=[1])
        return image, mask


class RandomRotation90:
    """隨機90度旋轉（0°, 90°, 180°, 270°）"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            k = np.random.randint(1, 4)  # 1, 2, 3 對應 90°, 180°, 270°
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
        return image, mask


class RandomRotation:
    """隨機角度旋轉（小角度）"""
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            angle = np.random.uniform(-self.degrees, self.degrees)
            
            # 轉為numpy進行旋轉
            image_np = image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            
            # 旋轉（保持大小）
            image_rot = rotate(image_np, angle, reshape=False, order=1)
            mask_rot = rotate(mask_np, angle, reshape=False, order=0)
            
            # 轉回tensor
            image = torch.from_numpy(image_rot).unsqueeze(0).float()
            mask = torch.from_numpy(mask_rot).unsqueeze(0).float()
        
        return image, mask


class RandomScale:
    """隨機縮放"""
    def __init__(self, scale_range=(0.9, 1.1), p=0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            scale = np.random.uniform(*self.scale_range)
            
            image_np = image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            
            h, w = image_np.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 縮放
            image_scaled = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_scaled = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 裁剪或填充回原始大小
            if scale > 1:  # 裁剪中心
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image_scaled = image_scaled[start_h:start_h+h, start_w:start_w+w]
                mask_scaled = mask_scaled[start_h:start_h+h, start_w:start_w+w]
            else:  # 填充
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image_scaled = np.pad(image_scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant')
                mask_scaled = np.pad(mask_scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant')
            
            image = torch.from_numpy(image_scaled).unsqueeze(0).float()
            mask = torch.from_numpy(mask_scaled).unsqueeze(0).float()
        
        return image, mask


class RandomShift:
    """隨機平移"""
    def __init__(self, shift_limit=0.1, p=0.5):
        self.shift_limit = shift_limit
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image_np = image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            
            h, w = image_np.shape
            shift_h = int(h * np.random.uniform(-self.shift_limit, self.shift_limit))
            shift_w = int(w * np.random.uniform(-self.shift_limit, self.shift_limit))
            
            image_shifted = shift(image_np, (shift_h, shift_w), order=1, mode='constant')
            mask_shifted = shift(mask_np, (shift_h, shift_w), order=0, mode='constant')
            
            image = torch.from_numpy(image_shifted).unsqueeze(0).float()
            mask = torch.from_numpy(mask_shifted).unsqueeze(0).float()
        
        return image, mask


class ElasticTransform:
    """彈性變形 - 模擬組織變形"""
    def __init__(self, alpha=30, sigma=5, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image_np = image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            
            shape = image_np.shape
            
            # 生成隨機位移場
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            
            # 應用變形
            image_elastic = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
            mask_elastic = map_coordinates(mask_np, indices, order=0, mode='reflect').reshape(shape)
            
            image = torch.from_numpy(image_elastic).unsqueeze(0).float()
            mask = torch.from_numpy(mask_elastic).unsqueeze(0).float()
        
        return image, mask


class RandomGamma:
    """Gamma調整 - 改變對比度"""
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.gamma_range = gamma_range
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            gamma = np.random.uniform(*self.gamma_range)
            
            # Gamma校正
            image_min = image.min()
            image_max = image.max()
            
            # 正規化到0-1
            image_norm = (image - image_min) / (image_max - image_min + 1e-8)
            
            # 應用gamma
            image_gamma = torch.pow(image_norm, gamma)
            
            # 還原範圍
            image = image_gamma * (image_max - image_min) + image_min
        
        return image, mask


class RandomBrightness:
    """隨機亮度調整"""
    def __init__(self, brightness_range=(-0.2, 0.2), p=0.5):
        self.brightness_range = brightness_range
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            brightness = np.random.uniform(*self.brightness_range)
            image = image + brightness * image.std()
        
        return image, mask


class RandomContrast:
    """隨機對比度調整"""
    def __init__(self, contrast_range=(0.8, 1.2), p=0.5):
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            contrast = np.random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast + mean
        
        return image, mask


class GaussianNoise:
    """高斯噪聲"""
    def __init__(self, noise_std=0.05, p=0.3):
        self.noise_std = noise_std
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            noise = torch.randn_like(image) * self.noise_std * image.std()
            image = image + noise
        
        return image, mask


class CLAHE:
    """對比度受限自適應直方圖均衡化 - 醫學影像常用"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), p=0.3):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image_np = image.squeeze().numpy()
            
            # 轉換到0-255範圍
            image_min = image_np.min()
            image_max = image_np.max()
            image_uint8 = ((image_np - image_min) / (image_max - image_min + 1e-8) * 255).astype(np.uint8)
            
            # 應用CLAHE
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            image_clahe = clahe.apply(image_uint8)
            
            # 轉回原始範圍
            image_clahe = image_clahe.astype(np.float32) / 255.0 * (image_max - image_min) + image_min
            
            image = torch.from_numpy(image_clahe).unsqueeze(0).float()
        
        return image, mask


class Compose:
    """組合多個變換"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# 預定義增強策略
def get_training_augmentation(strength='medium'):
    """
    獲取訓練時的增強策略
    
    Args:
        strength: 'light', 'medium', 'strong', 'champion'
    """
    if strength == 'light':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(p=0.3),
        ])
    
    elif strength == 'medium':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(p=0.3),
            RandomRotation(degrees=15, p=0.3),
            RandomGamma(gamma_range=(0.8, 1.2), p=0.3),
            GaussianNoise(noise_std=0.03, p=0.2),
        ])
    
    elif strength == 'strong':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(p=0.5),
            RandomRotation(degrees=20, p=0.4),
            RandomScale(scale_range=(0.85, 1.15), p=0.4),
            RandomShift(shift_limit=0.1, p=0.3),
            ElasticTransform(alpha=30, sigma=5, p=0.2),
            RandomGamma(gamma_range=(0.7, 1.3), p=0.4),
            RandomBrightness(brightness_range=(-0.2, 0.2), p=0.3),
            RandomContrast(contrast_range=(0.8, 1.2), p=0.3),
            GaussianNoise(noise_std=0.05, p=0.3),
            CLAHE(p=0.2),
        ])
    
    elif strength == 'champion':
        # PGS冠軍級別增強
        return Compose([
            # 幾何變換
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(p=0.5),
            RandomRotation(degrees=25, p=0.5),
            RandomScale(scale_range=(0.8, 1.2), p=0.5),
            RandomShift(shift_limit=0.15, p=0.4),
            ElasticTransform(alpha=40, sigma=6, p=0.3),
            
            # 強度變換
            RandomGamma(gamma_range=(0.7, 1.4), p=0.5),
            RandomBrightness(brightness_range=(-0.3, 0.3), p=0.4),
            RandomContrast(contrast_range=(0.7, 1.3), p=0.4),
            CLAHE(clip_limit=3.0, p=0.3),
            
            # 噪聲
            GaussianNoise(noise_std=0.08, p=0.4),
        ])
    
    else:
        raise ValueError(f"Unknown strength: {strength}. Choose from: light, medium, strong, champion")


if __name__ == "__main__":
    # 測試增強
    print("測試增強策略...")
    
    # 創建假數據
    image = torch.randn(1, 256, 256)
    mask = torch.randint(0, 2, (1, 256, 256)).float()
    
    for strength in ['light', 'medium', 'strong', 'champion']:
        aug = get_training_augmentation(strength)
        img_aug, mask_aug = aug(image.clone(), mask.clone())
        
        print(f"{strength:10s}: 變換數量={len(aug.transforms)}, "
              f"輸出shape={img_aug.shape}")
    
    print("\n✓ 所有增強策略測試通過！")
