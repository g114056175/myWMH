"""
2D切片資料集載入器
每個2D FLAIR切片作為獨立樣本
"""
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class WMHDataset2D(Dataset):
    """
    2D WMH分割資料集
    
    將3D NIfTI volumes分解為2D切片，每個切片是獨立樣本
    
    Args:
        data_dir (str): 資料目錄路徑
        split (str): 'training' or 'test'
        empty_ratio (float): 保留空白切片的比例 (0.0-1.0)
        transform (callable, optional): 資料增強函數
        preprocess (bool): 是否應用CLAHE等預處理
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'training',
        empty_ratio: float = 0.1,
        transform=None,
        preprocess: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.empty_ratio = empty_ratio
        self.transform = transform
        self.preprocess = preprocess
        
        # 收集所有切片
        self.slices = []
        self._load_dataset()
        
        print(f"Loaded {len(self.slices)} 2D slices from {split} set")
        if empty_ratio < 1.0:
            non_empty = sum(1 for s in self.slices if s['has_lesion'])
            print(f"  - Non-empty slices: {non_empty}")
            print(f"  - Empty slices: {len(self.slices) - non_empty}")
    
    def _load_dataset(self):
        """載入資料集並提取2D切片資訊"""
        split_dir = self.data_dir / self.split
        
        # 遍歷所有cases
        for region_dir in split_dir.iterdir():
            if not region_dir.is_dir():
                continue
            for scanner_dir in region_dir.iterdir():
                if not scanner_dir.is_dir():
                    continue
                for case_dir in scanner_dir.iterdir():
                    if not case_dir.is_dir():
                        continue
                    
                    flair_path = case_dir / 'pre' / 'FLAIR.nii.gz'
                    wmh_path = case_dir / 'wmh.nii.gz'
                    
                    if flair_path.exists() and (self.split == 'test' or wmh_path.exists()):
                        self._extract_slices(flair_path, wmh_path if self.split == 'training' else None)
    
    def _extract_slices(self, flair_path: Path, wmh_path: Optional[Path]):
        """從單個volume提取所有2D切片"""
        # 載入NIfTI
        flair_nii = nib.load(str(flair_path))
        flair_data = flair_nii.get_fdata()
        
        if wmh_path:
            wmh_nii = nib.load(str(wmh_path))
            wmh_data = wmh_nii.get_fdata()
        else:
            wmh_data = None
        
        H, W, D = flair_data.shape
        
        # 逐切片處理
        for slice_idx in range(D):
            flair_slice = flair_data[:, :, slice_idx]
            
            # 檢查是否為有效切片（非全黑）
            if flair_slice.max() < 1e-6:
                continue
            
            # 檢查是否有病變
            has_lesion = False
            if wmh_data is not None:
                wmh_slice = wmh_data[:, :, slice_idx]
                has_lesion = wmh_slice.max() > 0.5
                
                # 過濾空白切片
                if not has_lesion and np.random.rand() > self.empty_ratio:
                    continue
            
            # 記錄切片資訊
            self.slices.append({
                'flair_path': flair_path,
                'wmh_path': wmh_path,
                'slice_idx': slice_idx,
                'has_lesion': has_lesion,
                'volume_shape': (H, W, D)
            })
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        """
        獲取單個2D切片
        
        Returns:
            image: (1, H, W) tensor - 標準化FLAIR切片
            mask: (1, H, W) tensor - 二值WMH mask（training only）
            info: dict - 切片元資訊
        """
        slice_info = self.slices[idx]
        
        # 載入FLAIR切片
        flair_nii = nib.load(str(slice_info['flair_path']))
        flair_data = flair_nii.get_fdata()
        flair_slice = flair_data[:, :, slice_info['slice_idx']]
        
        # 載入WMH mask（如果有）
        if slice_info['wmh_path']:
            wmh_nii = nib.load(str(slice_info['wmh_path']))
            wmh_data = wmh_nii.get_fdata()
            wmh_slice = wmh_data[:, :, slice_info['slice_idx']]
            mask = (wmh_slice > 0.5).astype(np.float32)
        else:
            mask = np.zeros_like(flair_slice, dtype=np.float32)
        
        # 預處理
        image = self._preprocess(flair_slice)
        
        # 確保大小一致 - pad到256x256
        target_size = 256
        h, w = image.shape
        
        if h != target_size or w != target_size:
            # 計算padding
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Pad
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            # 如果太大則裁剪中心
            if image.shape[0] > target_size:
                start = (image.shape[0] - target_size) // 2
                image = image[start:start+target_size, :]
                mask = mask[start:start+target_size, :]
            if image.shape[1] > target_size:
                start = (image.shape[1] - target_size) // 2
                image = image[:, start:start+target_size]
                mask = mask[:, start:start+target_size]
        
        # 轉換為tensor (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # 資料增強
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # 元資訊
        info = {
            'case_id': slice_info['flair_path'].parent.parent.name,
            'slice_idx': slice_info['slice_idx'],
            'has_lesion': slice_info['has_lesion']
        }
        
        return image, mask, info
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        預處理單個切片
        
        1. Z-score標準化（使用整個切片的統計量）
        2. Clip極值
        3. 可選：CLAHE對比度增強
        """
        # Z-score標準化
        mean = image.mean()
        std = image.std()
        normalized = (image - mean) / (std + 1e-8)
        
        # Clip極值
        normalized = np.clip(normalized, -5, 5)
        
        # TODO: 可選CLAHE
        if self.preprocess:
            # 將在preprocessing.py中實現
            pass
        
        return normalized.astype(np.float32)


# 簡單的資料增強（可擴展）
class RandomFlip:
    """隨機翻轉"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        # 垂直翻轉
        if np.random.rand() < self.p:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        
        # 水平翻轉
        if np.random.rand() < self.p:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        
        return image, mask


class RandomRotation:
    """隨機旋轉（90度倍數）"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            k = np.random.randint(1, 4)  # 90, 180, 270度
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
        
        return image, mask


class Compose:
    """組合多個transform"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


if __name__ == "__main__":
    # 測試資料載入
    data_dir = "data/wmh"
    
    # 創建訓練集
    train_transform = Compose([
        RandomFlip(p=0.5),
        RandomRotation(p=0.3)
    ])
    
    train_dataset = WMHDataset2D(
        data_dir=data_dir,
        split='training',
        empty_ratio=0.1,
        transform=train_transform
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    
    # 測試單個樣本
    image, mask, info = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Case ID: {info['case_id']}")
    print(f"  Slice index: {info['slice_idx']}")
    print(f"  Has lesion: {info['has_lesion']}")
    print(f"  Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"  Mask unique values: {mask.unique()}")
    
    # 測試DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Windows上設為0
    )
    
    batch_image, batch_mask, batch_info = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"  Batch image shape: {batch_image.shape}")
    print(f"  Batch mask shape: {batch_mask.shape}")
    print(f"  Lesion count in batch: {sum(batch_info['has_lesion'])}")
    
    print("\n✓ Dataset test passed!")
