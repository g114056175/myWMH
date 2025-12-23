"""
nnU-Net数据准备 - 4通道2.5D (FLAIR t-1, t, t+1, T1)
"""

import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def prepare_4channel_data():
    print("="*70)
    print("nnU-Net数据准备 - 4通道2.5D")
    print("="*70)
    
    source_train = Path("d:/VSCode/AIOT_E1/data/wmh/training")
    source_test = Path("d:/VSCode/AIOT_E1/data/wmh/test")
    target_base = Path("d:/VSCode/AIOT_E1/nnUnet/nnUNet_raw/Dataset002_WMH_4CH")
    
    # 创建目录
    target_base.mkdir(parents=True, exist_ok=True)
    (target_base / "imagesTr").mkdir(exist_ok=True)
    (target_base / "labelsTr").mkdir(exist_ok=True)
    (target_base / "imagesTs").mkdir(exist_ok=True)
    (target_base / "labelsTs").mkdir(exist_ok=True)
    
    print(f"\n数据源: {source_train}")
    print(f"目标: {target_base}")
    
    # === 训练数据 ===
    print("\n" + "="*70)
    print("处理训练数据 - 提取2.5D slices")
    print("="*70)
    
    # 找到所有训练病例
    train_cases = []
    for site_dir in source_train.iterdir():
        if not site_dir.is_dir():
            continue
        
        has_scanner_subdir = any(d.is_dir() and not d.name.isdigit() for d in site_dir.iterdir())
        
        if has_scanner_subdir:
            for scanner_dir in site_dir.iterdir():
                if not scanner_dir.is_dir():
                    continue
                for case_dir in scanner_dir.iterdir():
                    if case_dir.is_dir() and case_dir.name.replace('_', '').isdigit():
                        flair_path = case_dir / "pre" / "FLAIR.nii.gz"
                        t1_path = case_dir / "pre" / "T1.nii.gz"
                        mask_path = case_dir / "wmh.nii.gz"
                        
                        if flair_path.exists() and t1_path.exists() and mask_path.exists():
                            train_cases.append({
                                'site': site_dir.name,
                                'case_id': case_dir.name,
                                'flair': flair_path,
                                't1': t1_path,
                                'mask': mask_path
                            })
        else:
            for case_dir in site_dir.iterdir():
                if case_dir.is_dir() and case_dir.name.replace('_', '').isdigit():
                    flair_path = case_dir / "pre" / "FLAIR.nii.gz"
                    t1_path = case_dir / "pre" / "T1.nii.gz"
                    mask_path = case_dir / "wmh.nii.gz"
                    
                    if flair_path.exists() and t1_path.exists() and mask_path.exists():
                        train_cases.append({
                            'site': site_dir.name,
                            'case_id': case_dir.name,
                            'flair': flair_path,
                            't1': t1_path,
                            'mask': mask_path
                        })
    
    train_cases = sorted(train_cases, key=lambda x: (x['site'], x['case_id']))
    print(f"\n找到 {len(train_cases)} 个训练病例")
    
    # 转换训练数据 - 逐slice处理
    print(f"\n开始转换为2.5D slices...")
    slice_idx = 0
    
    for case in tqdm(train_cases, desc="训练数据"):
        # 加载3D volumes
        flair_nii = nib.load(case['flair'])
        t1_nii = nib.load(case['t1'])
        mask_nii = nib.load(case['mask'])
        
        flair_data = flair_nii.get_fdata()
        t1_data = t1_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # 处理每个slice (跳过首尾2个slice)
        for z in range(2, flair_data.shape[2] - 2):
            slice_idx += 1
            case_id = f"WMH_{slice_idx:05d}"
            
            # 提取3个连续FLAIR slices
            flair_prev = flair_data[:, :, z-1]
            flair_curr = flair_data[:, :, z]
            flair_next = flair_data[:, :, z+1]
            t1_slice = t1_data[:, :, z]
            mask_slice = mask_data[:, :, z]
            
            # 保存4个通道
            for ch_idx, data in enumerate([flair_prev, flair_curr, flair_next, t1_slice]):
                img_2d = nib.Nifti1Image(data.astype(np.float32), 
                                         flair_nii.affine)
                img_path = target_base / "imagesTr" / f"{case_id}_{ch_idx:04d}.nii.gz"
                nib.save(img_2d, img_path)
            
            # 保存mask (处理标签2)
            mask_slice[mask_slice == 2] = 0
            mask_2d = nib.Nifti1Image(mask_slice.astype(np.uint8),
                                     flair_nii.affine)
            mask_path = target_base / "labelsTr" / f"{case_id}.nii.gz"
            nib.save(mask_2d, mask_path)
    
    num_train_slices = slice_idx
    print(f"\n✓ 训练数据: {len(train_cases)} volumes → {num_train_slices} slices")
    
    # === 测试数据 ===
    print("\n" + "="*70)
    print("处理测试数据 - 提取2.5D slices")
    print("="*70)
    
    # 找到所有测试病例 (类似逻辑)
    test_cases = []
    for site_dir in source_test.iterdir():
        if not site_dir.is_dir():
            continue
        
        has_scanner_subdir = any(d.is_dir() and not d.name.isdigit() for d in site_dir.iterdir())
        
        if has_scanner_subdir:
            for scanner_dir in site_dir.iterdir():
                if not scanner_dir.is_dir():
                    continue
                for case_dir in scanner_dir.iterdir():
                    if case_dir.is_dir() and case_dir.name.replace('_', '').isdigit():
                        flair_path = case_dir / "pre" / "FLAIR.nii.gz"
                        t1_path = case_dir / "pre" / "T1.nii.gz"
                        mask_path = case_dir / "wmh.nii.gz"
                        
                        if flair_path.exists() and t1_path.exists():
                            test_cases.append({
                                'site': site_dir.name,
                                'case_id': case_dir.name,
                                'flair': flair_path,
                                't1': t1_path,
                                'mask': mask_path if mask_path.exists() else None
                            })
        else:
            for case_dir in site_dir.iterdir():
                if case_dir.is_dir() and case_dir.name.replace('_', '').isdigit():
                    flair_path = case_dir / "pre" / "FLAIR.nii.gz"
                    t1_path = case_dir / "pre" / "T1.nii.gz"
                    mask_path = case_dir / "wmh.nii.gz"
                    
                    if flair_path.exists() and t1_path.exists():
                        test_cases.append({
                            'site': site_dir.name,
                            'case_id': case_dir.name,
                            'flair': flair_path,
                            't1': t1_path,
                            'mask': mask_path if mask_path.exists() else None
                        })
    
    test_cases = sorted(test_cases, key=lambda x: (x['site'], x['case_id']))
    print(f"\n找到 {len(test_cases)} 个测试病例")
    
    # 转换测试数据
    slice_idx = 0
    for case in tqdm(test_cases, desc="测试数据"):
        flair_nii = nib.load(case['flair'])
        t1_nii = nib.load(case['t1'])
        flair_data = flair_nii.get_fdata()
        t1_data = t1_nii.get_fdata()
        
        if case['mask']:
            mask_nii = nib.load(case['mask'])
            mask_data = mask_nii.get_fdata()
        
        for z in range(2, flair_data.shape[2] - 2):
            slice_idx += 1
            case_id = f"WMH_test_{slice_idx:05d}"
            
            flair_prev = flair_data[:, :, z-1]
            flair_curr = flair_data[:, :, z]
            flair_next = flair_data[:, :, z+1]
            t1_slice = t1_data[:, :, z]
            
            for ch_idx, data in enumerate([flair_prev, flair_curr, flair_next, t1_slice]):
                img_2d = nib.Nifti1Image(data.astype(np.float32),
                                        flair_nii.affine)
                img_path = target_base / "imagesTs" / f"{case_id}_{ch_idx:04d}.nii.gz"
                nib.save(img_2d, img_path)
            
            if case['mask']:
                mask_slice = mask_data[:, :, z]
                mask_slice[mask_slice == 2] = 0
                mask_2d = nib.Nifti1Image(mask_slice.astype(np.uint8),
                                         flair_nii.affine)
                mask_path = target_base / "labelsTs" / f"{case_id}.nii.gz"
                nib.save(mask_2d, mask_path)
    
    num_test_slices = slice_idx
    print(f"\n✓ 测试数据: {len(test_cases)} volumes → {num_test_slices} slices")
    
    # === dataset.json ===
    dataset_json = {
        "channel_names": {
            "0": "FLAIR_t-1",
            "1": "FLAIR_t",
            "2": "FLAIR_t+1",
            "3": "T1"
        },
        "labels": {
            "background": 0,
            "WMH": 1
        },
        "numTraining": num_train_slices,
        "file_ending": ".nii.gz"
    }
    
    json_path = target_base / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n✓ dataset.json: {json_path}")
    
    # === 总结 ===
    print("\n" + "="*70)
    print("数据准备完成")
    print("="*70)
    print(f"\n通道配置:")
    print(f"  0: FLAIR[t-1]")
    print(f"  1: FLAIR[t]")
    print(f"  2: FLAIR[t+1]")
    print(f"  3: T1")
    print(f"\n训练: {len(train_cases)} volumes → {num_train_slices} slices")
    print(f"测试: {len(test_cases)} volumes → {num_test_slices} slices")
    print(f"\n下一步:")
    print(f"  nnUNetv2_plan_and_preprocess -d 002")
    print(f"  nnUNetv2_train 002 2d 0")
    
    return num_train_slices, num_test_slices


if __name__ == "__main__":
    prepare_4channel_data()
