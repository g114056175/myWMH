"""
nnU-Net数据准备脚本 - 2通道baseline (FLAIR + T1)

关键要求:
1. Patient-level处理 (每个volume作为整体)
2. 训练/验证不混到同一病人
3. 60训练 + 110测试
"""

import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def prepare_nnunet_data():
    print("="*70)
    print("nnU-Net数据准备 - 2通道Baseline")
    print("="*70)
    
    # 路径配置
    source_train = Path("d:/VSCode/AIOT_E1/data/wmh/training")
    source_test = Path("d:/VSCode/AIOT_E1/data/wmh/test")
    
    target_base = Path("d:/VSCode/AIOT_E1/nnUnet/nnUNet_raw/Dataset001_WMH")
    
    # 创建目录
    target_base.mkdir(parents=True, exist_ok=True)
    (target_base / "imagesTr").mkdir(exist_ok=True)
    (target_base / "labelsTr").mkdir(exist_ok=True)
    (target_base / "imagesTs").mkdir(exist_ok=True)
    (target_base / "labelsTs").mkdir(exist_ok=True)  # 可选，用于评估
    
    print(f"\n数据源:")
    print(f"  训练: {source_train}")
    print(f"  测试: {source_test}")
    print(f"\n目标: {target_base}")
    
    # ===== 训练数据 =====
    print("\n" + "="*70)
    print("处理训练数据")
    print("="*70)
    
    # 找到所有训练病例目录 (支持不同结构)
    train_cases = []
    for site_dir in source_train.iterdir():
        if not site_dir.is_dir():
            continue
        
        # 检查是否是直接包含流水号的目录 (Singapore, Utrecht)
        # 或包含scanner子目录 (Amsterdam/GE3T/)
        
        has_scanner_subdir = any(d.is_dir() and not d.name.isdigit() for d in site_dir.iterdir())
        
        if has_scanner_subdir:
            # Amsterdam结构: site/scanner/case_number/
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
                                'dir': case_dir,
                                'site': site_dir.name,
                                'scanner': scanner_dir.name,
                                'case_id': case_dir.name,
                                'flair': flair_path,
                                't1': t1_path,
                                'mask': mask_path
                            })
        else:
            # Singapore/Utrecht结构: site/case_number/
            for case_dir in site_dir.iterdir():
                if case_dir.is_dir() and case_dir.name.replace('_', '').isdigit():
                    flair_path = case_dir / "pre" / "FLAIR.nii.gz"
                    t1_path = case_dir / "pre" / "T1.nii.gz"
                    mask_path = case_dir / "wmh.nii.gz"
                    
                    if flair_path.exists() and t1_path.exists() and mask_path.exists():
                        train_cases.append({
                            'dir': case_dir,
                            'site': site_dir.name,
                            'scanner': 'default',
                            'case_id': case_dir.name,
                            'flair': flair_path,
                            't1': t1_path,
                            'mask': mask_path
                        })
    
    train_cases = sorted(train_cases, key=lambda x: (x['site'], x['scanner'], x['case_id']))
    
    print(f"\n找到 {len(train_cases)} 个训练病例:")
    sites = {}
    for case in train_cases:
        site = case['site']
        sites[site] = sites.get(site, 0) + 1
    for site, count in sites.items():
        print(f"  {site}: {count} cases")
    
    # 转换训练数据
    print(f"\n开始转换...")
    for idx, case in enumerate(tqdm(train_cases, desc="训练数据"), 1):
        case_id = f"WMH_{idx:04d}"
        
        # Channel 0: FLAIR (直接复制原始数据)
        flair_dst = target_base / "imagesTr" / f"{case_id}_0000.nii.gz"
        shutil.copy(case['flair'], flair_dst)
        
        # Channel 1: T1
        t1_dst = target_base / "imagesTr" / f"{case_id}_0001.nii.gz"
        shutil.copy(case['t1'], t1_dst)
        
        # Label (处理WMH Challenge的标签2)
        # 0=background, 1=WMH, 2=don't care → 将2转为0
        mask_nii = nib.load(case['mask'])
        mask_data = mask_nii.get_fdata()
        
        # 将标签2（don't care）转为0（background）
        mask_data[mask_data == 2] = 0
        
        # 保存修正后的mask
        mask_corrected = nib.Nifti1Image(mask_data.astype(np.uint8), 
                                         mask_nii.affine, 
                                         mask_nii.header)
        mask_dst = target_base / "labelsTr" / f"{case_id}.nii.gz"
        nib.save(mask_corrected, mask_dst)
    
    print(f"\n✓ 训练数据转换完成: {len(train_cases)} cases")
    
    # ===== 测试数据 =====
    print("\n" + "="*70)
    print("处理测试数据")
    print("="*70)
    
    # 找到所有测试病例 (支持不同结构)
    test_cases = []
    for site_dir in source_test.iterdir():
        if not site_dir.is_dir():
            continue
        
        has_scanner_subdir = any(d.is_dir() and not d.name.isdigit() for d in site_dir.iterdir())
        
        if has_scanner_subdir:
            # Amsterdam结构
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
                                'dir': case_dir,
                                'site': site_dir.name,
                                'scanner': scanner_dir.name,
                                'case_id': case_dir.name,
                                'flair': flair_path,
                                't1': t1_path,
                                'mask': mask_path if mask_path.exists() else None
                            })
        else:
            # Singapore/Utrecht结构
            for case_dir in site_dir.iterdir():
                if case_dir.is_dir() and case_dir.name.replace('_', '').isdigit():
                    flair_path = case_dir / "pre" / "FLAIR.nii.gz"
                    t1_path = case_dir / "pre" / "T1.nii.gz"
                    mask_path = case_dir / "wmh.nii.gz"
                    
                    if flair_path.exists() and t1_path.exists():
                        test_cases.append({
                            'dir': case_dir,
                            'site': site_dir.name,
                            'scanner': 'default',
                            'case_id': case_dir.name,
                            'flair': flair_path,
                            't1': t1_path,
                            'mask': mask_path if mask_path.exists() else None
                        })
    
    test_cases = sorted(test_cases, key=lambda x: (x['site'], x['scanner'], x['case_id']))
    
    print(f"\n找到 {len(test_cases)} 个测试病例:")
    sites = {}
    for case in test_cases:
        site = case['site']
        sites[site] = sites.get(site, 0) + 1
    for site, count in sites.items():
        print(f"  {site}: {count} cases")
    
    # 转换测试数据
    print(f"\n开始转换...")
    for idx, case in enumerate(tqdm(test_cases, desc="测试数据"), 1):
        case_id = f"WMH_test_{idx:04d}"
        
        # FLAIR
        flair_dst = target_base / "imagesTs" / f"{case_id}_0000.nii.gz"
        shutil.copy(case['flair'], flair_dst)
        
        # T1
        t1_dst = target_base / "imagesTs" / f"{case_id}_0001.nii.gz"
        shutil.copy(case['t1'], t1_dst)
        
        # Label (如果有，处理标签2)
        if case['mask']:
            mask_nii = nib.load(case['mask'])
            mask_data = mask_nii.get_fdata()
            mask_data[mask_data == 2] = 0
            mask_corrected = nib.Nifti1Image(mask_data.astype(np.uint8),
                                            mask_nii.affine,
                                            mask_nii.header)
            mask_dst = target_base / "labelsTs" / f"{case_id}.nii.gz"
            nib.save(mask_corrected, mask_dst)
    
    print(f"\n✓ 测试数据转换完成: {len(test_cases)} cases")
    
    # ===== 创建dataset.json =====
    print("\n" + "="*70)
    print("创建dataset.json")
    print("="*70)
    
    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1"
        },
        "labels": {
            "background": 0,
            "WMH": 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz"
    }
    
    json_path = target_base / "dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ dataset.json创建完成: {json_path}")
    
    # ===== 总结 =====
    print("\n" + "="*70)
    print("数据准备完成总结")
    print("="*70)
    print(f"\n训练数据: {len(train_cases)} cases")
    print(f"测试数据: {len(test_cases)} cases")
    print(f"\n通道配置:")
    print(f"  Channel 0: FLAIR (原始)")
    print(f"  Channel 1: T1 (原始)")
    print(f"\n文件结构:")
    print(f"  {target_base}/")
    print(f"    imagesTr/  ({len(train_cases) * 2} files)")
    print(f"    labelsTr/  ({len(train_cases)} files)")
    print(f"    imagesTs/  ({len(test_cases) * 2} files)")
    print(f"    labelsTs/  ({len([c for c in test_cases if c['mask']])} files)")
    print(f"    dataset.json")
    
    print(f"\n✅ 数据准备完成!")
    print(f"\n下一步:")
    print(f"  1. cd d:\\VSCode\\AIOT_E1\\nnUnet")
    print(f"  2. 设置环境变量 (setup_env.bat)")
    print(f"  3. nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
    print(f"  4. nnUNetv2_train 001 2d 0")
    
    return len(train_cases), len(test_cases)


if __name__ == "__main__":
    try:
        num_train, num_test = prepare_nnunet_data()
        print(f"\n" + "="*70)
        print(f"SUCCESS: {num_train} train + {num_test} test cases ready")
        print("="*70)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
