"""
快速測試：Z-score正規化對Baseline模型的影響
使用當前訓練好的3個模型在幾個測試樣本上評估
"""
import torch
import numpy as np
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm

from model_2d import UNet2D
from tta_inference import ensemble_tta_predict


def load_models(model_ids=[3, 4, 5], device='cuda'):
    """載入訓練好的模型"""
    models = []
    for model_id in model_ids:
        checkpoint_path = Path(f'checkpoints_2d_model{model_id}/best_model.pth')
        if not checkpoint_path.exists():
            print(f"⚠️ Model {model_id} 不存在")
            continue
        
        model = UNet2D(in_channels=1, out_channels=1)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
        print(f"✓ 載入 Model {model_id}")
    
    return models


def preprocess_slice_zscore(flair_slice):
    """
    Z-score正規化（與訓練時相同）
    """
    mean = flair_slice.mean()
    std = flair_slice.std()
    normalized = (flair_slice - mean) / (std + 1e-8)
    normalized = np.clip(normalized, -5, 5)
    return normalized.astype(np.float32)


def preprocess_slice_no_norm(flair_slice):
    """
    無正規化（僅裁剪範圍）
    """
    # 簡單的min-max到0-1
    normalized = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
    return normalized.astype(np.float32)


def predict_volume(models, flair_path, device='cuda', use_zscore=True, threshold=0.5):
    """
    對整個volume進行預測
    """
    flair_nii = nib.load(str(flair_path))
    flair_data = flair_nii.get_fdata()
    
    H, W, D = flair_data.shape
    prob_volume = np.zeros((H, W, D), dtype=np.float32)
    
    # 選擇預處理方法
    preprocess_fn = preprocess_slice_zscore if use_zscore else preprocess_slice_no_norm
    
    for z in range(D):
        flair_slice = flair_data[:, :, z]
        
        if flair_slice.max() < 1e-6:
            continue
        
        # 預處理
        normalized = preprocess_fn(flair_slice)
        
        # Pad到256x256
        target_size = 256
        h, w = normalized.shape
        
        if h != target_size or w != target_size:
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            normalized = np.pad(normalized, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='constant', constant_values=0)
        
        # 預測
        image_tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        pred = ensemble_tta_predict(models, image_tensor, device)
        pred_np = pred.squeeze().cpu().numpy()
        
        # 移除padding
        if h != target_size or w != target_size:
            pred_np = pred_np[pad_top:pad_top+h, pad_left:pad_left+w]
        
        prob_volume[:, :, z] = pred_np
    
    # 二值化
    binary_mask = (prob_volume > threshold).astype(np.uint8)
    return binary_mask


def calculate_3d_metrics(pred_mask, gt_mask):
    """計算3D metrics"""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    
    return {
        'dice': float(dice),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
        'tp': int(TP),
        'fp': int(FP),
        'fn': int(FN)
    }


def main():
    print("=" * 70)
    print("Z-score正規化對Baseline模型影響測試")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")
    
    # 載入模型
    print("\n載入模型...")
    models = load_models([3, 4, 5], device)
    print(f"成功載入 {len(models)} 個模型")
    
    # 找測試cases（選5個快速測試）
    test_dir = Path('data/wmh/test')
    test_cases = []
    
    for region_dir in test_dir.iterdir():
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
                if flair_path.exists() and wmh_path.exists():
                    test_cases.append(case_dir)
    
    # 只測試5個cases
    test_cases = test_cases[:5]
    print(f"\n測試 {len(test_cases)} 個cases")
    
    # 測試兩種配置
    results = {
        'with_zscore': [],
        'without_zscore': []
    }
    
    print("\n" + "=" * 70)
    print("測試1: 使用Z-score正規化（訓練時的配置）")
    print("=" * 70)
    
    for case_dir in tqdm(test_cases, desc="Z-score正規化"):
        flair_path = case_dir / 'pre' / 'FLAIR.nii.gz'
        wmh_path = case_dir / 'wmh.nii.gz'
        
        # 預測
        pred_mask = predict_volume(models, flair_path, device, use_zscore=True)
        
        # 載入GT
        wmh_nii = nib.load(str(wmh_path))
        gt_mask = (wmh_nii.get_fdata() > 0.5).astype(np.uint8)
        
        # 計算metrics
        metrics = calculate_3d_metrics(pred_mask, gt_mask)
        case_id = f"{case_dir.parent.parent.name}_{case_dir.parent.name}_{case_dir.name}"
        metrics['case_id'] = case_id
        
        results['with_zscore'].append(metrics)
        print(f"  {case_id}: Dice={metrics['dice']:.4f}")
    
    print("\n" + "=" * 70)
    print("測試2: 不使用Z-score正規化（僅min-max）")
    print("=" * 70)
    
    for case_dir in tqdm(test_cases, desc="無正規化"):
        flair_path = case_dir / 'pre' / 'FLAIR.nii.gz'
        wmh_path = case_dir / 'wmh.nii.gz'
        
        # 預測
        pred_mask = predict_volume(models, flair_path, device, use_zscore=False)
        
        # 載入GT
        wmh_nii = nib.load(str(wmh_path))
        gt_mask = (wmh_nii.get_fdata() > 0.5).astype(np.uint8)
        
        # 計算metrics
        metrics = calculate_3d_metrics(pred_mask, gt_mask)
        case_id = f"{case_dir.parent.parent.name}_{case_dir.parent.name}_{case_dir.name}"
        metrics['case_id'] = case_id
        
        results['without_zscore'].append(metrics)
        print(f"  {case_id}: Dice={metrics['dice']:.4f}")
    
    # 計算平均
    print("\n" + "=" * 70)
    print("結果對比")
    print("=" * 70)
    
    metrics_keys = ['dice', 'sensitivity', 'precision']
    
    print(f"\n{'配置':<30} {'Dice':<10} {'Sensitivity':<12} {'Precision':<12}")
    print("-" * 70)
    
    for config_name, config_results in [('使用Z-score正規化', 'with_zscore'), 
                                        ('不使用Z-score', 'without_zscore')]:
        avg_metrics = {}
        for key in metrics_keys:
            values = [r[key] for r in results[config_results]]
            avg_metrics[key] = np.mean(values)
        
        print(f"{config_name:<30} {avg_metrics['dice']:<10.4f} "
              f"{avg_metrics['sensitivity']:<12.4f} {avg_metrics['precision']:<12.4f}")
    
    # 計算差異
    print("\n" + "=" * 70)
    print("差異分析")
    print("=" * 70)
    
    for key in metrics_keys:
        with_z = np.mean([r[key] for r in results['with_zscore']])
        without_z = np.mean([r[key] for r in results['without_zscore']])
        diff = with_z - without_z
        pct_change = (diff / without_z * 100) if without_z > 0 else 0
        
        status = "✅ 提升" if diff > 0 else "⚠️ 下降" if diff < -0.01 else "➖ 相當"
        
        print(f"{key.capitalize():12s}: {with_z:.4f} vs {without_z:.4f} → "
              f"差異 {diff:+.4f} ({pct_change:+.1f}%) {status}")
    
    # 保存結果
    output = {
        'test_cases': [r['case_id'] for r in results['with_zscore']],
        'with_zscore': results['with_zscore'],
        'without_zscore': results['without_zscore'],
        'summary': {
            'with_zscore_avg': {k: np.mean([r[k] for r in results['with_zscore']]) 
                               for k in metrics_keys},
            'without_zscore_avg': {k: np.mean([r[k] for r in results['without_zscore']]) 
                                  for k in metrics_keys}
        }
    }
    
    output_file = 'zscore_impact_test.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n結果已保存至: {output_file}")
    print("=" * 70)
    
    print("\n💡 結論:")
    if output['summary']['with_zscore_avg']['dice'] > output['summary']['without_zscore_avg']['dice']:
        print("  ✅ Z-score正規化對模型性能有正面影響")
        print("  ✅ 建議繼續使用Z-score正規化")
    else:
        print("  ⚠️ Z-score正規化可能對性能有輕微負面影響")
        print("  ⚠️ 需要進一步調查原因")


if __name__ == "__main__":
    main()
