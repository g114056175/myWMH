"""
完整評估腳本：載入模型 + TTA + Ensemble + 評估
只使用訓練成功的模型 (Models 3, 4, 5)
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import nibabel as nib

from model_2d import UNet2D
from tta_inference import ensemble_tta_predict


def load_models(model_ids, device='cuda'):
    """
    載入訓練好的模型
    
    Args:
        model_ids: list of model IDs to load
        device: 計算設備
    
    Returns:
        list of loaded models
    """
    models = []
    
    for model_id in model_ids:
        checkpoint_path = Path(f'checkpoints_2d_model{model_id}/best_model.pth')
        
        if not checkpoint_path.exists():
            print(f"⚠️ Model {model_id} checkpoint不存在")
            continue
        
        # 創建模型
        model = UNet2D(in_channels=1, out_channels=1)
        
        # 載入權重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        models.append(model)
        print(f"✓ 載入Model {model_id}")
    
    return models


def predict_volume_2d(models, flair_path, device='cuda', threshold=0.5):
    """
    對整個3D volume進行預測
    
    Args:
        models: list of trained models
        flair_path: path to FLAIR.nii.gz
        device: 計算設備
        threshold: 二值化閾值
    
    Returns:
        3D prediction mask
    """
    # 載入FLAIR volume
    flair_nii = nib.load(str(flair_path))
    flair_data = flair_nii.get_fdata()
    
    H, W, D = flair_data.shape
    pred_volume = np.zeros((H, W, D), dtype=np.float32)
    
    # 逐切片預測
    for z in range(D):
        flair_slice = flair_data[:, :, z]
        
        # 跳過空白切片
        if flair_slice.max() < 1e-6:
            continue
        
        # 預處理
        mean = flair_slice.mean()
        std = flair_slice.std()
        normalized = (flair_slice - mean) / (std + 1e-8)
        normalized = np.clip(normalized, -5, 5)
        
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
        
        # 轉為tensor
        image_tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        
        # Ensemble預測 (with TTA)
        pred = ensemble_tta_predict(models, image_tensor, device)
        
        # 轉回numpy並移除padding
        pred_np = pred.squeeze().cpu().numpy()
        
        if h != target_size or w != target_size:
            # 移除padding
            pred_np = pred_np[pad_top:pad_top+h, pad_left:pad_left+w]
        
        pred_volume[:, :, z] = pred_np
    
    # 二值化
    binary_mask = (pred_volume > threshold).astype(np.uint8)
    
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
    specificity = TN / (TN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    
    return {
        '3d_dice': float(dice),
        '3d_sensitivity': float(sensitivity),
        '3d_specificity': float(specificity),
        '3d_precision': float(precision),
        'tp': int(TP),
        'fp': int(FP),
        'fn': int(FN)
    }


def main():
    print("=" * 70)
    print("2D Ensemble + TTA 評估")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")
    
    # 只載入成功的模型 (3, 4, 5)
    print("\n載入訓練好的模型...")
    model_ids = [3, 4, 5]  # Dice, Tversky, Combo
    models = load_models(model_ids, device)
    
    if len(models) == 0:
        print("錯誤：沒有可用的模型")
        return
    
    print(f"成功載入 {len(models)} 個模型")
    
    # 找到測試cases
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
                if flair_path.exists():
                    test_cases.append(case_dir)
    
    print(f"\n找到 {len(test_cases)} 個測試cases")
    
    # 限制評估數量（可調整）
    max_cases = min(10, len(test_cases))  # 先評估10個
    test_cases = test_cases[:max_cases]
    
    print(f"評估 {len(test_cases)} 個cases（快速測試）\n")
    
    # 評估
    results = []
    
    for case_dir in tqdm(test_cases, desc="評估中"):
        flair_path = case_dir / 'pre' / 'FLAIR.nii.gz'
        wmh_path = case_dir / 'wmh.nii.gz'
        
        if not wmh_path.exists():
            continue
        
        # 預測
        pred_mask = predict_volume_2d(models, flair_path, device, threshold=0.5)
        
        # 載入ground truth
        wmh_nii = nib.load(str(wmh_path))
        gt_mask = (wmh_nii.get_fdata() > 0.5).astype(np.uint8)
        
        # 計算metrics
        metrics = calculate_3d_metrics(pred_mask, gt_mask)
        
        case_id = f"{case_dir.parent.parent.name}_{case_dir.parent.name}_{case_dir.name}"
        results.append({
            'case_id': case_id,
            **metrics
        })
        
        print(f"  {case_id}: Dice={metrics['3d_dice']:.4f}")
    
    # 計算平均metrics
    avg_metrics = {}
    for key in ['3d_dice', '3d_sensitivity', '3d_specificity', '3d_precision']:
        values = [r[key] for r in results]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    # 保存結果
    final_results = {
        'method': '2D Ensemble (Models 3-5) + TTA',
        'models_used': model_ids,
        'num_models': len(models),
        'individual_results': results,
        'average_metrics': avg_metrics,
        'num_cases': len(results)
    }
    
    with open('ensemble_2d_evaluation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 打印總結
    print("\n" + "=" * 70)
    print("評估結果")
    print("=" * 70)
    print(f"使用模型: Models {model_ids} (共{len(models)}個)")
    print(f"評估cases: {len(results)}")
    print(f"\n平均Metrics:")
    print(f"  3D Dice:        {avg_metrics['3d_dice']:.4f} ± {avg_metrics['3d_dice_std']:.4f}")
    print(f"  3D Sensitivity: {avg_metrics['3d_sensitivity']:.4f} ± {avg_metrics['3d_sensitivity_std']:.4f}")
    print(f"  3D Specificity: {avg_metrics['3d_specificity']:.4f} ± {avg_metrics['3d_specificity_std']:.4f}")
    print(f"  3D Precision:   {avg_metrics['3d_precision']:.4f} ± {avg_metrics['3d_precision_std']:.4f}")
    print(f"\n結果已保存至: ensemble_2d_evaluation_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
