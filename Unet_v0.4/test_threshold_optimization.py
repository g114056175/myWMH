"""
使用当前最佳模型(Epoch 32)测试不同阈值
重点：降低FP，提高Precision
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import sys
sys.path.append('d:/VSCode/AIOT_E1/Unet_v0.4')

from model import AttentionUNet
from dataset import normalize_slice, center_crop, apply_clahe, compute_highpass, compute_asymmetry
import matplotlib.pyplot as plt

print("="*70)
print("阈值优化 - 降低FP，提高Precision")
print("="*70)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = AttentionUNet(in_channels=7, base_channels=64, depth=5).to(device)
checkpoint = torch.load('d:/VSCode/AIOT_E1/Unet_v0.4/checkpoints/best_model.pth', 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model: Epoch {checkpoint['epoch']}, Val Dice {checkpoint['val_dice']:.4f}")

# 加载spatial atlas
spatial_atlas = np.load('d:/VSCode/AIOT_E1/Unet_v0.4/spatial_atlas_prior.npy')
print(f"Spatial atlas loaded: {spatial_atlas.shape}")

# 测试不同阈值
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
print(f"\n测试阈值: {thresholds}")

# 加载测试集
test_dir = Path('d:/VSCode/AIOT_E1/data/wmh/test')
test_samples = sorted([d for d in test_dir.iterdir() if d.is_dir()])

print(f"\n找到 {len(test_samples)} 测试样本")
print("评估前10个样本...")

results = {th: {'dice': [], 'sensitivity': [], 'precision': [], 'fp': [], 'fn': []} 
          for th in thresholds}

def prepare_input(flair_3d, t1_3d, slice_idx, spatial_atlas):
    """准备7通道输入"""
    # 归一化和裁剪
    flair_norm = normalize_slice(flair_3d)
    t1_norm = normalize_slice(t1_3d)
    flair_crop = center_crop(flair_norm, 224)
    t1_crop = center_crop(t1_norm, 224)
    
    # CLAHE × 3
    clahe_prev = apply_clahe(flair_crop[:,:,max(0, slice_idx-1)])
    clahe_curr = apply_clahe(flair_crop[:,:,slice_idx])
    clahe_next = apply_clahe(flair_crop[:,:,min(flair_crop.shape[2]-1, slice_idx+1)])
    
    # T1
    t1_slice = t1_crop[:,:,slice_idx]
    
    # HighPass
    highpass = compute_highpass(flair_crop[:,:,slice_idx], sigma=2.0)
    
    # Asymmetry
    asymmetry = compute_asymmetry(flair_crop[:,:,slice_idx])
    
    # Stack
    input_7ch = np.stack([
        clahe_prev, clahe_curr, clahe_next,
        t1_slice, highpass, asymmetry, spatial_atlas
    ], axis=0)
    
    return torch.from_numpy(input_7ch).float().unsqueeze(0)

# 只评估前10个样本
for sample_dir in tqdm(test_samples[:10], desc="Processing"):
    # 加载FLAIR和mask
    flair_path = sample_dir / 'pre' / 'FLAIR.nii.gz'
    t1_path = sample_dir / 'pre' / 'T1.nii.gz'
    mask_path = sample_dir / 'wmh.nii.gz'
    
    if not all([flair_path.exists(), t1_path.exists(), mask_path.exists()]):
        continue
    
    flair_3d = nib.load(flair_path).get_fdata()
    t1_3d = nib.load(t1_path).get_fdata()
    mask_3d = nib.load(mask_path).get_fdata()
    
    # 处理每个slice
    for slice_idx in range(2, flair_3d.shape[2] - 2):
        mask_slice = mask_3d[:,:,slice_idx]
        mask_crop = center_crop(mask_slice, 224)
        
        if mask_crop.sum() == 0:
            continue
        
        # 准备输入
        input_tensor = prepare_input(flair_3d, t1_3d, slice_idx, spatial_atlas).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # 测试不同阈值
        for th in thresholds:
            pred = (prob >= th).astype(np.uint8)
            gt = (mask_crop > 0).astype(np.uint8)
            
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            tn = np.sum((pred == 0) & (gt == 0))
            
            # Dice
            dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
            
            # Sensitivity
            sensitivity = tp / (tp + fn + 1e-7)
            
            # Precision
            precision = tp / (tp + fp + 1e-7)
            
            results[th]['dice'].append(dice)
            results[th]['sensitivity'].append(sensitivity)
            results[th]['precision'].append(precision)
            results[th]['fp'].append(fp)
            results[th]['fn'].append(fn)

# 汇总结果
print("\n" + "="*70)
print("阈值优化结果 (前10个样本)")
print("="*70)

print(f"\n{'Threshold':<12} {'Dice':<10} {'Sens':<10} {'Prec':<10} {'FP':<10} {'FN':<10}")
print("-"*70)

best_th = None
best_metric = 0

for th in thresholds:
    avg_dice = np.mean(results[th]['dice'])
    avg_sens = np.mean(results[th]['sensitivity'])
    avg_prec = np.mean(results[th]['precision'])
    avg_fp = np.mean(results[th]['fp'])
    avg_fn = np.mean(results[th]['fn'])
    
    print(f"{th:<12.2f} {avg_dice:<10.4f} {avg_sens:<10.4f} {avg_prec:<10.4f} "
          f"{avg_fp:<10.1f} {avg_fn:<10.1f}")
    
    # 找最佳（优先考虑Precision>0.80的情况下Dice最高）
    if avg_prec >= 0.75:
        if avg_dice > best_metric:
            best_metric = avg_dice
            best_th = th

print("="*70)
print(f"\n推荐阈值: {best_th if best_th else 0.7}")
print(f"  目标: Precision >=0.75 同时Dice最高")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

thresholds_arr = np.array(thresholds)
dice_arr = [np.mean(results[th]['dice']) for th in thresholds]
sens_arr = [np.mean(results[th]['sensitivity']) for th in thresholds]
prec_arr = [np.mean(results[th]['precision']) for th in thresholds]
fp_arr = [np.mean(results[th]['fp']) for th in thresholds]

# Dice
axes[0, 0].plot(thresholds_arr, dice_arr, 'b-o', linewidth=2, markersize=8)
if best_th:
    axes[0, 0].axvline(best_th, color='r', linestyle='--', label=f'Best={best_th}')
axes[0, 0].set_xlabel('Threshold', fontsize=12)
axes[0, 0].set_ylabel('Dice Score', fontsize=12)
axes[0, 0].set_title('Dice vs Threshold', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Precision
axes[0, 1].plot(thresholds_arr, prec_arr, 'g-o', linewidth=2, markersize=8)
axes[0, 1].axhline(0.75, color='orange', linestyle=':', label='Target=0.75')
if best_th:
    axes[0, 1].axvline(best_th, color='r', linestyle='--', label=f'Best={best_th}')
axes[0, 1].set_xlabel('Threshold', fontsize=12)
axes[0, 1].set_ylabel('Precision', fontsize=12)
axes[0, 1].set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Sensitivity
axes[1, 0].plot(thresholds_arr, sens_arr, 'r-o', linewidth=2, markersize=8)
if best_th:
    axes[1, 0].axvline(best_th, color='r', linestyle='--', label=f'Best={best_th}')
axes[1, 0].set_xlabel('Threshold', fontsize=12)
axes[1, 0].set_ylabel('Sensitivity', fontsize=12)
axes[1, 0].set_title('Sensitivity vs Threshold', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# FP count
axes[1, 1].plot(thresholds_arr, fp_arr, 'purple', linewidth=2, marker='o', markersize=8)
if best_th:
    axes[1, 1].axvline(best_th, color='r', linestyle='--', label=f'Best={best_th}')
axes[1, 1].set_xlabel('Threshold', fontsize=12)
axes[1, 1].set_ylabel('Average FP per slice', fontsize=12)
axes[1, 1].set_title('False Positives vs Threshold', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('d:/VSCode/AIOT_E1/TEMP2/threshold_optimization.png', dpi=150)
print(f"\n✓ Saved: d:/VSCode/AIOT_E1/TEMP2/threshold_optimization.png")
