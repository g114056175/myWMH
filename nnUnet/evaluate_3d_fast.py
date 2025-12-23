"""
加速版3D評估 - 隨機抽樣5個新案例
批量處理，模型只載入一次
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import json
import subprocess
import shutil
import random

# 載入所有有病灶的案例
with open("selected_3d_cases.json", "r") as f:
    all_cases = json.load(f)

# 使用全部110個案例進行完整測試集評估
selected_cases = all_cases  # 全部案例

print("=" * 80)
print(f"完整測試集評估 - {len(selected_cases)} 個案例")
print("=" * 80)
print(f"\n使用checkpoint: checkpoint_bestv0.1.pth")
print(f"預計時間: ~10-15分鐘\n")

# 創建批量處理目錄
batch_dir = Path("batch_eval")
if batch_dir.exists():
    shutil.rmtree(batch_dir)
batch_dir.mkdir()

images_dir = batch_dir / "images"
labels_dir = batch_dir / "labels"
predictions_dir = batch_dir / "predictions"

images_dir.mkdir()
labels_dir.mkdir()
predictions_dir.mkdir()

# 記錄每個案例的slice範圍
case_slice_ranges = {}
global_slice_idx = 0

print("\n" + "=" * 80)
print("準備所有案例的slices...")
print("=" * 80)

for case_idx, case in enumerate(selected_cases):
    case_id = case['id'].replace('/', '_').replace('\\', '_')
    case_path = Path(case['path'])
    
    # 載入數據
    flair_path = case_path / "pre" / "FLAIR.nii.gz"
    t1_path = case_path / "pre" / "T1.nii.gz"
    wmh_path = case_path / "wmh.nii.gz"
    
    flair_nii = nib.load(flair_path)
    t1_nii = nib.load(t1_path)
    wmh_nii = nib.load(wmh_path)
    
    flair_data = flair_nii.get_fdata()
    t1_data = t1_nii.get_fdata()
    wmh_data = wmh_nii.get_fdata()
    
    num_slices = flair_data.shape[2]
    start_idx = global_slice_idx
    
    # 處理所有slices
    for z in range(num_slices):
        slice_name = f"case{case_idx:02d}_slice{z:03d}"
        
        # 準備4通道
        flair_prev = flair_data[:, :, max(0, z-1)]
        flair_curr = flair_data[:, :, z]
        flair_next = flair_data[:, :, min(num_slices-1, z+1)]
        t1_curr = t1_data[:, :, z]
        
        # 保存4個通道
        for ch_idx, data in enumerate([flair_prev, flair_curr, flair_next, t1_curr]):
            img_2d = nib.Nifti1Image(data.astype(np.float32), flair_nii.affine)
            img_path = images_dir / f"{slice_name}_{ch_idx:04d}.nii.gz"
            nib.save(img_2d, img_path)
        
        # 保存label
        mask_slice = wmh_data[:, :, z]
        mask_2d = nib.Nifti1Image(mask_slice.astype(np.uint8), flair_nii.affine)
        mask_path = labels_dir / f"{slice_name}.nii.gz"
        nib.save(mask_2d, mask_path)
        
        global_slice_idx += 1
    
    end_idx = global_slice_idx
    case_slice_ranges[case_id] = {
        'start': start_idx,
        'end': end_idx,
        'num_slices': num_slices,
        'case_info': case
    }
    
    print(f"✓ {case_id}: {num_slices} slices")

print(f"\n總計: {global_slice_idx} slices")

# 一次性運行預測
print("\n" + "=" * 80)
print("運行批量預測（模型只載入一次）...")
print("=" * 80)

cmd = [
    "nnUNetv2_predict",
    "-i", str(images_dir.absolute()),
    "-o", str(predictions_dir.absolute()),
    "-d", "002",
    "-c", "2d",
    "-f", "0",
    "-chk", "checkpoint_bestv0.1.pth"  # 使用備份的checkpoint
]

result = subprocess.run(cmd, cwd="d:/VSCode/AIOT_E1/nnUnet", 
                       capture_output=True, text=True)

if result.returncode != 0:
    print(f"預測失敗: {result.stderr}")
    exit(1)

print("✓ 預測完成")

# 計算每個案例的3D Dice
print("\n" + "=" * 80)
print("計算3D Dice...")
print("=" * 80)

results = []

for case_id, info in case_slice_ranges.items():
    print(f"\n{case_id}:")
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    
    # 遍歷該案例的所有slices
    pred_files = sorted(predictions_dir.glob(f"case{list(case_slice_ranges.keys()).index(case_id):02d}_slice*.nii.gz"))
    label_files = sorted(labels_dir.glob(f"case{list(case_slice_ranges.keys()).index(case_id):02d}_slice*.nii.gz"))
    
    for pred_path, label_path in zip(pred_files, label_files):
        pred_data = nib.load(pred_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()
        
        # 忽略 label=2
        valid_mask = (label_data != 2)
        
        pred_bin = ((pred_data > 0) & valid_mask).astype(bool)
        true_bin = ((label_data == 1) & valid_mask).astype(bool)
        
        tp_total += np.sum(pred_bin & true_bin)
        fp_total += np.sum(pred_bin & ~true_bin)
        fn_total += np.sum(~pred_bin & true_bin)
        tn_total += np.sum(~pred_bin & ~true_bin)
    
    # 計算metrics
    if (2 * tp_total + fp_total + fn_total) == 0:
        dice_3d = 1.0
    else:
        dice_3d = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    
    sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    
    results.append({
        'case': case_id,
        'volume': info['case_info']['volume'],
        'metrics': {
            'dice': dice_3d,
            'sensitivity': sensitivity,
            'precision': precision,
            'tp': int(tp_total),
            'fp': int(fp_total),
            'fn': int(fn_total),
            'tn': int(tn_total)
        }
    })
    
    print(f"  Dice: {dice_3d:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  TP: {tp_total}, FP: {fp_total}, FN: {fn_total}")

# 清理臨時文件
print("\n清理臨時文件...")
shutil.rmtree(batch_dir)

# 總結
print("\n" + "=" * 80)
print("評估完成")
print("=" * 80)

avg_dice = np.mean([r['metrics']['dice'] for r in results])
print(f"\n平均3D Dice: {avg_dice:.4f}")

# 詳細表格
print("\n詳細結果:")
print("-" * 80)
print(f"{'案例':<30} {'病灶大小':<12} {'Dice':<10} {'Sens':<10} {'Prec':<10}")
print("-" * 80)
for r in results:
    print(f"{r['case']:<30} {r['volume']:<12} {r['metrics']['dice']:<10.4f} "
          f"{r['metrics']['sensitivity']:<10.4f} {r['metrics']['precision']:<10.4f}")
print("-" * 80)

# 保存結果
with open("full_test_set_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ 結果已保存到 full_test_set_results.json")
