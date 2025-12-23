"""
清理臨時和測試檔案
保留核心專案和baseline資料夾
"""
import os
from pathlib import Path

print("=" * 70)
print("清理臨時和測試檔案")
print("=" * 70)

# 要刪除的檔案列表
files_to_delete = [
    # 測試腳本
    'test_high_recall_baseline.py',
    'check_training_results.py',
    
    # 視覺化腳本（已完成任務）
    'visualize_flair_t1_difference.py',
    'visualize_simple_difference.py',
    
    # 計算腳本
    'calculate_baseline_metrics.py',
    
    # 優化實驗腳本（結果已整合到報告）
    'optimize_and_postprocess.py',
    'final_optimize.py',
    
    # 影像處理baseline（已完成評估）
    'baseline_image_processing.py',
    
    # 清理腳本
    'cleanup_project.py',
    
    # 組織腳本（已執行完成）
    'organize_baseline.py',
    
    # 臨時結果檔案（已整合）
    'high_recall_threshold_analysis.json',
    'baseline_image_processing_results.json',
    'optimized_ensemble_results.json',
    
    # 臨時資料夾
    'temp/',
]

print("\n將要刪除的檔案:")
print("-" * 70)

deleted = []
not_found = []

for item in files_to_delete:
    path = Path(item)
    if path.exists():
        if path.is_file():
            try:
                os.remove(path)
                deleted.append(str(path))
                print(f"  ✓ 已刪除: {path}")
            except Exception as e:
                print(f"  ✗ 無法刪除 {path}: {e}")
        elif path.is_dir():
            try:
                import shutil
                shutil.rmtree(path)
                deleted.append(str(path))
                print(f"  ✓ 已刪除資料夾: {path}")
            except Exception as e:
                print(f"  ✗ 無法刪除資料夾 {path}: {e}")
    else:
        not_found.append(str(path))

if not_found:
    print("\n未找到（可能已刪除）:")
    for item in not_found:
        print(f"  - {item}")

print("\n" + "=" * 70)
print("保留的重要檔案/資料夾:")
print("-" * 70)

keep_items = [
    'data/',
    'openspec/',
    '.agent/',
    'baseline_2d_ensemble/',
    'checkpoints_2d_model3/',
    'checkpoints_2d_model4/',
    'checkpoints_2d_model5/',
    'requirements.txt',
    'metrics.py',
    'model_2d.py',
    'dataset_2d.py',
    'augmentations.py',
    'losses.py',
    'tta_inference.py',
    'train_2d.py',
    'evaluate_ensemble_2d.py',
    'train_all_models_champion.bat',
    'training_summary.json',
    'ensemble_2d_evaluation_results.json',
    'final_optimized_results.json',
]

for item in keep_items:
    status = "✓" if Path(item).exists() else "✗"
    print(f"  {status} {item}")

print("\n" + "=" * 70)
print(f"清理完成！刪除了 {len(deleted)} 個項目")
print("=" * 70)
print("\n專案結構:")
print("  📁 data/                      # 原始數據")
print("  📁 baseline_2d_ensemble/      # ⭐ Baseline完整包")
print("     ├── 核心程式碼 (7個)")
print("     ├── 模型權重 (3個)")
print("     ├── 評估結果 (4個)")
print("     ├── 分析報告 (4個)")
print("     ├── README.md")
print("     └── TECHNICAL_REPORT.md")
print("  📁 checkpoints_2d_model*/     # 原始模型checkpoint")
print("  📄 核心腳本 (7個.py)")
print("  📄 結果檔案 (3個.json)")
print("=" * 70)
