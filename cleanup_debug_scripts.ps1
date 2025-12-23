# 刪除測試/Debug腳本
# 清理所有分析、測試、可視化用的臨時腳本

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Removing Test/Debug Scripts" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

$deletedCount = 0
$savedSpace = 0

# Function to delete script
function Remove-DebugScript {
    param($Path)
    
    if (Test-Path $Path) {
        $size = (Get-Item $Path).Length / 1KB
        Remove-Item $Path -Force
        $script:deletedCount++
        $script:savedSpace += $size
        Write-Host "[DELETED] $Path" -ForegroundColor Green
        return $true
    }
    return $false
}

Write-Host "Step 1: Root directory..." -ForegroundColor Yellow
Remove-DebugScript "add_metrics_to_demo.py"
Remove-DebugScript "generate_demo_output.py"
Remove-DebugScript "check_training_data.py"
Remove-DebugScript "cleanup_temp_files.py"
Remove-DebugScript "evaluate_ensemble_2d.py"
Remove-DebugScript "test_zscore_impact.py"
Remove-DebugScript "tta_inference.py"

Write-Host "`nStep 2: Unet_v0.4/ ..." -ForegroundColor Yellow
$v04Scripts = @(
    "analyze_channel_weights_v04.py",
    "analyze_early_stop.py",
    "analyze_error_types.py",
    "analyze_v04_results.py",
    "analyze_v06_channels.py",
    "check_data_split.py",
    "compare_with_baseline.py",
    "demo_flip_types.py",
    "evaluate_test_3d.py",
    "evaluate_tta.py",
    "get_channel_weights.py",
    "recheck_training.py",
    "test_threshold_optimization.py",
    "verify_split.py",
    "visualize_mixup_demo.py"
)
foreach ($script in $v04Scripts) {
    Remove-DebugScript "Unet_v0.4\$script"
}

Write-Host "`nStep 3: newUnet/ ..." -ForegroundColor Yellow
$newUnetScripts = @(
    "analyze_channel_importance.py",
    "analyze_spatial_map.py",
    "check_ckpt.py",
    "evaluate.py",
    "evaluate_test_3d.py",
    "evaluate_tta.py",
    "generate_complete_report.py",
    "generate_final_report.py",
    "quick_sanity_check.py",
    "quick_test_10samples.py",
    "quick_test_check.py",
    "quick_val.py",
    "quick_validate.py",
    "save_checkpoint.py",
    "test_5samples.py",
    "test_thresholds.py",
    "test_validation.py",
    "update_snapshot.py",
    "validate_components.py",
    "visualize_basic.py",
    "visualize_enhancement_methods.py",
    "visualize_final_methods.py",
    "visualize_poor_samples.py",
    "visualize_predictions.py"
)
foreach ($script in $newUnetScripts) {
    Remove-DebugScript "newUnet\$script"
}

Write-Host "`nStep 4: newUnet2/ ..." -ForegroundColor Yellow
$newUnet2Scripts = @(
    "analyze_channel_importance.py",
    "check_augmentation_flip.py",
    "check_dimensions.py",
    "diagnose_data_loading.py",
    "diagnose_flip_axis.py",
    "evaluate_test_3d.py",
    "generate_final_evaluation_report.py",
    "generate_spatial_atlas.py",
    "plot_training_curves.py",
    "quick_validate_atlas.py",
    "test_asymmetry_feature.py",
    "test_raw_flair_aug.py",
    "test_tta.py",
    "visualize_augmentation.py",
    "visualize_augmentation_CLAHE.py",
    "visualize_augmentation_detailed.py",
    "visualize_spatial_atlas.py"
)
foreach ($script in $newUnet2Scripts) {
    Remove-DebugScript "newUnet2\$script"
}

Write-Host "`nStep 5: nnUnet/ ..." -ForegroundColor Yellow
$nnUnetScripts = @(
    "analyze_channel_importance.py",
    "evaluate_3d_fast.py",
    "evaluate_7ch_v1_complete.py",
    "evaluate_7ch_v2_complete.py",
    "evaluate_7ch_v2_full.py",
    "show_results.py"
)
foreach ($script in $nnUnetScripts) {
    Remove-DebugScript "nnUnet\$script"
}

Write-Host "`nStep 6: TEMP_nnUnet/ ..." -ForegroundColor Yellow
Remove-DebugScript "TEMP_nnUnet\visualize_feature_samples.py"

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete!" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Deleted: $deletedCount scripts" -ForegroundColor Green
Write-Host "Saved: $([math]::Round($savedSpace, 1)) KB" -ForegroundColor Green
Write-Host ""
Write-Host "Kept:" -ForegroundColor Cyan
Write-Host "  ✓ Core scripts: train.py, model.py, dataset.py" -ForegroundColor White
Write-Host "  ✓ Data preparation: prepare_*.py" -ForegroundColor White
Write-Host "  ✓ Configuration: config.py, losses.py" -ForegroundColor White
Write-Host "  ✓ All Archive folders" -ForegroundColor White
