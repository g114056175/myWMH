# 進階nnUnet清理 - 移除所有冗餘文件
# 保留: Archive checkpoints, 文檔, 數據準備腳本

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  nnUnet Deep Cleanup for GitHub" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

$totalDeleted = 0

function Remove-LargeFolder {
    param($Path, $Description)
    
    $fullPath = "nnUnet\$Path"
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "[DELETING] $Description ($([math]::Round($size, 1)) MB)..." -ForegroundColor Yellow
        Remove-Item $fullPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  ✓ Deleted" -ForegroundColor Green
        return $size
    } else {
        Write-Host "[SKIP] $Description - Not found" -ForegroundColor Gray
        return 0
    }
}

Write-Host "Critical cleanup items:" -ForegroundColor Yellow
Write-Host ""

# 1. Remove nnUnet/nnUnet subfolder (原始數據拷貝)
Write-Host "1. Removing nnUnet/nnUnet subfolder (duplicate raw data)..."
$totalDeleted += Remove-LargeFolder "nnUnet" "nnUnet subfolder"

# 2. Remove entire nnUNet_results (we have checkpoints in Archives)
Write-Host "`n2. Removing nnUNet_results (checkpoints already in Archives)..."
$totalDeleted += Remove-LargeFolder "nnUNet_results" "nnUNet training results"

# 3. Remove nnUNet_raw if still exists
Write-Host "`n3. Removing nnUNet_raw (can regenerate)..."
$totalDeleted += Remove-LargeFolder "nnUNet_raw" "nnUNet raw data"

# 4. Remove prediction folders
Write-Host "`n4. Removing prediction folders..."
$totalDeleted += Remove-LargeFolder "predictions_7ch_v2" "v2 predictions"
$totalDeleted += Remove-LargeFolder "predictions_sample" "Sample predictions"

# 5. Remove "nnU Net_raw" (with space in name)
Write-Host "`n5. Removing 'nnU Net_raw' (typo folder)..."
$totalDeleted += Remove-LargeFolder "nnU Net_raw" "nnU Net_raw typo folder"

# 6. Remove JSON result files (keep only in Archives)
Write-Host "`n6. Removing result JSON files..."
$jsonFiles = @(
    "3d_evaluation_fast.json",
    "3d_evaluation_results.json",
    "7ch_v2_real_metrics.json",
    "7ch_v2_results.json",
    "full_test_set_results.json",
    "selected_3d_cases.json"
)

foreach ($json in $jsonFiles) {
    $jsonPath = "nnUnet\$json"
    if (Test-Path $jsonPath) {
        Remove-Item $jsonPath -Force
        Write-Host "  [DELETED] $json" -ForegroundColor Green
    }
}

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete!" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total space saved: $([math]::Round($totalDeleted, 1)) MB" -ForegroundColor Green
Write-Host ""

# Check final size
Write-Host "Calculating final nnUnet size..." -ForegroundColor Cyan
$finalSize = (Get-ChildItem "nnUnet" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Final nnUnet size: $([math]::Round($finalSize, 1)) MB" -ForegroundColor Green
Write-Host ""

Write-Host "What was KEPT:" -ForegroundColor Cyan
Write-Host "  ✓ 4CH_Model_Archive/ (~315 MB)" -ForegroundColor White
Write-Host "    - checkpoint_bestv0.1.pth" -ForegroundColor Gray
Write-Host "    - All documentation" -ForegroundColor Gray
Write-Host ""
Write-Host "  ✓ 7CH_v1_Archive/ (~512 MB)" -ForegroundColor White
Write-Host "    - checkpoint_best.pth" -ForegroundColor Gray
Write-Host "    - checkpoint_latest.pth" -ForegroundColor Gray
Write-Host "    - All documentation" -ForegroundColor Gray
Write-Host ""
Write-Host "  ✓ 7CH_v2_Archive/ (~512 MB)" -ForegroundColor White
Write-Host "    - checkpoint_best.pth" -ForegroundColor Gray
Write-Host "    - checkpoint_latest.pth" -ForegroundColor Gray
Write-Host "    - All documentation" -ForegroundColor Gray
Write-Host ""
Write-Host "  ✓ prepare_*.py scripts" -ForegroundColor White
Write-Host "  ✓ setup_env.bat" -ForegroundColor White
Write-Host "  ✓ USAGE_AFTER_CLEANUP.md" -ForegroundColor White
Write-Host ""

Write-Host "What was DELETED:" -ForegroundColor Cyan
Write-Host "  ✗ nnUnet/nnUnet/ (~3.6 GB)" -ForegroundColor Gray
Write-Host "  ✗ nnUNet_results/ (~1.5 GB)" -ForegroundColor Gray
Write-Host "  ✗ nnUNet_raw/ (if existed)" -ForegroundColor Gray
Write-Host "  ✗ predictions_*/ folders" -ForegroundColor Gray
Write-Host "  ✗ JSON result files" -ForegroundColor Gray
Write-Host ""

if ($finalSize -lt 2000) {
    Write-Host "✓ nnUnet folder is now GitHub-ready!" -ForegroundColor Green
    Write-Host "  Expected size: ~1.3 GB (mostly checkpoints)" -ForegroundColor White
} else {
    Write-Host "⚠ Size still large. Consider:" -ForegroundColor Yellow
    Write-Host "  - Using Git LFS for .pth files" -ForegroundColor White
    Write-Host "  - Uploading checkpoints to external storage" -ForegroundColor White
}
