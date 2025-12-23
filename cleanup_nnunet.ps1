# nnUnet Folder Cleanup Script
# Keeps only essential model weights and documentation
# Removes preprocessed data and training intermediates

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  nnUnet Folder Cleanup" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

$nnUnetPath = "nnUnet"
$totalDeleted = 0

# Function to safely delete folder and report size
function Remove-SafeFolder {
    param($Path, $Description)
    
    $fullPath = Join-Path $nnUnetPath $Path
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
        Remove-Item $fullPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "[DELETED] $Description" -ForegroundColor Green
        Write-Host "          Saved: $([math]::Round($size, 1)) MB" -ForegroundColor Gray
        return $size
    } else {
        Write-Host "[SKIP] $Description - Not found" -ForegroundColor Gray
        return 0
    }
}

Write-Host "Step 1: Removing preprocessed data folders..." -ForegroundColor Yellow
Write-Host "These contain slice-level preprocessed images (can be regenerated)"
Write-Host ""
$totalDeleted += Remove-SafeFolder "nnUNet_preprocessed" "Preprocessed training data"
$totalDeleted += Remove-SafeFolder "nnUNet_raw" "Raw training data copies"

Write-Host "`nStep 2: Cleaning training intermediate files..." -ForegroundColor Yellow
Write-Host "Keeping only best checkpoints in Archives"
Write-Host ""

# For each dataset in results, keep only fold_0 best checkpoint
$resultsPath = Join-Path $nnUnetPath "nnUNet_results"
if (Test-Path $resultsPath) {
    Get-ChildItem $resultsPath -Directory | ForEach-Object {
        $datasetName = $_.Name
        $foldPath = Join-Path $_.FullName "nnUNetTrainer__nnUNetPlans__2d\fold_0"
        
        if (Test-Path $foldPath) {
            # Keep only checkpoint_best.pth and checkpoint_latest.pth
            Get-ChildItem $foldPath -File | Where-Object {
                $_.Name -notlike "checkpoint_best*" -and 
                $_.Name -notlike "checkpoint_latest*" -and
                $_.Name -notlike "checkpoint_final*"
            } | ForEach-Object {
                $size = $_.Length / 1MB
                Remove-Item $_.FullName -Force
                $totalDeleted += $size
                Write-Host "[DELETED] $datasetName - $($_.Name) ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
            }
        }
    }
}

Write-Host "`nStep 3: Removing temporary evaluation folders..." -ForegroundColor Yellow
$totalDeleted += Remove-SafeFolder "temp_v2_eval" "Temporary v2 evaluation"
$totalDeleted += Remove-SafeFolder "batch_eval_7ch_v2" "Batch evaluation temp"
$totalDeleted += Remove-SafeFolder "predictions_7ch_v1" "v1 predictions (archived)"

Write-Host "`nStep 4: Removing old evaluation scripts..." -ForegroundColor Yellow
$scriptsToDelete = @(
    "evaluate_sample.py",
    "evaluate_7ch_v1.py", 
    "evaluate_7ch_v2_quick.py",
    "evaluate_7ch_v2_batch.py",
    "compute_real_confusion_matrix.py",
    "plot_7ch_v1_curves.py",
    "plot_7ch_v2_curves.py",
    "plot_7ch_v2_eval.py",
    "plot_7ch_v2_final.py",
    "create_v2_confusion_matrix.py",
    "update_v2_archive.py"
)

foreach ($script in $scriptsToDelete) {
    $scriptPath = Join-Path $nnUnetPath $script
    if (Test-Path $scriptPath) {
        Remove-Item $scriptPath -Force
        Write-Host "[DELETED] $script" -ForegroundColor Green
    }
}

Write-Host "`nStep 5: Removing JSON result files..." -ForegroundColor Yellow
$jsonFiles = @(
    "selected_3d_cases.json",
    "7ch_v2_results.json",
    "7ch_v2_real_metrics.json"
)

foreach ($json in $jsonFiles) {
    $jsonPath = Join-Path $nnUnetPath $json
    if (Test-Path $jsonPath) {
        Remove-Item $jsonPath -Force
        Write-Host "[DELETED] $json" -ForegroundColor Green
    }
}

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete!" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total space saved: $([math]::Round($totalDeleted, 1)) MB" -ForegroundColor Green
Write-Host ""
Write-Host "What was KEPT:" -ForegroundColor Cyan
Write-Host "  ✓ 4CH_Model_Archive/ (checkpoint + docs)" -ForegroundColor White
Write-Host "  ✓ 7CH_v1_Archive/ (checkpoint + docs)" -ForegroundColor White
Write-Host "  ✓ 7CH_v2_Archive/ (checkpoint + docs)" -ForegroundColor White
Write-Host "  ✓ prepare_7channel_*.py scripts" -ForegroundColor White
Write-Host "  ✓ evaluate_7ch_*_complete.py scripts" -ForegroundColor White
Write-Host ""
Write-Host "What was DELETED:" -ForegroundColor Cyan
Write-Host "  ✗ nnUNet_preprocessed/ (can regenerate)" -ForegroundColor Gray
Write-Host "  ✗ nnUNet_raw/ (can regenerate)" -ForegroundColor Gray
Write-Host "  ✗ Training intermediate checkpoints" -ForegroundColor Gray
Write-Host "  ✗ Temporary evaluation folders" -ForegroundColor Gray
Write-Host "  ✗ One-time evaluation scripts" -ForegroundColor Gray
Write-Host ""
Write-Host "See Archive READMEs for usage instructions!" -ForegroundColor Green
