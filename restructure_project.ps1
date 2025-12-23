# 專案結構重組腳本
# 整理成清晰的GitHub專案結構

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Project Restructuring for GitHub" -ForegroundColor Cyan  
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Rename nnUnet to models
Write-Host "Step 1: Renaming nnUnet → models..." -ForegroundColor Yellow
if (Test-Path "nnUnet") {
    if (Test-Path "models") {
        Remove-Item "models" -Recurse -Force
    }
    Rename-Item "nnUnet" "models"
    Write-Host "  ✓ Renamed" -ForegroundColor Green
}

# 2. Create experiments folder
Write-Host "`nStep 2: Creating experiments folder..." -ForegroundColor Yellow
if (-not (Test-Path "experiments")) {
    New-Item -ItemType Directory "experiments" | Out-Null
}

# Move Unet_v0.4
if (Test-Path "Unet_v0.4") {
    if (Test-Path "experiments\baseline_unet") {
        Remove-Item "experiments\baseline_unet" -Recurse -Force
    }
    Move-Item "Unet_v0.4" "experiments\baseline_unet"
    Write-Host "  ✓ Moved Unet_v0.4 → experiments/baseline_unet" -ForegroundColor Green
}

# Move newUnet
if (Test-Path "newUnet") {
    if (Test-Path "experiments\multi_channel_v1") {
        Remove-Item "experiments\multi_channel_v1" -Recurse -Force  
    }
    Move-Item "newUnet" "experiments\multi_channel_v1"
    Write-Host "  ✓ Moved newUnet → experiments/multi_channel_v1" -ForegroundColor Green
}

# 3. Create docs folder
Write-Host "`nStep 3: Creating docs folder..." -ForegroundColor Yellow
if (-not (Test-Path "docs")) {
    New-Item -ItemType Directory "docs" | Out-Null
}

$docsToMove = @(
    "GITHUB_UPLOAD_GUIDE.md",
    "DATA_README.md",
    "7CHANNEL_CONFIG.md",
    "CLEANUP_RECOMMENDATIONS.md",
    "DEBUG_SCRIPTS_ANALYSIS.md",
    "RESTRUCTURE_PLAN.md"
)

foreach ($doc in $docsToMove) {
    if (Test-Path $doc) {
        Move-Item $doc "docs\" -Force
        Write-Host "  ✓ Moved $doc → docs/" -ForegroundColor Green
    }
}

# Move from models if exists
if (Test-Path "models\USAGE_AFTER_CLEANUP.md") {
    Move-Item "models\USAGE_AFTER_CLEANUP.md" "docs\" -Force
    Write-Host "  ✓ Moved USAGE_AFTER_CLEANUP.md → docs/" -ForegroundColor Green
}

# 4. Delete temporary folders
Write-Host "`nStep 4: Deleting temporary folders..." -ForegroundColor Yellow
$tempFolders = @("TEMP", "TEMP2", "TEMP_nnUnet", "newUnet2")
foreach ($folder in $tempFolders) {
    if (Test-Path $folder) {
        Remove-Item $folder -Recurse -Force
        Write-Host "  ✓ Deleted $folder" -ForegroundColor Green
    }
}

# 5. Delete old experimental folders
Write-Host "`nStep 5: Deleting old experimental folders..." -ForegroundColor Yellow  
$oldFolders = @("baseline_2d_ensemble", "nnUnet2", ".agent")
foreach ($folder in $oldFolders) {
    if (Test-Path $folder) {
        Remove-Item $folder -Recurse -Force
        Write-Host "  ✓ Deleted $folder" -ForegroundColor Green
    }
}

# 6. Clean up root directory scripts
Write-Host "`nStep 6: Cleaning up root directory..." -ForegroundColor Yellow
$rootScripts = @(
    "add_metrics_to_demo.py",
    "generate_demo_output.py", 
    "check_training_data.py",
    "cleanup_temp_files.py",
    "test_zscore_impact.py",
    "tta_inference.py",
    "train_2d.py",
    "evaluate_ensemble_2d.py",
    "augmentations.py",
    "dataset_2d.py",
    "losses.py",
    "metrics.py",
    "model_2d.py"
)

foreach ($script in $rootScripts) {
    if (Test-Path $script) {
        Remove-Item $script -Force
        Write-Host "  ✓ Deleted $script" -ForegroundColor Green
    }
}

# Delete cleanup scripts themselves
$cleanupScripts = @(
    "cleanup_nnunet.ps1",
    "cleanup_nnunet_deep.ps1",
    "cleanup_debug_scripts.ps1",
    "cleanup_for_github.ps1"
)

foreach ($script in $cleanupScripts) {
    if (Test-Path $script) {
        Remove-Item $script -Force
        Write-Host "  ✓ Deleted $script" -ForegroundColor Green
    }
}

# 7. Create README in experiments
Write-Host "`nStep 7: Creating experiment READMEs..." -ForegroundColor Yellow

$expReadme = @"
# Experimental Models

This folder contains earlier experimental approaches before adopting nnU-Net.

## baseline_unet/
Custom 2.5D Attention U-Net implementation.
- **Result**: Dice ~0.78-0.80
- **Issue**: Severe overfitting with small dataset
- **Status**: Archived

## multi_channel_v1/
7-channel model with CLAHE and Spatial Map.
- **Result**: Validation Dice ~0.85-0.88 (expected)
- **Issue**: Complex, high VRAM usage
- **Status**: Archived

## Why Switch to nnU-Net?
- Better generalization
- Automatic configuration
- State-of-the-art performance
- Final Dice: **0.8575** (7CH v2)

See `../models/` for production models.
"@

if (Test-Path "experiments") {
    $expReadme | Out-File "experiments\README.md" -Encoding utf8
    Write-Host "  ✓ Created experiments/README.md" -ForegroundColor Green
}

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "  Restructuring Complete!" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Show final structure
Write-Host "Final structure:" -ForegroundColor Cyan
Write-Host ""
Write-Host "AIOT_E1/" -ForegroundColor White
Write-Host "  ├── README.md" -ForegroundColor Gray
Write-Host "  ├── PROJECT_REPORT.md" -ForegroundColor Gray
Write-Host "  ├── .gitignore" -ForegroundColor Gray
Write-Host "  ├── models/" -ForegroundColor Green
Write-Host "  │   ├── 4CH_Model_Archive/" -ForegroundColor Gray
Write-Host "  │   ├── 7CH_v1_Archive/" -ForegroundColor Gray
Write-Host "  │   └── 7CH_v2_Archive/ ⭐" -ForegroundColor Yellow
Write-Host "  ├── output/" -ForegroundColor Green
Write-Host "  │   └── demo_predictions/" -ForegroundColor Gray
Write-Host "  ├── experiments/" -ForegroundColor Green
Write-Host "  │   ├── baseline_unet/" -ForegroundColor Gray
Write-Host "  │   └── multi_channel_v1/" -ForegroundColor Gray
Write-Host "  └── docs/" -ForegroundColor Green
Write-Host "      └── *.md files" -ForegroundColor Gray
Write-Host ""
Write-Host "[OK] Project is now GitHub-ready!" -ForegroundColor Green
