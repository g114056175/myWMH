# GitHub Upload Cleanup Script
# Run this to prepare project for GitHub upload

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  WMH Project - GitHub Cleanup Script" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Get initial size
$initialSize = (Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Initial project size: $([math]::Round($initialSize, 2)) GB" -ForegroundColor Yellow
Write-Host ""

$deletedSize = 0

# Function to safely delete folder
function Remove-SafeFolder {
    param($Path, $Description)
    
    if (Test-Path $Path) {
        $size = (Get-ChildItem $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
        Remove-Item $Path -Recurse -Force
        Write-Host "[DELETED] $Description - Saved $([math]::Round($size, 1)) MB" -ForegroundColor Green
        return $size
    } else {
        Write-Host "[SKIP] $Description - Not found" -ForegroundColor Gray
        return 0
    }
}

Write-Host "Step 1: Deleting data folder..." -ForegroundColor Cyan
$deletedSize += Remove-SafeFolder "data" "Data folder (medical images)"

Write-Host "`nStep 2: Deleting nnU-Net generated files..." -ForegroundColor Cyan
$deletedSize += Remove-SafeFolder "nnUnet\nnUNet_raw" "nnU-Net raw data"
$deletedSize += Remove-SafeFolder "nnUnet\nnUNet_preprocessed" "nnU-Net preprocessed"

Write-Host "`nStep 3: Deleting temporary evaluation folders..." -ForegroundColor Cyan
$deletedSize += Remove-SafeFolder "nnUnet\temp_v2_eval" "Temp v2 evaluation"
$deletedSize += Remove-SafeFolder "nnUnet\batch_eval_7ch_v2" "Batch eval v2"
$deletedSize += Remove-SafeFolder "nnUnet\predictions_7ch_v1" "Predictions v1"

Write-Host "`nStep 4: Deleting TEMP folders..." -ForegroundColor Cyan
$deletedSize += Remove-SafeFolder "TEMP" "TEMP folder"
$deletedSize += Remove-SafeFolder "TEMP2" "TEMP2 folder"

Write-Host "`nStep 5: Cleaning Python cache..." -ForegroundColor Cyan
$cacheCount = 0
Get-ChildItem -Recurse -Directory -Force | Where-Object {$_.Name -eq '__pycache__'} | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force
    $cacheCount++
}
Write-Host "[DELETED] $cacheCount __pycache__ folders" -ForegroundColor Green

Write-Host "`nStep 6: Deleting old experiment checkpoints..." -ForegroundColor Cyan
$deletedSize += Remove-SafeFolder "Unet_v0.4\checkpoints" "Unet v0.4 checkpoints"
$deletedSize += Remove-SafeFolder "newUnet\checkpoints" "newUnet checkpoints"
$deletedSize += Remove-SafeFolder "newUnet2\checkpoints" "newUnet2 checkpoints (if exists)"

Write-Host "`nStep 7: Deleting old experimental folders (optional)..." -ForegroundColor Cyan
$response = Read-Host "Delete old experimental folders (baseline_2d_ensemble, nnUnet2)? (y/N)"
if ($response -eq 'y') {
    $deletedSize += Remove-SafeFolder "baseline_2d_ensemble" "Baseline 2D ensemble"
    $deletedSize += Remove-SafeFolder "nnUnet2" "nnUnet2 (threshold opt)"
}

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete!" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Get final size
$finalSize = (Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB

Write-Host ""
Write-Host "Initial size:  $([math]::Round($initialSize, 2)) GB" -ForegroundColor Yellow
Write-Host "Final size:    $([math]::Round($finalSize, 2)) GB" -ForegroundColor Green
Write-Host "Saved:         $([math]::Round($deletedSize/1024, 2)) GB" -ForegroundColor Green
Write-Host ""

if ($finalSize -lt 5) {
    Write-Host "Ready for GitHub upload!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Review .gitignore file" -ForegroundColor White
    Write-Host "2. git init (if not done)" -ForegroundColor White
    Write-Host "3. git add ." -ForegroundColor White
    Write-Host "4. git commit -m 'Initial commit'" -ForegroundColor White
    Write-Host "5. git remote add origin YOUR_REPO_URL" -ForegroundColor White
    Write-Host "6. git push -u origin main" -ForegroundColor White
} else {
    Write-Host "Warning: Size still large ($([math]::Round($finalSize, 2)) GB)" -ForegroundColor Yellow
    Write-Host "Consider:" -ForegroundColor Yellow
    Write-Host "- Using Git LFS for checkpoint files" -ForegroundColor White
    Write-Host "- Uploading checkpoints to external storage" -ForegroundColor White
    Write-Host "- Further cleanup of old experiments" -ForegroundColor White
}

Write-Host ""
Write-Host "See GITHUB_UPLOAD_GUIDE.md for detailed instructions" -ForegroundColor Cyan
