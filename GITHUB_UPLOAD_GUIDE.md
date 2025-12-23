# GitHub Upload Preparation Guide

## Current Status
- **Total Size**: 41.11 GB ❌ TOO LARGE for GitHub
- **GitHub Limit**: 100 MB per file, ~5 GB per repo recommended

## What to Keep vs Delete

### ✅ KEEP (Upload to GitHub)

**Code & Scripts**:
- All `.py` files
- `requirements.txt`
- Config files (`config.py`, etc.)

**Documentation**:
- `README.md`
- `PROJECT_REPORT.md`
- All `*_REPORT.md`, `*_ANALYSIS.md`
- Archive READMEs and docs

**Critical Checkpoints** (Keep only best models):
- `nnUnet/4CH_Model_Archive/checkpoint_bestv0.1.pth`
- `nnUnet/7CH_v1_Archive/checkpoint_best.pth`
- `nnUnet/7CH_v2_Archive/checkpoint_best.pth`

**Visualizations**:
- `*.png` (training curves, evaluation results)
- Small demo images

---

### ❌ DELETE (DO NOT Upload)

**Data Folder** (~40 GB):
```
data/wmh/
```
- 原始醫學影像數據
- 不應公開分享（隱私、版權）
- **建議**: 提供數據集來源鏈接即可

**nnU-Net Generated** (~500 MB):
```
nnUNet_raw/
nnUNet_preprocessed/
nnUNet_results/*/fold_*/  (except best checkpoints)
```
- 自動生成的預處理數據
- 可從原始數據重新生成

**Temporary Files**:
```
temp_v2_eval/
batch_eval*/
predictions_*/
__pycache__/
*.pyc
checkpoints/ (in Unet_v0.4, newUnet, etc.)
```

**Old/Experimental Folders** (Optional Delete):
```
TEMP/
TEMP2/
newUnet2/  (如果是未完成實驗)
baseline_2d_ensemble/
nnUnet2/
```

---

## 📋 Cleanup Checklist

### Option 1: Manual Cleanup (Recommended for First Time)

1. **Delete data folder**:
   ```powershell
   Remove-Item data -Recurse -Force
   ```

2. **Delete nnU-Net generated**:
   ```powershell
   Remove-Item nnUnet/nnUNet_raw -Recurse -Force
   Remove-Item nnUnet/nnUNet_preprocessed -Recurse -Force
   ```

3. **Delete temp folders**:
   ```powershell
   Remove-Item temp_v2_eval, TEMP, TEMP2 -Recurse -Force -ErrorAction SilentlyContinue
   ```

4. **Clean Python cache**:
   ```powershell
   Get-ChildItem -Recurse -Directory -Force -Filter __pycache__ | Remove-Item -Recurse -Force
   ```

5. **Delete old checkpoints** (keep only Archive):
   ```powershell
   Remove-Item Unet_v0.4/checkpoints -Recurse -Force -ErrorAction SilentlyContinue
   Remove-Item newUnet/checkpoints -Recurse -Force -ErrorAction SilentlyContinue
   ```

### Option 2: Automated Script

Run the cleanup script:
```powershell
.\cleanup_for_github.ps1
```

---

## Expected Size After Cleanup

| Component | Before | After |
|-----------|--------|-------|
| Data folder | ~40 GB | 0 GB ✅ |
| nnU-Net generated | ~500 MB | 0 MB ✅ |
| Checkpoints | ~1 GB | ~800 MB ⚠️ |
| Code & Docs | ~10 MB | ~10 MB ✅ |
| **Total** | **41.11 GB** | **~1 GB** ✅ |

**Note**: 如果checkpoint仍太大，考慮：
- 使用Git LFS for checkpoint files
- 或將checkpoints上傳到Google Drive/Hugging Face

---

## 🚀 Git Upload Steps

After cleanup:

```bash
# 1. Initialize git (if not done)
git init

# 2. Add .gitignore
git add .gitignore

# 3. Add all files (respecting .gitignore)
git add .

# 4. Check what will be committed
git status

# 5. Commit
git commit -m "Initial commit: WMH Segmentation Project"

# 6. Add remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 7. Push
git push -u origin main
```

---

## 📝 README Updates Needed

Before upload, update README.md:

1. **Data section**: Add link to WMH Challenge dataset
2. **Quick start**: Mention data download requirement
3. **Checkpoints**: Provide download links if using external storage
4. **Environment**: Update requirements.txt if needed

---

## ⚠️ Important Notes

1. **Data Privacy**: Never upload patient data to public GitHub
2. **Checkpoint Size**: If >100 MB, use Git LFS or external hosting
3. **Credentials**: Remove any API keys, passwords
4. **Paths**: Change hardcoded paths like `d:\VSCode\AIOT_E1` to relative

---

**Created**: 2025-12-23  
**Status**: Ready for cleanup
