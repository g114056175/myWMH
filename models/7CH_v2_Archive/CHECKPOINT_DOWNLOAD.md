# Model Checkpoint - Download Instructions

## ⚠️ Checkpoint Not Included in Repository

Due to GitHub file size limitations (100MB max), the model checkpoint is hosted externally.

## 📥 Download

**File**: `checkpoint_best.pth`  
**Size**: 256 MB  
**Download Link**: **[您需要在此處添加Google Drive或Hugging Face連結]**

### Option 1: Google Drive
```
[將checkpoint上傳到Google Drive並分享連結]
```

### Option 2: Hugging Face Model Hub
```
[或上傳到Hugging Face: https://huggingface.co/]
```

## 📦 Installation

After downloading, place the checkpoint here:
```
models/7CH_v2_Archive/checkpoint_best.pth
```

Verify installation:
```bash
ls -lh models/7CH_v2_Archive/checkpoint_best.pth
# Should show: ~256 MB file
```

## ✅ File Integrity

**MD5 Checksum**: [您需要生成MD5]
```bash
# Windows
certutil -hashfile models/7CH_v2_Archive/checkpoint_best.pth MD5

# Linux/Mac
md5sum models/7CH_v2_Archive/checkpoint_best.pth
```

## 🔧 Quick Test

After placing checkpoint:
```bash
cd models
python -c "import torch; ckpt=torch.load('7CH_v2_Archive/checkpoint_best.pth', map_location='cpu'); print('Checkpoint loaded successfully!')"
```

---

## 📊 Checkpoint Details

- **Model**: 7-Channel nnU-Net v2
- **Training Epochs**: 102
- **Best Validation Dice**: 0.8763 (Epoch 70)
- **Test Performance**: Overall Dice 0.8575
- **Framework**: nnU-Net v2
- **Architecture**: PlainConvUNet (2D, 7 input channels)

---

## ❓ Troubleshooting

### File Not Found
Ensure the file is placed at:
```
AIOT_E1/models/7CH_v2_Archive/checkpoint_best.pth
```

### Download Interrupted
Use a download manager for large files:
- **Windows**: Free Download Manager
- **Linux**: wget or curl
- **Mac**: curl or Safari

### Size Mismatch
If file size ≠ 256 MB, re-download. File may be corrupted.

---

For usage instructions, see [README.md](README.md) in the archive folder.
