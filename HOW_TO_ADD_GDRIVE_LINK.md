# 如何添加Google Drive連結

## 步驟1: 上傳Checkpoint到Google Drive

1. 找到檔案：`nnUnet/7CH_v2_Archive/checkpoint_best.pth`（256 MB）
2. 上傳到您的Google Drive
3. 右鍵點擊檔案 → "取得連結"
4. 設定為 "知道連結的任何人都可以檢視"
5. 複製連結

## 步驟2: 更新README.md

在GitHub網頁上直接編輯：

找到這一行（第34行左右）：
```markdown
- **Google Drive**: [Download checkpoint_best.pth](在此處貼上您的Google Drive連結)
```

將 `在此處貼上您的Google Drive連結` 替換為實際的Google Drive連結：
```markdown
- **Google Drive**: [Download checkpoint_best.pth](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)
```

## 步驟3: 同時更新CHECKPOINT_DOWNLOAD.md

檔案位置：`models/7CH_v2_Archive/CHECKPOINT_DOWNLOAD.md`

找到：
```markdown
**Google Drive**: [您需要在此處添加Google Drive連結]
```

替換為：
```markdown
**Google Drive**: [Download Here](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)
```

## 步驟4: 生成MD5 (可選)

在本地執行：
```powershell
certutil -hashfile "nnUnet\7CH_v2_Archive\checkpoint_best.pth" MD5
```

將MD5值添加到CHECKPOINT_DOWNLOAD.md中，方便使用者驗證檔案完整性。

---

**注意**: 
- ✅ 是的，使用 `nnUnet/7CH_v2_Archive/checkpoint_best.pth` 這個檔案
- ✅ Google Drive連結應該是完整的sharing link
- ✅ 確保設定為"任何人都可以檢視"

兩個檔案都要更新：
1. README.md (主頁)
2. models/7CH_v2_Archive/CHECKPOINT_DOWNLOAD.md (詳細說明)
