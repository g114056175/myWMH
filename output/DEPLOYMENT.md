# Demo 部署說明

## 📁 需要上傳的文件

只需要上傳以下兩個項目：

```
output/
├── demo_viewer.html       ← HTML 查看器 (15.5 KB)
└── demo_cases/           ← 圖片資料夾 (531 張圖片)
    ├── Amsterdam_GE3T_119/
    ├── Amsterdam_Philips_VU .PETMR_01__160/
    ├── Amsterdam_Philips_VU .PETMR_01__169/
    ├── Singapore_72/
    └── Utrecht_9/
```

## 🚀 GitHub Pages 部署步驟

### 1. 上傳文件
將 `output/` 資料夾的內容上傳到 GitHub repo 的根目錄或 `docs/` 資料夾

### 2. 啟用 GitHub Pages
1. 進入 repo 的 Settings → Pages
2. Source 選擇 `main` branch
3. 選擇根目錄 `/` 或 `/docs` 資料夾
4. 保存

### 3. 訪問
訪問 `https://your-username.github.io/your-repo/demo_viewer.html`

## 💻 本地測試

直接雙擊 `demo_viewer.html` 在瀏覽器中打開即可

## 🎮 使用方式

1. **選擇病例** - 點擊病例按鈕
2. **查看切片** - 滾輪或方向鍵切換
3. **觀察結果** - 同時查看 FLAIR、GT、預測

## ⚠️ 注意事項

- **保持目錄結構** - `demo_viewer.html` 和 `demo_cases/` 必須在同一層
- **文件路徑** - 不要重新命名資料夾
- **瀏覽器** - 推薦使用 Chrome, Firefox, Edge

## 📊 包含的數據

- 5 個病例
- 177 個切片（Amsterdam 3個×27 + Singapore/Utrecht 2個×48）
- 531 張圖片（177切片 × 3視圖）
