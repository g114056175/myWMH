"""
TTA (Test-Time Augmentation) 推論模組
實現4-way TTA: 原圖, 垂直翻轉, 水平翻轉, 雙翻轉
"""
import torch
import torch.nn.functional as F


def tta_predict(model, image, device='cuda'):
    """
    使用TTA對單個圖像進行預測
    
    Args:
        model: 訓練好的模型
        image: (C, H, W) tensor
        device: 計算設備
    
    Returns:
        averaged prediction: (C, H, W) tensor
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 確保是batch格式
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        
        image = image.to(device)
        
        # 1. 原圖
        pred = torch.sigmoid(model(image))
        predictions.append(pred)
        
        # 2. 垂直翻轉
        img_vflip = torch.flip(image, dims=[2])  # flip height
        pred_vflip = torch.sigmoid(model(img_vflip))
        pred_vflip = torch.flip(pred_vflip, dims=[2])  # 還原
        predictions.append(pred_vflip)
        
        # 3. 水平翻轉
        img_hflip = torch.flip(image, dims=[3])  # flip width
        pred_hflip = torch.sigmoid(model(img_hflip))
        pred_hflip = torch.flip(pred_hflip, dims=[3])  # 還原
        predictions.append(pred_hflip)
        
        # 4. 雙翻轉
        img_both = torch.flip(image, dims=[2, 3])
        pred_both = torch.sigmoid(model(img_both))
        pred_both = torch.flip(pred_both, dims=[2, 3])  # 還原
        predictions.append(pred_both)
    
    # 平均4個預測
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    
    return avg_pred.squeeze(0)  # 移除batch維度


def ensemble_tta_predict(models, image, device='cuda'):
    """
    使用多個模型 + TTA進行ensemble預測
    
    Args:
        models: list of trained models
        image: (C, H, W) tensor
        device: 計算設備
    
    Returns:
       ensembled prediction: (C, H, W) tensor
    """
    all_predictions = []
    
    for model in models:
        model.to(device)
        model.eval()
        
        # 每個模型使用TTA
        pred = tta_predict(model, image, device)
        all_predictions.append(pred)
    
    # 對所有模型的預測取平均
    ensemble_pred = torch.mean(torch.stack(all_predictions), dim=0)
    
    return ensemble_pred


if __name__ == "__main__":
    from model_2d import UNet2D
    
    # 測試TTA
    print("測試TTA功能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D().to(device)
    model.eval()
    
    # 測試圖像
    test_image = torch.randn(1, 256, 256)
    
    # 單一預測
    with torch.no_grad():
        single_pred = torch.sigmoid(model(test_image.unsqueeze(0).to(device)))
    
    # TTA預測
    tta_pred = tta_predict(model, test_image, device)
    
    print(f"單一預測shape: {single_pred.squeeze().shape}")
    print(f"TTA預測shape: {tta_pred.shape}")
    print(f"預測範圍: [{tta_pred.min():.4f}, {tta_pred.max():.4f}]")
    
    # 測試Ensemble
    print("\n測試Ensemble功能...")
    models = [UNet2D().to(device) for _ in range(3)]
    ensemble_pred = ensemble_tta_predict(models, test_image, device)
    
    print(f"Ensemble預測shape: {ensemble_pred.shape}")
    print(f"預測範圍: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
    
    print("\n✓ TTA和Ensemble測試通過！")
