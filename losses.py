"""
損失函數庫 - 支持多種損失函數用於ensemble
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss - 適合分割任務"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss - 處理類別不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - 可調節FP/FN權重"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # False Negative權重
        self.beta = beta    # False Positive權重
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        
        return 1 - tversky


class ComboLoss(nn.Module):
    """組合損失 - BCE + Dice"""
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + self.beta * self.dice(pred, target)


def get_loss_function(loss_name):
    """
    獲取損失函數
    
    Args:
        loss_name: 'bce', 'dice', 'focal', 'tversky', 'combo'
    """
    loss_map = {
        'bce': nn.BCEWithLogitsLoss(),
        'dice': DiceLoss(),
        'focal': FocalLoss(alpha=0.25, gamma=2.0),
        'tversky': TverskyLoss(alpha=0.7, beta=0.3),
        'combo': ComboLoss(alpha=0.5, beta=0.5)
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name]


if __name__ == "__main__":
    # 測試損失函數
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    for loss_name in ['bce', 'dice', 'focal', 'tversky', 'combo']:
        loss_fn = get_loss_function(loss_name)
        loss_value = loss_fn(pred, target)
        print(f"{loss_name:10s}: {loss_value.item():.4f}")
