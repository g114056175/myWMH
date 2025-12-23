"""
損失函數和評估指標
==================

設計巧思:
1. Combo Loss: Dice + BCE 結合，處理類別不平衡
2. Dice Coefficient: 醫學影像分割標準指標
3. IoU: 交並比
4. Hausdorff Distance: 邊界精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff


# ==============================================================================
# Dice Loss
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Segmentation
    
    設計理由:
    1. 對類別不平衡魯棒（WMH << 背景）
    2. 直接優化 Dice Coefficient
    3. 平滑項防止除零
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - 預測（logits 或 sigmoid 後）
            target: (B, 1, H, W) - Ground truth (0 或 1)
        """
        # Sigmoid 激活（如果尚未激活）
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Loss = 1 - Dice
        return 1 - dice


# ==============================================================================
# Combo Loss
# ==============================================================================

class ComboLoss(nn.Module):
    """
    組合損失：Dice Loss + Weighted BCE Loss
    
    Args:
        dice_weight: Dice Loss 的權重
        bce_weight: BCE Loss 的權重
        pos_weight: BCE中正樣本的權重（用於極不平衡數據）
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=1.0):
        super(ComboLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - 模型輸出 logits
            target: (B, 1, H, W) - Ground truth (0或1)
        """
        # Dice loss on sigmoid probabilities
        dice = self.dice_loss(torch.sigmoid(pred), target)
        
        # BCE loss on logits with pos_weight on same device
        pos_weight_tensor = torch.tensor([self.pos_weight], device=pred.device)
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight_tensor
        )
        
        return self.dice_weight * dice + self.bce_weight * bce


# ==============================================================================
# Tversky Loss (針對類別不平衡優化)
# ==============================================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss for Imbalanced Segmentation
    
    設計理由:
    1. 精確控制 False Negative (FN) 和 False Positive (FP) 的權重
    2. 對類別不平衡問題天然魯棒
    3. 通過調整 alpha/beta 平衡 Sensitivity/Specificity
    
    數學公式:
        Tversky Index = TP / (TP + α*FN + β*FP)
        
    當 α=β=0.5 時，等同於 Dice Loss
    當 α>β 時，更注重減少漏檢（提高 Sensitivity）
    當 α<β 時，更注重減少誤報（提高 Specificity）
    
    對 WMH 分割推薦: α=0.7, β=0.3
    - 漏檢懲罰是誤報懲罰的 2.3 倍
    - 適合醫學影像（漏診比誤診更嚴重）
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Args:
            alpha: False Negative 權重（漏檢懲罰）
            beta: False Positive 權重（誤報懲罰）
            smooth: 平滑項防止除零
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - 預測（logits 或 sigmoid 後）
            target: (B, 1, H, W) - Ground truth (0 或 1)
        
        Returns:
            Tversky Loss (0~1)
        """
        # Sigmoid 激活（如果尚未激活）
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 計算 TP, FP, FN
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        
        # Loss = 1 - Tversky Index
        return 1 - tversky_index


# ==============================================================================
# Focal Tversky Loss (WMH Challenge Champion Strategy)
# ==============================================================================

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - 多個WMH Challenge獲獎團隊使用
    
    結合：
    1. Tversky Loss: 非對稱FN/FP懲罰，直接優化Sensitivity
    2. Focal機制: 關注難檢測的小病變
    
    參數：
        alpha: FP的權重（越小越容忍誤報）
        beta: FN的權重（越大越懲罰漏檢）
        gamma: Focal參數（越大越關注難例）
        
    推薦配置（高Sensitivity）：
        alpha=0.2, beta=0.8, gamma=1.5
    """
    def __init__(self, alpha=0.2, beta=0.8, gamma=1.5, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - 模型輸出 logits
            target: (B, 1, H, W) - Ground truth
        """
        # Sigmoid激活
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 計算TP, FP, FN
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        # Tversky Index
        # 當beta > alpha時，更重視減少FN（提高Sensitivity）
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        # Focal component
        # gamma越大，越關注Tversky Index低的難樣本
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        
        return focal_tversky


# ==============================================================================
# 評估指標
# ==============================================================================

def dice_coefficient(pred, target, threshold=0.5):
    """
    計算 Dice Coefficient
    
    Args:
        pred: (B, 1, H, W) - 預測概率
        target: (B, 1, H, W) - Ground truth
        threshold: 二值化閾值
    
    Returns:
        dice score (0~1)
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1) / (pred.sum() + target.sum() + 1)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5):
    """
    計算 IoU (Intersection over Union)
    
    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
    
    Returns:
        IoU score (0~1)
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1) / (union + 1)
    return iou.item()


def sensitivity(pred, target, threshold=0.5):
    """
    計算 Sensitivity (Recall, True Positive Rate)
    靈敏度 = TP / (TP + FN)
    """
    pred = (pred > threshold).float().view(-1)
    target = target.float().view(-1)
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    return (tp / (tp + fn + 1e-8)).item()


def specificity(pred, target, threshold=0.5):
    """
    計算 Specificity (True Negative Rate)
    特異度 = TN / (TN + FP)
    """
    pred = (pred > threshold).float().view(-1)
    target = target.float().view(-1)
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tn / (tn + fp + 1e-8)).item()


def hausdorff_distance_95(pred, target, threshold=0.5, voxel_spacing=(1, 1)):
    """
    計算 95th percentile Hausdorff Distance
    
    Args:
        pred: (H, W) numpy array
        target: (H, W) numpy array
        threshold: 二值化閾值
        voxel_spacing: 體素間距 (默認 1x1 mm)
    
    Returns:
        HD95 (mm)
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    
    # 獲取邊界點
    pred_points = np.argwhere(pred_binary)
    target_points = np.argwhere(target_binary)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return np.inf
    
    # 計算雙向 Hausdorff distance
    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]
    
    # 取 95th percentile
    hd95 = np.percentile([hd1, hd2], 95)
    
    # 考慮體素間距
    hd95 *= np.mean(voxel_spacing)
    
    return hd95


class MetricsTracker:
    """
    訓練過程中追蹤指標
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.sensitivities = []
        self.specificities = []
        
    def update(self, pred, target):
        """
        更新指標
        
        Args:
            pred: (B, 1, H, W) - 預測概率
            target: (B, 1, H, W) - Ground truth
        """
        pred = torch.sigmoid(pred)
        
        self.dice_scores.append(dice_coefficient(pred, target))
        self.iou_scores.append(iou_score(pred, target))
        self.sensitivities.append(sensitivity(pred, target))
        self.specificities.append(specificity(pred, target))
    
    def get_average(self):
        """獲取平均指標"""
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'sensitivity': np.mean(self.sensitivities),
            'specificity': np.mean(self.specificities)
        }
    
    def print_summary(self, prefix=""):
        """打印統計摘要"""
        avg = self.get_average()
        print(f"{prefix}Dice: {avg['dice']:.4f} | "
              f"IoU: {avg['iou']:.4f} | "
              f"Sens: {avg['sensitivity']:.4f} | "
              f"Spec: {avg['specificity']:.4f}")


if __name__ == "__main__":
    # 測試損失函數
    pred = torch.randn(2, 1, 240, 240) # logits
    target = torch.randint(0, 2, (2, 1, 240, 240)).float()
    
    criterion = ComboLoss()
    loss = criterion(pred, target)
    print(f"Combo Loss: {loss.item():.4f}")
    
    # 測試指標
    pred_prob = torch.sigmoid(pred)
    dice = dice_coefficient(pred_prob, target)
    iou = iou_score(pred_prob, target)
    print(f"Dice: {dice:.4f}, IoU: {iou:.4f}")
    """
    Focal Loss?於??類別不平?
    專注?困????
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_term * BCE_loss
        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """結?Focal Loss?Dice Loss"""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
