"""
Loss functions for WMH segmentation
Focal Tversky Loss - optimized for high sensitivity (avoiding false negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    
    Tversky Index controls FP/FN trade-off via alpha/beta
    Focal weight focuses on hard samples via gamma
    
    Perfect for:
    - Extreme class imbalance (WMH ~0.5% of pixels)
    - High sensitivity requirement (beta > alpha)
    """
    
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        """
        Args:
            alpha: Weight for false positives (lower = less penalty)
            beta: Weight for false negatives (higher = more penalty for missing WMH)
            gamma: Focal parameter (higher = focus more on hard samples)
            smooth: Smoothing factor to avoid division by zero
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred, target, brain_mask=None, valid_mask=None):
        """
        Args:
            pred: (B, 1, H, W) - raw logits
            target: (B, 1, H, W) or (B, H, W) - binary mask (0 or 1)
            brain_mask: Optional (B, 1, H, W) - brain tissue mask
            valid_mask: Optional (B, 1, H, W) - valid region mask (1=compute, 0=don't care)
        """
        # Ensure target has channel dimension
        if target.ndim == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Resize pred to match target if needed
        if pred.shape != target.shape:
            target_size = (target.shape[2], target.shape[3])
            pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
        
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # Apply brain mask if provided
        if brain_mask is not None:
            if brain_mask.ndim == 3:
                brain_mask = brain_mask.unsqueeze(1)
            if brain_mask.shape != pred.shape:
                mask_size = (pred.shape[2], pred.shape[3])
                brain_mask = F.interpolate(brain_mask, size=mask_size, mode='nearest')
            # Apply mask to both pred and target
            pred = pred * brain_mask
            target = target * brain_mask
        
        # 🔧 Apply valid_mask to exclude don't care regions
        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.shape != pred.shape:
                mask_size = (pred.shape[2], pred.shape[3])
                valid_mask = F.interpolate(valid_mask, size=mask_size, mode='nearest')
            # Exclude don't care regions
            pred = pred * valid_mask
            target = target * valid_mask
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        # 🔧 NUMERICAL STABILITY: Optimized for WMH small target segmentation
        smooth = 0.01  # Balanced for WMH (TP typically 50-500 pixels)
        
        # Calculate denominator with protection
        denominator = TP + self.alpha * FP + self.beta * FN + smooth
        denominator = torch.clamp(denominator, min=1e-3)  # Prevent near-zero
        
        # Tversky index with clamping
        tversky = (TP + smooth) / denominator
        tversky = torch.clamp(tversky, min=0.0, max=1.0)  # Ensure [0,1] range
        
        # 🔧 PREVENT OVERFLOW: Limit base before exponentiation
        base = 1.0 - tversky
        base = torch.clamp(base, min=0.0, max=10.0)  # Prevent extreme values
        
        # Focal Tversky loss
        focal_tversky = torch.pow(base, self.gamma)
        
        # 🔧 FINAL CHECK: Replace any NaN/Inf with safe value
        if torch.isnan(focal_tversky).any() or torch.isinf(focal_tversky).any():
            print("⚠️  WARNING: NaN/Inf detected in loss, using fallback value")
            focal_tversky = torch.tensor(1.0, device=focal_tversky.device)
        
        return focal_tversky


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal Tversky + BCE
    """
    
    def __init__(self, tversky_weight=0.7, bce_weight=0.3, 
                 alpha=0.3, beta=0.7, gamma=1.33):
        super(CombinedLoss, self).__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.focal_tversky = FocalTverskyLoss(alpha, beta, gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([200.0]))  # Class imbalance
        
    def forward(self, pred, target):
        loss_tversky = self.focal_tversky(pred, target)
        loss_bce = self.bce(pred, target)
        return self.tversky_weight * loss_tversky + self.bce_weight * loss_bce


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        pred: (B, 1, H, W) - raw logits or probabilities
        target: (B, 1, H, W) or (B, H, W) - binary mask
    """
    # Ensure target has channel dimension
    if target.ndim == 3:
        target = target.unsqueeze(1)
    
    # Match shapes
    if pred.shape != target.shape:
        target_size = (target.shape[2], target.shape[3])
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
    
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def sensitivity(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate sensitivity (recall) - critical for WMH detection
    
    Sensitivity = TP / (TP + FN)
    """
    # Ensure target has channel dimension
    if target.ndim == 3:
        target = target.unsqueeze(1)
    
    # Match shapes
    if pred.shape != target.shape:
        target_size = (target.shape[2], target.shape[3])
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
    
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()
    target = target.float()
    
    TP = (pred * target).sum()
    FN = ((1 - pred) * target).sum()
    
    sens = (TP + smooth) / (TP + FN + smooth)
    return sens


def precision(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate precision
    
    Precision = TP / (TP + FP)
    """
    # Ensure target has channel dimension
    if target.ndim == 3:
        target = target.unsqueeze(1)
    
    # Match shapes
    if pred.shape != target.shape:
        target_size = (target.shape[2], target.shape[3])
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
    
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()
    target = target.float()
    
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    
    prec = (TP + smooth) / (TP + FP + smooth)
    return prec
