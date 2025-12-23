"""
2D模型訓練腳本 - PGS風格
專注於FLAIR單通道輸入，強化資料增強
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse

from model_2d import UNet2D, count_parameters
from dataset_2d import WMHDataset2D
from augmentations import get_training_augmentation
from losses import get_loss_function


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """訓練一個epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            dice = calculate_dice(preds, masks)
        
        total_loss += loss.item()
        total_dice += dice
        
        pbar.set_postfix({'loss': loss.item(), 'dice': dice})
    
    return total_loss / len(dataloader), total_dice / len(dataloader)


def validate(model, dataloader, criterion, device):
    """驗證"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_sensitivity = 0
    total_precision = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = torch.sigmoid(outputs) > 0.5
            dice = calculate_dice(preds, masks)
            sensitivity = calculate_sensitivity(preds, masks)
            precision = calculate_precision(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice
            total_sensitivity += sensitivity
            total_precision += precision
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'dice': total_dice / n,
        'sensitivity': total_sensitivity / n,
        'precision': total_precision / n
    }


def calculate_dice(pred, target):
    """計算Dice係數"""
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()


def calculate_sensitivity(pred, target):
    """計算Sensitivity (Recall)"""
    pred = pred.float()
    target = target.float()
    TP = (pred * target).sum()
    FN = (target * (1 - pred)).sum()
    sensitivity = TP / (TP + FN + 1e-8)
    return sensitivity.item()


def calculate_precision(pred, target):
    """計算Precision"""
    pred = pred.float()
    target = target.float()
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    precision = TP / (TP + FP + 1e-8)
    return precision.item()


def main(args):
    # 設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    print(f"模型ID: {args.model_id}")
    print(f"損失函數: {args.loss}")
    
    # 創建輸出目錄
    checkpoint_dir = Path(f'checkpoints_2d_model{args.model_id}')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # 資料增強策略
    train_transform = get_training_augmentation(args.aug_strength)
    
    # 數據集
    print("\n載入數據集...")
    train_dataset = WMHDataset2D(
        data_dir='data/wmh',
        split='training',
        empty_ratio=args.empty_ratio,
        transform=train_transform
    )
    
    val_dataset = WMHDataset2D(
        data_dir='data/wmh',
        split='training',
        empty_ratio=1.0,  # 驗證集保留所有切片
        transform=None
    )
    
    # 簡單分割：80%訓練，20%驗證
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"訓練集: {len(train_dataset)} 切片")
    print(f"驗證集: {len(val_dataset)} 切片")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows設為0
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 模型
    print("\n創建模型...")
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        dropout_rate=args.dropout
    ).to(device)
    
    print(f"參數量: {count_parameters(model):,}")
    
    # 損失函數
    criterion = get_loss_function(args.loss)
    
    # 優化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 學習率調度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # 訓練循環
    print("\n開始訓練...")
    best_dice = 0
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # 訓練
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 驗證
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 學習率調整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice'],
            'val_sensitivity': val_metrics['sensitivity'],
            'val_precision': val_metrics['precision'],
            'lr': current_lr
        }
        history.append(epoch_result)
        
        # 打印結果
        print(f"訓練 - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"驗證 - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"Sens: {val_metrics['sensitivity']:.4f}, Prec: {val_metrics['precision']:.4f}")
        print(f"學習率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }, checkpoint_dir / 'best_model.pth')
            
            print(f"✓ 最佳模型已保存 (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # 保存訓練歷史
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"最佳驗證Dice: {best_dice:.4f}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='訓練2D WMH分割模型')
    
    # 模型參數
    parser.add_argument('--model_id', type=int, required=True, help='模型ID (1-5)')
    parser.add_argument('--loss', type=str, default='bce', 
                       choices=['bce', 'dice', 'focal', 'tversky', 'combo'],
                       help='損失函數類型')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16, help='Batch大小')
    parser.add_argument('--epochs', type=int, default=100, help='訓練epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='權重衰減')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping耐心值')
    
    # 數據參數
    parser.add_argument('--empty_ratio', type=float, default=0.1, help='空白切片比例')
    parser.add_argument('--aug_strength', type=str, default='medium',
                       choices=['light', 'medium', 'strong', 'champion'],
                       help='資料增強強度')
    
    args = parser.parse_args()
    
    main(args)
