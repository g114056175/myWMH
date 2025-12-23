"""
Training script for WMH segmentation with Attention U-Net
Supports: 5-channel input, Focal Tversky Loss, AMP, Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time

from model import AttentionUNet, count_parameters
from dataset import WMHDataset, get_train_transform, get_val_transform
from losses import FocalTverskyLoss, dice_coefficient, sensitivity, precision
import config


class EarlyStopping:
    """Early stopping to avoid overfitting"""
    
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with Deep Supervision"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_sens = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=config.USE_AMP):
            # Deep Supervision: model returns [aux1, aux2, aux3, final]
            outputs = model(images, deep_supervision=True)
            
            # Multi-output loss
            # V0.7: Enhanced Deep Supervision weights for stronger regularization
            loss = 0.0
            weights = [0.5, 0.7, 0.9, 1.0]  # Give aux outputs more pressure
            for i, (out, w) in enumerate(zip(outputs, weights)):
                loss += w * criterion(out, masks)
            
            # Normalize by sum of weights
            loss = loss / sum(weights)
        
        # Backpropagation
        scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        
        # Check for NaN
        has_nan = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan = True
                break
        
        if has_nan:
            print(f"⚠️  WARNING: NaN detected in gradients. Skipping this batch.")
            optimizer.zero_grad()
            scaler.update()
            continue
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics (use final output only)
        with torch.no_grad():
            final_output = outputs[-1]
            dice = dice_coefficient(final_output, masks)
            sens = sensitivity(final_output, masks)
        
        running_loss += loss.item()
        running_dice += dice.item()
        running_sens += sens.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}',
            'sens': f'{sens.item():.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_sens = running_sens / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_sens


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_sens = 0.0
    running_prec = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass (inference mode - single output)
            outputs = model(images, deep_supervision=False)
            loss = criterion(outputs, masks)
            
            # Metrics
            dice = dice_coefficient(outputs, masks)
            sens = sensitivity(outputs, masks)
            prec = precision(outputs, masks)
            
            running_loss += loss.item()
            running_dice += dice.item()
            running_sens += sens.item()
            running_prec += prec.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}',
                'sens': f'{sens.item():.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_sens = running_sens / len(dataloader)
    epoch_prec = running_prec / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_sens, epoch_prec


def main():
    print("="*60)
    print("WMH Segmentation Training")
    print("="*60)
    
    # Create directories
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    print("\nLoading dataset...")
    full_dataset = WMHDataset(
        data_root=config.TRAIN_DIR,
        target_size=config.TARGET_SIZE,
        transform=None,  # Will add later for train/val separately
        clahe_clip_limit=config.CLAHE_CLIP_LIMIT,
        highpass_sigma=config.HIGHPASS_SIGMA
    )
    
    # ===== FIX DATA LEAKAGE: Patient-level split =====
    # 按volume（病例）拆分，而非按slice拆分
    print("\n🔧 Applying patient-level split (fixing data leakage)...")
    
    num_volumes = len(full_dataset.samples)  # 60 volumes
    train_vol_size = int(0.8 * num_volumes)  # 48 volumes
    val_vol_size = num_volumes - train_vol_size  # 12 volumes
    
    # 随机打乱volumes
    volume_indices = list(range(num_volumes))
    np.random.seed(42)  # 固定seed保证可复现
    np.random.shuffle(volume_indices)
    
    train_volumes = set(volume_indices[:train_vol_size])
    val_volumes = set(volume_indices[train_vol_size:])
    
    # 根据volume归属确定每个slice
    train_indices = []
    val_indices = []
    
    for i, (vol_idx, slice_idx) in enumerate(full_dataset.slice_indices):
        if vol_idx in train_volumes:
            train_indices.append(i)
        else:
            val_indices.append(i)
    
    # 创建Subset
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 验证无数据泄漏
    train_vols_check = set([full_dataset.slice_indices[i][0] for i in train_indices])
    val_vols_check = set([full_dataset.slice_indices[i][0] for i in val_indices])
    assert len(train_vols_check & val_vols_check) == 0, "Data leakage detected!"
    
    print(f"✓ Patient-level split complete:")
    print(f"  Train: {len(train_volumes)} volumes ({len(train_indices)} slices)")
    print(f"  Val:   {len(val_volumes)} volumes ({len(val_indices)} slices)")
    print(f"  No overlap: {len(train_vols_check & val_vols_check)} volumes")
    
    # Add transforms
    train_dataset.dataset.transform = get_train_transform(
        # V0.4: No horizontal flip
        aug_p_vertical=config.AUG_VERTICAL_FLIP,
        aug_p_rotate=config.AUG_ROTATE_PROB,
        aug_p_brightness=config.AUG_BRIGHTNESS_CONTRAST,
        aug_p_gamma=config.AUG_GAMMA_PROB,
        aug_p_elastic=config.AUG_ELASTIC_PROB,
        aug_p_grid=0.3
    )
    val_dataset.dataset.transform = get_val_transform()
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Model
    print("\nBuilding model...")
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        depth=config.DEPTH,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function
    criterion = FocalTverskyLoss(
        alpha=config.TVERSKY_ALPHA,
        beta=config.TVERSKY_BETA,
        gamma=config.FOCAL_GAMMA
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler (CosineAnnealingWarmRestarts)
    # Using longer T_0 to keep LR higher for more epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.LR_COSINE_T0,  # Use config value (100)
        T_mult=1,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        min_delta=config.EARLY_STOP_MIN_DELTA
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_sens': [],
        'val_loss': [],
        'val_dice': [],
        'val_sens': [],
        'val_prec': [],
        'lr': []
    }
    
    best_val_dice = 0.0
    
    print("\nStarting training...")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Early stopping patience: {config.EARLY_STOP_PATIENCE}")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_dice, train_sens = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_dice, val_sens, val_prec = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # V0.3: Update LR scheduler (CosineAnnealing steps every epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_sens'].append(train_sens)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_sens'].append(val_sens)
        history['val_prec'].append(val_prec)
        history['lr'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, Sens: {train_sens:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Sens: {val_sens:.4f}, Prec: {val_prec:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_sens': val_sens,
                'val_prec': val_prec
            }, checkpoint_path)
            print(f"  ✓ Saved best model (Dice: {val_dice:.4f})")
        
        # Early stopping check
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        print("="*60)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    
    # Save history
    history_path = Path(config.LOG_DIR) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
