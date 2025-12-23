"""
2D U-Net架構
純2D分割模型 - 單切片輸入，無空間上下文依賴
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet2D(nn.Module):
    """
    純2D U-Net分割模型
    
    Args:
        in_channels (int): 輸入通道數（FLAIR單通道=1）
        out_channels (int): 輸出通道數（二分類=1）
        base_channels (int): 基礎通道數，默認64
        dropout_rate (float): Dropout比率，防止過擬合
    
    Architecture:
        Encoder: 4層下採樣 [64, 128, 256, 512]
        Bottleneck: 1024通道
        Decoder: 4層上採樣 + skip connections
        Output: Sigmoid激活
    
    Input Shape: (B, 1, H, W)
    Output Shape: (B, 1, H, W) - 像素級WMH概率
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, dropout_rate=0.3):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)  # 16 = 8 (upconv) + 8 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Output logits (B, 1, H, W) - apply sigmoid for probability
        """
        # Encoder
        enc1 = self.enc1(x)          # (B, 64, H, W)
        pool1 = self.pool1(enc1)     # (B, 64, H/2, W/2)
        
        enc2 = self.enc2(pool1)      # (B, 128, H/2, W/2)
        pool2 = self.pool2(enc2)     # (B, 128, H/4, W/4)
        
        enc3 = self.enc3(pool2)      # (B, 256, H/4, W/4)
        pool3 = self.pool3(enc3)     # (B, 256, H/8, W/8)
        
        enc4 = self.enc4(pool3)      # (B, 512, H/8, W/8)
        pool4 = self.pool4(enc4)     # (B, 512, H/16, W/16)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # (B, 1024, H/16, W/16)
        bottleneck = self.dropout(bottleneck)
        
        # Decoder with skip connections
        up4 = self.upconv4(bottleneck)  # (B, 512, H/8, W/8)
        cat4 = torch.cat([up4, enc4], dim=1)  # (B, 1024, H/8, W/8)
        dec4 = self.dec4(cat4)          # (B, 512, H/8, W/8)
        
        up3 = self.upconv3(dec4)        # (B, 256, H/4, W/4)
        cat3 = torch.cat([up3, enc3], dim=1)  # (B, 512, H/4, W/4)
        dec3 = self.dec3(cat3)          # (B, 256, H/4, W/4)
        
        up2 = self.upconv2(dec3)        # (B, 128, H/2, W/2)
        cat2 = torch.cat([up2, enc2], dim=1)  # (B, 256, H/2, W/2)
        dec2 = self.dec2(cat2)          # (B, 128, H/2, W/2)
        
        up1 = self.upconv1(dec2)        # (B, 64, H, W)
        cat1 = torch.cat([up1, enc1], dim=1)  # (B, 128, H, W)
        dec1 = self.dec1(cat1)          # (B, 64, H, W)
        
        # Output
        out = self.out_conv(dec1)       # (B, 1, H, W)
        
        return out


def count_parameters(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 測試模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    
    print(f"Model: UNet2D")
    print(f"Parameters: {count_parameters(model):,}")
    
    # 測試forward pass
    x = torch.randn(2, 1, 256, 256).to(device)
    out = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # 測試Sigmoid輸出
    prob = torch.sigmoid(out)
    print(f"Probability range: [{prob.min():.4f}, {prob.max():.4f}]")
    
    print("\n✓ Model architecture test passed!")
