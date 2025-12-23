"""
Attention U-Net for WMH segmentation
Deeper architecture to handle high-pass filter features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module - V0.3"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
       
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class AttentionGate(nn.Module):
    """Spatial Attention Gate for skip connections - V0.2"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (upsampled)
            x: Skip connection from encoder
        """
        # Match sizes if needed
        if g.shape[2:] != x.shape[2:]:
            # Resize g to match x
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DualAttentionGate(nn.Module):
    """Dual Attention Gate (Channel + Spatial) - V0.3"""
    
    def __init__(self, F_g, F_l, F_int):
        super(DualAttentionGate, self).__init__()
        
        # Channel Attention
        self.channel_attention = ChannelAttention(F_l, reduction=16)
        
        # Spatial Attention (same as before)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (upsampled)
            x: Skip connection from encoder
        """
        # 1. Channel Attention: refine channel-wise features
        x = self.channel_attention(x)
        
        # 2. Spatial Attention: focus on important spatial locations
        # Match sizes if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net with Dual Attention - V0.3
    
    Args:
        in_channels: 7 (CLAHE×3 + T1 + HighPass + Asymmetry + Spatial Atlas)
        out_channels: 1 (binary segmentation)
        base_channels: 64 (starting channels)
        depth: 5 (encoder/decoder depth)
    """
    
    def __init__(self, in_channels=7, out_channels=1, base_channels=64, depth=5):
        super(AttentionUNet, self).__init__()
        
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(current_channels, out_ch))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = out_ch
        
        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth - 1):
            in_ch = base_channels * (2 ** (depth - 1 - i))
            out_ch = in_ch // 2
            
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # V0.3: Use DualAttentionGate instead of AttentionGate
            self.attention_gates.append(
                DualAttentionGate(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2)
            )
            self.decoders.append(
                ConvBlock(in_ch, out_ch)  # in_ch because of concatenation
            )
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        skip_connections = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            if i < self.depth - 1:
                skip_connections.append(x)
                x = self.pools[i](x)
        
        # Decoder with attention gates
        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            
            # Get skip connection
            skip = skip_connections[-(i + 1)]
            
            # Match sizes (in case of odd dimensions)
            if x.shape != skip.shape:
                # Crop or pad to match
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                
                if diff_h > 0 or diff_w > 0:
                    # Crop skip
                    skip = skip[:, :, 
                                diff_h//2:diff_h//2 + x.shape[2],
                                diff_w//2:diff_w//2 + x.shape[3]]
                elif diff_h < 0 or diff_w < 0:
                    # Pad x
                    x = F.pad(x, [
                        -diff_w//2, (-diff_w) - (-diff_w//2),
                        -diff_h//2, (-diff_h) - (-diff_h//2)
                    ])
            
            # Apply attention gate to skip connection
            skip = self.attention_gates[i](g=x, x=skip)
            
            # Concatenate
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)
        
        # Final output
        output = self.final(x)
        return output


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model - V0.3
    model = AttentionUNet(in_channels=7, out_channels=1, base_channels=64, depth=5)
    
    # Test input (batch_size=2, channels=7, height=224, width=224)
    x = torch.randn(2, 7, 224, 224)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Expected output: (2, 1, 224, 224)
    assert output.shape == (2, 1, 224, 224), f"Unexpected output shape: {output.shape}"
    print("✓ Model test passed!")
