"""
Attention U-Net with Deep Supervision for WMH segmentation
V0.6: Deep Supervision + Dropout 0.15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
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
    """Spatial Attention Gate for skip connections"""
    
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
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DualAttentionGate(nn.Module):
    """Dual Attention Gate (Channel + Spatial)"""
    
    def __init__(self, F_g, F_l, F_int):
        super(DualAttentionGate, self).__init__()
        self.channel_att = ChannelAttention(F_l)
        self.spatial_att = AttentionGate(F_g, F_l, F_int)
    
    def forward(self, g, x):
        x_ch = self.channel_att(x)
        x_sp = self.spatial_att(g, x_ch)
        return x_sp


class ConvBlock(nn.Module):
    """Convolutional block with Dropout - V0.6: Dropout 0.15"""
    
    def __init__(self, in_channels, out_channels, dropout=0.15):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # V0.6: 0.15
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net with Deep Supervision
    
    V0.6 Features:
    - Deep Supervision: 3 auxiliary outputs
    - Dropout: 0.15 (stronger regularization)
    - 7-channel input (CLAHE×3, T1, HighPass, Asymmetry, SpatialAtlas)
    """
    
    def __init__(self, in_channels=7, base_channels=64, depth=5, dropout=0.15):
        super(AttentionUNet, self).__init__()
        
        self.depth = depth
        self.base_channels = base_channels
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(current_channels, out_ch, dropout))
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
            self.attention_gates.append(
                DualAttentionGate(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2)
            )
            self.decoders.append(
                ConvBlock(in_ch, out_ch, dropout)  # in_ch because of concat
            )
        
        # === Deep Supervision: Auxiliary Heads ===
        # Head 1: From decoder layer 1 (deepest decoder, 1/8 resolution)
        self.aux_head1 = nn.Conv2d(base_channels * 8, 1, kernel_size=1)
        
        # Head 2: From decoder layer 2 (1/4 resolution)
        self.aux_head2 = nn.Conv2d(base_channels * 4, 1, kernel_size=1)
        
        # Head 3: From decoder layer 3 (1/2 resolution)
        self.aux_head3 = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        
        # Final output
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        
    def forward(self, x, deep_supervision=True):
        """
        Args:
            x: Input tensor [B, 7, H, W]
            deep_supervision: If True, return auxiliary outputs for training
        
        Returns:
            During training (deep_supervision=True): 
                [aux1, aux2, aux3, final] - list of 4 outputs
            During inference (deep_supervision=False):
                final output only
        """
        target_size = x.shape[2:]  # Store for upsampling
        
        # Encoder
        enc_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            enc_features.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
        
        # Decoder with deep supervision
        aux_outputs = []
        
        for i, (upconv, att_gate, decoder) in enumerate(
            zip(self.upconvs, self.attention_gates, self.decoders)
        ):
            x = upconv(x)
            skip = enc_features[-(i+2)]  # Corresponding encoder feature
            skip_att = att_gate(x, skip)
            x = torch.cat([x, skip_att], dim=1)
            x = decoder(x)
            
            # Deep Supervision: Collect auxiliary outputs
            if deep_supervision and i < 3:  # First 3 decoder layers
                if i == 0:  # Deepest decoder
                    aux1 = self.aux_head1(x)
                    aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
                    aux_outputs.append(aux1)
                elif i == 1:
                    aux2 = self.aux_head2(x)
                    aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)
                    aux_outputs.append(aux2)
                elif i == 2:
                    aux3 = self.aux_head3(x)
                    aux3 = F.interpolate(aux3, size=target_size, mode='bilinear', align_corners=False)
                    aux_outputs.append(aux3)
        
        # Final output
        final = self.final_conv(x)
        
        if deep_supervision:
            # Return all outputs for training
            # Order: [aux1, aux2, aux3, final]
            return aux_outputs + [final]
        else:
            # Return only final output for inference
            return final


if __name__ == "__main__":
    # Test
    model = AttentionUNet(in_channels=7, base_channels=64, depth=5, dropout=0.15)
    x = torch.randn(2, 7, 224, 224)
    
    # Training mode
    outputs = model(x, deep_supervision=True)
    print(f"Training mode - {len(outputs)} outputs:")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: {out.shape}")
    
    # Inference mode
    output = model(x, deep_supervision=False)
    print(f"\nInference mode - 1 output: {output.shape}")
    
    # Parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
