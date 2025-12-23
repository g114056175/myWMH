"""
Analyze channel importance from trained V0.3 model
通过第一层卷积权重分析7通道的重要性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import AttentionUNet
import config

def main():
    print("="*70)
    print("7通道权重分析 - V0.3 Trained Model")
    print("="*70)
    
    # Load trained model
    device = torch.device('cpu')  # Use CPU for analysis
    model = AttentionUNet(in_channels=7)
    
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nModel loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Get first convolution layer weights
    # Shape: (out_channels, in_channels, kernel_h, kernel_w)
    first_conv = model.encoders[0].conv[0]  # First conv in first encoder
    weights = first_conv.weight.data  # (64, 7, 3, 3)
    
    print(f"\nFirst Conv Layer:")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Explanation: ({weights.shape[0]} output channels, {weights.shape[1]} input channels, {weights.shape[2]}x{weights.shape[3]} kernel)")
    
    # Calculate importance for each channel
    # Method 1: Mean absolute weight (L1 norm)
    channel_importance_l1 = weights.abs().mean(dim=[0, 2, 3]).cpu().numpy()
    
    # Method 2: L2 norm (energy)
    channel_importance_l2 = weights.pow(2).mean(dim=[0, 2, 3]).sqrt().cpu().numpy()
    
    # Method 3: Variance (contribution to output diversity)
    channel_variance = weights.var(dim=[0, 2, 3]).cpu().numpy()
    
    # Normalize to percentages
    total_l1 = channel_importance_l1.sum()
    percentage_l1 = (channel_importance_l1 / total_l1) * 100
    
    total_l2 = channel_importance_l2.sum()
    percentage_l2 = (channel_importance_l2 / total_l2) * 100
    
    # Channel names
    channel_names = [
        'CLAHE[t-1]',
        'CLAHE[t]',
        'CLAHE[t+1]',
        'T1',
        'HighPass',
        'Asymmetry',
        'Spatial Atlas'
    ]
    
    # Print results
    print("\n" + "="*70)
    print("通道重要性分析结果")
    print("="*70)
    print("\n基于第一层卷积权重的L1范数（绝对值平均）:")
    print(f"{'通道':<20} {'权重大小':<12} {'百分比':<10} {'排名'}")
    print("-"*70)
    
    # Sort by importance
    sorted_indices = np.argsort(percentage_l1)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        name = channel_names[idx]
        importance = channel_importance_l1[idx]
        percentage = percentage_l1[idx]
        print(f"{name:<20} {importance:<12.4f} {percentage:>6.2f}%   Rank {rank}")
    
    print("\n" + "="*70)
    print("基于L2范数（能量）:")
    print(f"{'通道':<20} {'L2 Norm':<12} {'百分比':<10}")
    print("-"*70)
    for idx in sorted_indices:
        name = channel_names[idx]
        importance = channel_importance_l2[idx]
        percentage = percentage_l2[idx]
        print(f"{name:<20} {importance:<12.4f} {percentage:>6.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Bar chart - L1 importance
    ax1 = axes[0, 0]
    colors = ['#FF6B6B' if 'Asymmetry' in name else '#4ECDC4' if 'Atlas' in name else '#95E1D3' 
              for name in channel_names]
    bars = ax1.bar(range(7), percentage_l1, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('重要性 (%)', fontsize=12)
    ax1.set_title('通道重要性 (L1权重)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentage_l1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Pie chart
    ax2 = axes[0, 1]
    ax2.pie(percentage_l1, labels=channel_names, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('通道权重分布', fontsize=14, fontweight='bold')
    
    # 3. Sorted bar chart
    ax3 = axes[1, 0]
    sorted_names = [channel_names[i] for i in sorted_indices]
    sorted_pct = [percentage_l1[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    bars2 = ax3.barh(range(7), sorted_pct, color=sorted_colors, alpha=0.8, edgecolor='black')
    ax3.set_yticks(range(7))
    ax3.set_yticklabels(sorted_names, fontsize=10)
    ax3.set_xlabel('重要性 (%)', fontsize=12)
    ax3.set_title('通道重要性排名', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # Add rank labels
    for i, (bar, pct) in enumerate(zip(bars2, sorted_pct)):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f' #{i+1} ({pct:.1f}%)', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
V0.3 通道权重分析总结

最重要通道 (Top 3):
1. {channel_names[sorted_indices[0]]}: {percentage_l1[sorted_indices[0]]:.2f}%
2. {channel_names[sorted_indices[1]]}: {percentage_l1[sorted_indices[1]]:.2f}%
3. {channel_names[sorted_indices[2]]}: {percentage_l1[sorted_indices[2]]:.2f}%

最不重要通道 (Bottom 2):
6. {channel_names[sorted_indices[5]]}: {percentage_l1[sorted_indices[5]]:.2f}%
7. {channel_names[sorted_indices[6]]}: {percentage_l1[sorted_indices[6]]:.2f}%

分析说明:
- 权重大小反映该通道对模型的贡献
- 注意: Asymmetry可能被HFlip破坏
- 第一层权重只是初步指标

建议:
- 如权重<10%: 考虑移除
- 如权重>20%: 该通道很重要
- 需要消融实验最终确认
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('V0.3 模型 - 7通道重要性分析 (基于训练权重)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path('d:/VSCode/AIOT_E1/TEMP2/v03_channel_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    # Additional analysis: correlation between channels
    print("\n" + "="*70)
    print("额外分析: 通道间权重相关性")
    print("="*70)
    print("\n注意: 这仅显示第一层的权重分析")
    print("完整的通道贡献需要:")
    print("  1. Grad-CAM可视化")
    print("  2. 消融实验")
    print("  3. 特征图分析")
    print("\n✅ 通道权重分析完成!")


if __name__ == '__main__':
    main()
