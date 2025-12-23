"""
分析V0.3的错误类型：FP vs FN
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# V0.3 Test结果
print("="*70)
print("V0.3 错误类型分析")
print("="*70)

# 从测试结果
sensitivity = 0.8220
precision = 0.7335

# 计算错误率
fn_rate = 1 - sensitivity  # False Negative Rate (漏判率)
# precision = TP / (TP + FP)
# 假设TP = 1000, 则 FP = TP/precision - TP = TP * (1/precision - 1)
fp_ratio = (1 / precision - 1)  # FP相对于TP的比例

print(f"\nV0.3 Test Set Performance:")
print(f"  Sensitivity: {sensitivity:.4f}")
print(f"  Precision:   {precision:.4f}")

print(f"\n错误分析:")
print(f"  漏判率 (FN Rate): {fn_rate:.2%}")
print(f"    → 每100个真实WMH像素，漏掉 {fn_rate*100:.1f}个")

print(f"\n  误判比例 (FP/TP): {fp_ratio:.2%}")
print(f"    → 每100个正确检测，误判 {fp_ratio*100:.1f}个")

# Dice的分解
# Dice = 2*TP / (2*TP + FP + FN)
# 假设TP=1
tp = 1.0
fn = tp * fn_rate / (1 - fn_rate)  # FN = TP * fn_rate / sensitivity
fp = tp * fp_ratio

dice_theoretical = 2*tp / (2*tp + fp + fn)

print(f"\nDice分解 (理论):")
print(f"  假设TP = {tp:.0f}")
print(f"  则 FN ≈ {fn:.3f}")
print(f"  则 FP ≈ {fp:.3f}")
print(f"  Dice = 2*{tp:.0f} / (2*{tp:.0f} + {fp:.3f} + {fn:.3f}) = {dice_theoretical:.4f}")

# 计算如果改善每个指标的收益
print("\n" + "="*70)
print("改善潜力分析")
print("="*70)

scenarios = {
    "提高Sensitivity到0.90 (减少漏判)": {
        'sens': 0.90,
        'prec': precision
    },
    "提高Precision到0.85 (减少误判)": {
        'sens': sensitivity,
        'prec': 0.85
    },
    "同时提高到0.90和0.85": {
        'sens': 0.90,
        'prec': 0.85
    }
}

for scenario, metrics in scenarios.items():
    sens = metrics['sens']
    prec = metrics['prec']
    
    fn_new = tp * (1 - sens) / sens
    fp_new = tp * (1 / prec - 1)
    dice_new = 2*tp / (2*tp + fp_new + fn_new)
    
    improvement = dice_new - dice_theoretical
    
    print(f"\n{scenario}:")
    print(f"  新Dice: {dice_new:.4f} ({improvement:+.4f}, {improvement/dice_theoretical*100:+.1f}%)")

print("\n" + "="*70)
print("结论")
print("="*70)

if fp > fn:
    print("\n⚠️  主要问题：误判过多 (FP > FN)")
    print(f"    FP ≈ {fp:.3f}, FN ≈ {fn:.3f}")
    print(f"    Precision偏低 ({precision:.4f})")
    print("\n改进方向:")
    print("  1. 提高预测阈值 (0.7 → 0.75-0.8)")
    print("  2. 后处理：移除小connection components")
    print("  3. 使用脑白质mask过滤")
    print("  4. 训练时增加FP惩罚 (调整Tversky α)")
else:
    print("\n⚠️  主要问题：漏判过多 (FN > FP)")
    print(f"    FN ≈ {fn:.3f}, FP ≈ {fp:.3f}")
    print(f"    Sensitivity偏低 ({sensitivity:.4f})")
    print("\n改进方向:")
    print("  1. 降低预测阈值 (0.7 → 0.5-0.6)")
    print("  2. 训练时增加FN惩罚 (调整Tversky β)")
    print("  3. 使用更强的特征提取")

print("\n" + "="*70)
