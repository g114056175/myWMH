"""
快速验证：测试集病变分布是否与spatial atlas一致
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'newUnet'))
from dataset import center_crop
import config

# 加载spatial atlas
atlas = np.load('spatial_atlas_prior.npy')

# 找到所有测试样本
test_dir = Path(config.TEST_DIR)
test_masks = list(test_dir.rglob('*/wmh.nii.gz'))

# 统计
correlations = []
overlaps = []
atlas_scores = []

for mask_file in test_masks:
    mask_raw = nib.load(mask_file).get_fdata()
    mask_cropped = center_crop(mask_raw, 224)
    
    for z in range(2, mask_cropped.shape[2] - 2):
        mask_slice = mask_cropped[:, :, z]
        wmh_mask = (mask_slice == 1).astype(np.float32)
        
        if wmh_mask.sum() == 0:
            continue
        
        # 计算WMH区域的atlas平均值
        wmh_atlas_score = atlas[wmh_mask == 1].mean()
        
        # 计算非WMH区域的atlas平均值
        non_wmh_atlas_score = atlas[wmh_mask == 0].mean()
        
        # 对比度
        if non_wmh_atlas_score > 0:
            contrast = wmh_atlas_score / non_wmh_atlas_score
        else:
            contrast = 0
        
        correlations.append(contrast)
        
        # 计算overlap（WMH与高atlas值区域的重叠）
        high_atlas = (atlas > atlas.mean()).astype(np.float32)
        overlap = (wmh_mask * high_atlas).sum() / wmh_mask.sum()
        overlaps.append(overlap)
        
        atlas_scores.append(wmh_atlas_score)

# 统计结果
avg_contrast = np.mean(correlations)
avg_overlap = np.mean(overlaps)
avg_atlas_score = np.mean(atlas_scores)

print(f"\n测试集验证结果 ({len(test_masks)} 样本):")
print(f"  WMH区域atlas得分 vs 非WMH区域对比度: {avg_contrast:.2f}x")
print(f"  WMH与高atlas区域重叠率: {avg_overlap*100:.1f}%")
print(f"  WMH区域平均atlas值: {avg_atlas_score:.4f}")
print(f"  Atlas全局平均值: {atlas.mean():.4f}")

# 结论
if avg_contrast > 1.5 and avg_overlap > 0.5:
    print(f"\n✅ 结论: SPATIAL ATLAS 非常合理！")
    print(f"   - WMH区域的atlas值是非WMH的 {avg_contrast:.1f}倍")
    print(f"   - {avg_overlap*100:.0f}% 的WMH出现在高先验区域")
elif avg_contrast > 1.2 and avg_overlap > 0.4:
    print(f"\n⚠️ 结论: SPATIAL ATLAS 合理，但效果中等")
    print(f"   - 对比度 {avg_contrast:.1f}x 略低")
    print(f"   - 重叠率 {avg_overlap*100:.0f}% 尚可")
else:
    print(f"\n❌ 结论: SPATIAL ATLAS 效果不佳")
    print(f"   - 对比度仅 {avg_contrast:.1f}x")
    print(f"   - 重叠率仅 {avg_overlap*100:.0f}%")
