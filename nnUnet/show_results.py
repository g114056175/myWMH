"""
簡單提取全測試集Dice統計
"""
import json

with open("full_test_set_results.json", "r") as f:
    results = json.load(f)

dices = [r['metrics']['dice'] for r in results]

print(f"=" * 80)
print(f"完整測試集結果 (110 cases)")
print(f"=" * 80)
print(f"Cases: {len(results)}")
print(f"Average Dice: {sum(dices)/len(dices):.4f}")
print(f"Median Dice: {sorted(dices)[len(dices)//2]:.4f}")
print(f"Min Dice: {min(dices):.4f}")
print(f"Max Dice: {max(dices):.4f}")
print(f"Std: {(sum([(d - sum(dices)/len(dices))**2 for d in dices])/len(dices))**0.5:.4f}")
