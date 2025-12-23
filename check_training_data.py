"""
檢查training數據完整性 - 詳細報告
"""
from pathlib import Path
import pandas as pd

def check_case_files(case_dir):
    """檢查單個case的文件狀態"""
    files_status = {}
    
    # 檢查FLAIR
    flair_path = case_dir / 'pre' / 'FLAIR.nii.gz'
    files_status['FLAIR'] = flair_path.exists()
    
    # 檢查T1
    t1_path = case_dir / 'pre' / 'T1.nii.gz'
    files_status['T1'] = t1_path.exists()
    
    # 檢查WMH標籤
    wmh_path = case_dir / 'wmh.nii.gz'
    files_status['WMH'] = wmh_path.exists()
    
    # 檢查所有文件
    all_files = list(case_dir.rglob('*'))
    files_status['total_files'] = len([f for f in all_files if f.is_file()])
    files_status['all_dirs'] = [d.name for d in case_dir.iterdir() if d.is_dir()]
    
    return files_status

def main():
    print("=" * 80)
    print("檢查Training數據完整性")
    print("=" * 80)
    
    data_dir = Path('data/wmh/training')
    
    results = []
    
    # 遍歷所有region
    for region_dir in sorted(data_dir.iterdir()):
        if not region_dir.is_dir():
            continue
        
        region_name = region_dir.name
        print(f"\n檢查 {region_name}...")
        
        # 遍歷所有scanner
        for scanner_dir in sorted(region_dir.iterdir()):
            if not scanner_dir.is_dir():
                continue
            
            scanner_name = scanner_dir.name
            
            # 遍歷所有cases
            for case_dir in sorted(scanner_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                
                case_id = case_dir.name
                full_id = f"{region_name}/{scanner_name}/{case_id}"
                
                status = check_case_files(case_dir)
                
                result = {
                    'Region': region_name,
                    'Scanner': scanner_name,
                    'CaseID': case_id,
                    'FullPath': full_id,
                    'FLAIR': '✓' if status['FLAIR'] else '✗',
                    'T1': '✓' if status['T1'] else '✗',
                    'WMH': '✓' if status['WMH'] else '✗',
                    'Complete': '✓' if all([status['FLAIR'], status['T1'], status['WMH']]) else '✗',
                    'TotalFiles': status['total_files'],
                    'Directories': ', '.join(status['all_dirs'])
                }
                
                results.append(result)
    
    # 創建DataFrame
    df = pd.DataFrame(results)
    
    # 統計
    print("\n" + "=" * 80)
    print("統計摘要")
    print("=" * 80)
    
    total_cases = len(df)
    complete_cases = len(df[df['Complete'] == '✓'])
    incomplete_cases = len(df[df['Complete'] == '✗'])
    
    print(f"\n總Cases數: {total_cases}")
    print(f"完整Cases: {complete_cases}")
    print(f"不完整Cases: {incomplete_cases}")
    
    # 按Region統計
    print("\n按Region統計:")
    for region in df['Region'].unique():
        region_df = df[df['Region'] == region]
        complete = len(region_df[region_df['Complete'] == '✓'])
        total = len(region_df)
        print(f"  {region}: {complete}/{total} 完整")
    
    # 顯示不完整的cases
    print("\n" + "=" * 80)
    print("不完整的Cases詳情")
    print("=" * 80)
    
    incomplete_df = df[df['Complete'] == '✗']
    
    if len(incomplete_df) > 0:
        print(f"\n共 {len(incomplete_df)} 個不完整cases:\n")
        for idx, row in incomplete_df.iterrows():
            print(f"{row['FullPath']}")
            print(f"  FLAIR: {row['FLAIR']}, T1: {row['T1']}, WMH: {row['WMH']}")
            print(f"  目錄: {row['Directories']}")
            print(f"  總文件數: {row['TotalFiles']}")
            print()
    else:
        print("\n✓ 所有cases都完整！")
    
    # 保存詳細報告
    output_file = 'training_data_check_report.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n詳細報告已保存至: {output_file}")
    
    # 顯示前20個cases作為樣本
    print("\n" + "=" * 80)
    print("前20個Cases狀態")
    print("=" * 80)
    print(df[['FullPath', 'FLAIR', 'T1', 'WMH', 'Complete']].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
