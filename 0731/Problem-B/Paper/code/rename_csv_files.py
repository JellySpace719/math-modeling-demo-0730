import os
import re

def rename_csv_files():
    # 获取当前目录下的所有文件
    files = os.listdir('.')
    
    # 筛选出以"附件2"开头的CSV文件
    pattern = r'附件2-2024年教师教学评分表（含20学院）-学院([A-Z]).csv'
    renamed_count = 0
    
    print("开始重命名文件...")
    
    # 遍历文件并重命名
    for file in files:
        match = re.match(pattern, file)
        if match:
            college_code = match.group(1)
            new_name = f"attachment2_college_{college_code}_scores.csv"
            
            try:
                os.rename(file, new_name)
                print(f"已重命名: {file} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名 {file} 时出错: {e}")
    
    print(f"\n重命名完成! 共重命名了 {renamed_count} 个文件。")

if __name__ == "__main__":
    rename_csv_files() 