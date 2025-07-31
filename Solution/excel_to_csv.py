import os
import pandas as pd
import glob

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_dir = os.path.dirname(current_dir)

# 设置源目录和目标目录的绝对路径
source_dir = os.path.join(project_dir, '中国城市统计年鉴2024（excel）')
target_dir = os.path.join(project_dir, '中国城市统计年鉴2024（csv）')

def convert_excel_to_csv():
    """
    将中国城市统计年鉴2024（excel）文件夹中的所有xlsx文件转换为csv格式
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")
    
    # 获取所有子目录
    for root, dirs, files in os.walk(source_dir):
        # 计算相对路径
        rel_path = os.path.relpath(root, source_dir)
        
        # 如果是根目录，设置为空字符串
        if rel_path == '.':
            rel_path = ''
        
        # 创建对应的目标子目录
        target_subdir = os.path.join(target_dir, rel_path)
        if rel_path and not os.path.exists(target_subdir):
            os.makedirs(target_subdir)
            print(f"创建子目录: {target_subdir}")
        
        # 处理当前目录下的所有xlsx文件
        excel_files = [f for f in files if f.endswith('.xlsx')]
        for excel_file in excel_files:
            excel_path = os.path.join(root, excel_file)
            csv_file = excel_file.replace('.xlsx', '.csv')
            csv_path = os.path.join(target_subdir, csv_file)
            
            try:
                # 读取Excel文件
                df = pd.read_excel(excel_path)
                
                # 保存为CSV文件
                df.to_csv(csv_path, encoding='utf-8-sig', index=False)
                print(f"已转换: {excel_file}")
            except Exception as e:
                print(f"转换失败: {excel_file} - 错误: {str(e)}")
    
    print("\n转换完成！")
    print(f"Excel文件所在目录: {source_dir}")
    print(f"CSV文件保存目录: {target_dir}")
    
    # 返回生成的CSV文件目录
    return target_dir

if __name__ == "__main__":
    print("开始将Excel文件转换为CSV格式...")
    output_dir = convert_excel_to_csv()
    
    # 输出转换后的文件列表
    print("\n转换后的文件列表:")
    csv_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.csv'):
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                csv_files.append(rel_path)
    
    # 显示部分文件
    for i, file in enumerate(sorted(csv_files)[:10]):
        print(f"{i+1}. {file}")
    
    print(f"共转换了 {len(csv_files)} 个文件") 