import pandas as pd
import os

def excel_to_csv(excel_file, output_dir=None):
    """
    将Excel文件中的每个工作表转换为CSV文件，并使用工作表名称命名
    
    参数:
        excel_file: Excel文件路径
        output_dir: 输出目录，默认为None，表示与Excel文件相同目录
    """
    print(f"正在处理: {excel_file}")
    
    # 如果未指定输出目录，则使用Excel文件所在的目录
    if output_dir is None:
        output_dir = os.path.dirname(excel_file)
        if output_dir == '':
            output_dir = '.'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取不带扩展名的Excel文件名
    base_name = os.path.basename(excel_file)
    file_name = os.path.splitext(base_name)[0]
    
    # 读取Excel文件中的所有工作表
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names
    
    # 处理每个工作表
    for sheet_name in sheet_names:
        # 读取工作表数据
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 构建CSV文件名
        csv_file = f"{file_name}-{sheet_name}.csv"
        csv_path = os.path.join(output_dir, csv_file)
        
        # 保存为CSV文件，使用UTF-8编码
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  导出工作表 '{sheet_name}' 到 {csv_file}")
    
    print(f"共导出 {len(sheet_names)} 个工作表")
    return len(sheet_names)

def main():
    # 要处理的Excel文件列表
    excel_files = [
        "附件1-2023年教师教学评分表（含2个表格）.xls",
        "附件2-2024年教师教学评分表（含20学院）.xls"
    ]
    
    total_sheets = 0
    
    # 处理每个Excel文件
    for excel_file in excel_files:
        if os.path.exists(excel_file):
            sheets = excel_to_csv(excel_file)
            total_sheets += sheets
        else:
            print(f"错误: 文件 '{excel_file}' 不存在!")
    
    print(f"总共导出 {total_sheets} 个CSV文件")

if __name__ == "__main__":
    main() 