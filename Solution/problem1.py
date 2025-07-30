import os
import pandas as pd
import glob

# 定义数据目录
data_dir = "../题目/附件"

# 获取所有CSV文件路径
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

# 初始化变量
all_scores = []  # 所有评分
city_bs_count = {}  # 每个城市拥有最高评分的景点数量

# 遍历所有城市CSV文件
for file_path in csv_files:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 提取城市名称（从文件名中）
        city_name = os.path.basename(file_path).replace('.csv', '')
        
        # 将评分列转换为数值型，忽略无法转换的值
        df['评分'] = pd.to_numeric(df['评分'], errors='coerce')
        
        # 添加所有有效评分到列表中
        valid_scores = df['评分'].dropna().tolist()
        all_scores.extend(valid_scores)
        
        # 统计该城市拥有的最高评分景点数量（先不确定最高评分是多少）
        if len(valid_scores) > 0:
            city_max_score = max(valid_scores)
            city_bs_count[city_name] = {'max_score': city_max_score, 'count': 0}
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 计算全局最高评分
if all_scores:
    best_score = max(all_scores)
    print(f"问题1: 所有景点评分的最高分(BS)是: {best_score}")
else:
    print("未找到有效的评分数据")
    best_score = 0

# 重新遍历所有城市，统计拥有最高评分的景点数量
total_bs_spots = 0
for file_path in csv_files:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 提取城市名称
        city_name = os.path.basename(file_path).replace('.csv', '')
        
        # 将评分列转换为数值型
        df['评分'] = pd.to_numeric(df['评分'], errors='coerce')
        
        # 计算该城市拥有的最高评分景点数量
        bs_count = sum(df['评分'] == best_score)
        city_bs_count[city_name] = bs_count
        total_bs_spots += bs_count
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 输出拥有最高评分的景点总数
print(f"全国有 {total_bs_spots} 个景点获评了最高评分(BS)")

# 按照拥有最高评分景点数量排序城市
sorted_cities = sorted(city_bs_count.items(), key=lambda x: x[1], reverse=True)

# 输出拥有最高评分景点最多的城市
if sorted_cities and sorted_cities[0][1] > 0:
    top_cities = [city for city, count in sorted_cities if count == sorted_cities[0][1]]
    print(f"获评最高评分(BS)景点最多的城市有: {', '.join(top_cities)}")

# 输出前10个拥有最高评分景点数量最多的城市
print("\n依据拥有最高评分(BS)景点数量的多少排序，前10个城市是:")
for i, (city, count) in enumerate(sorted_cities[:10], 1):
    print(f"{i}. {city}: {count}个")

# 将结果保存到文件
with open('problem1_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"问题1: 所有景点评分的最高分(BS)是: {best_score}\n")
    f.write(f"全国有 {total_bs_spots} 个景点获评了最高评分(BS)\n")
    if sorted_cities and sorted_cities[0][1] > 0:
        top_cities = [city for city, count in sorted_cities if count == sorted_cities[0][1]]
        f.write(f"获评最高评分(BS)景点最多的城市有: {', '.join(top_cities)}\n\n")
    
    f.write("依据拥有最高评分(BS)景点数量的多少排序，前10个城市是:\n")
    for i, (city, count) in enumerate(sorted_cities[:10], 1):
        f.write(f"{i}. {city}: {count}个\n") 