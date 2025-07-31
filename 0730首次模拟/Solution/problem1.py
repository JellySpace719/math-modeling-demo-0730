import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 生成优化后的饼状图
def generate_pie_chart():
    # 获取前10个城市的数据
    top_10_cities = sorted_cities[:10]
    cities = [city for city, count in top_10_cities]
    counts = [count for city, count in top_10_cities]
    
    # 设置颜色方案 - 使用漂亮的颜色
    colors = [
        '#C70039', '#FF5733', '#FFC300', '#FF8D1A', 
        '#DAF7A6', '#C4E538', '#7ED957', '#3498DB', 
        '#9B59B6', '#2471A3'
    ]
    
    # 创建图形和子图
    plt.figure(figsize=(14, 10), facecolor='white')
    
    # 创建两个子图，左边是饼图，右边是图例
    ax1 = plt.subplot(121)
    
    # 突出显示最大的部分
    explode = [0.05 if i == 0 else 0 for i in range(len(cities))]
    
    # 绘制饼图 - 取消阴影效果，使用平面设计
    wedges, texts, autotexts = ax1.pie(
        counts, 
        explode=explode,
        labels=None,  # 不在饼图上直接显示标签
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=False,  # 取消阴影
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'},
        pctdistance=0.75
    )
    
    # 设置百分比文本的样式
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # 添加标题
    ax1.set_title('获评最高评分(BS=5.0)景点数量前10城市分布', fontsize=16, pad=20)
    
    # 创建单独的图例
    ax2 = plt.subplot(122)
    ax2.axis('off')  # 关闭坐标轴
    
    # 创建自定义图例
    legend_elements = []
    for i, (city, count) in enumerate(top_10_cities):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i], 
                                           label=f"{city} ({count}个)"))
    
    # 添加图例，并调整位置和字体大小
    legend = ax2.legend(handles=legend_elements, loc='center', 
                       title="城市及最高评分景点数量", 
                       fontsize=12, title_fontsize=14)
    
    # 添加副标题
    plt.figtext(0.5, 0.94, f"全国共有{total_bs_spots}个景点获评最高评分", 
               ha='center', fontsize=14)
    
    # 添加数据来源
    plt.figtext(0.5, 0.02, "数据来源: 全国352个城市旅游景点数据集", 
               ha='center', fontsize=10, style='italic')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('problem1_pie_chart.png', dpi=300, bbox_inches='tight')
    print("优化后的饼状图已保存为 'problem1_pie_chart.png'")
    
    # 显示图表
    plt.show()

# 生成优化后的水平条形图（替代柱状图，更清晰）
def generate_bar_chart():
    # 获取前10个城市的数据
    top_10_cities = sorted_cities[:10]
    cities = [city for city, count in top_10_cities]
    counts = [count for city, count in top_10_cities]
    
    # 反转顺序，使最高的在顶部
    cities.reverse()
    counts.reverse()
    
    # 设置颜色方案
    colors = [
        '#2471A3', '#9B59B6', '#3498DB', '#7ED957', '#C4E538',
        '#DAF7A6', '#FF8D1A', '#FFC300', '#FF5733', '#C70039'
    ]
    colors.reverse()  # 反转颜色顺序
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    # 创建水平条形图
    bars = ax.barh(
        cities, 
        counts, 
        color=colors,
        height=0.6,
        edgecolor='white',
        linewidth=1
    )
    
    # 在条形上显示数值
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.5, 
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}个',
            ha='left', 
            va='center',
            fontsize=11,
            fontweight='bold'
        )
    
    # 设置标题和轴标签
    ax.set_title('获评最高评分(BS=5.0)景点数量前10城市', fontsize=16, pad=20)
    ax.set_xlabel('最高评分景点数量', fontsize=12, labelpad=10)
    
    # 设置网格线
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 设置背景色
    ax.set_facecolor('#f9f9f9')
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数据来源注释
    plt.figtext(0.02, 0.02, "数据来源: 全国352个城市旅游景点数据集", fontsize=9, style='italic')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('problem1_bar_chart.png', dpi=300, bbox_inches='tight')
    print("优化后的条形图已保存为 'problem1_bar_chart.png'")
    
    # 显示图表
    plt.show()

# 生成图表
generate_pie_chart()
generate_bar_chart() 