import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob
import warnings
warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据目录
data_dir = "../题目/附件"
yearbook_dir = "../中国城市统计年鉴2024（csv）"

# 定义结果保存目录
result_dir = "."
os.makedirs(result_dir, exist_ok=True)

# 1. 数据收集与处理函数
def get_best_spot_scores():
    """从题目附件中获取每个城市最高评分的景点分数"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    city_best_scores = {}
    
    for file_path in csv_files:
        try:
            city_name = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path, encoding='utf-8')
            df['评分'] = pd.to_numeric(df['评分'], errors='coerce')
            
            # 获取该城市的最高景点评分
            max_score = df['评分'].max()
            if not pd.isna(max_score):
                city_best_scores[city_name] = max_score
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return pd.DataFrame(list(city_best_scores.items()), columns=['城市', '最高景点评分'])

def get_city_population():
    """从统计年鉴获取城市规模数据"""
    file_path = os.path.join(yearbook_dir, "2、地级以上城市统计资料", "2-1_人口数.csv")
    try:
        df = pd.read_csv(file_path)
        # 处理数据：提取城市名称、常住人口和城镇化率
        data = []
        for i in range(len(df)):
            row = df.iloc[i]
            if not pd.isna(row['Unnamed: 0']) and '市' in str(row['Unnamed: 0']):
                city_name = row['Unnamed: 0'].strip()
                if city_name.startswith('  '):  # 处理缩进的城市名
                    city_name = city_name.strip()
                population = row['Unnamed: 2'] if not pd.isna(row['Unnamed: 2']) else None
                urbanization = row['Unnamed: 4'] if not pd.isna(row['Unnamed: 4']) else None
                
                # 去掉城市名称中的"市"字
                if city_name.endswith('市'):
                    city_name = city_name[:-1]
                
                data.append([city_name, population, urbanization])
        
        return pd.DataFrame(data, columns=['城市', '常住人口(万人)', '城镇化率(%)'])
    except Exception as e:
        print(f"获取城市人口数据时出错: {e}")
        return pd.DataFrame(columns=['城市', '常住人口(万人)', '城镇化率(%)'])

def get_environmental_data():
    """从统计年鉴获取环境环保数据"""
    file_path = os.path.join(yearbook_dir, "2、地级以上城市统计资料", "2-5_空气质量状况和污水及生活垃圾处理率(全市).csv")
    try:
        df = pd.read_csv(file_path)
        # 处理数据：提取城市名称、PM2.5浓度、空气质量优良天数比例、污水处理率、垃圾处理率
        data = []
        for i in range(len(df)):
            row = df.iloc[i]
            if not pd.isna(row['Unnamed: 0']) and '市' in str(row['Unnamed: 0']):
                city_name = row['Unnamed: 0'].strip()
                if city_name.startswith('  '):  # 处理缩进的城市名
                    city_name = city_name.strip()
                pm25 = row['Unnamed: 2'] if not pd.isna(row['Unnamed: 2']) else None
                air_quality = row['Unnamed: 3'] if not pd.isna(row['Unnamed: 3']) else None
                sewage = row['Unnamed: 4'] if not pd.isna(row['Unnamed: 4']) else None
                garbage = row['Unnamed: 5'] if not pd.isna(row['Unnamed: 5']) else None
                
                # 去掉城市名称中的"市"字
                if city_name.endswith('市'):
                    city_name = city_name[:-1]
                
                data.append([city_name, pm25, air_quality, sewage, garbage])
        
        return pd.DataFrame(data, columns=['城市', 'PM2.5浓度', '空气质量优良天数比例(%)', '污水处理率(%)', '垃圾处理率(%)'])
    except Exception as e:
        print(f"获取环境数据时出错: {e}")
        return pd.DataFrame(columns=['城市', 'PM2.5浓度', '空气质量优良天数比例(%)', '污水处理率(%)', '垃圾处理率(%)'])

def get_cultural_data():
    """从统计年鉴获取人文底蕴数据"""
    file_path = os.path.join(yearbook_dir, "2、地级以上城市统计资料", "2-22_文化体育设施.csv")
    try:
        df = pd.read_csv(file_path)
        # 处理数据：提取城市名称、图书馆藏书量、博物馆数量
        data = []
        for i in range(len(df)):
            row = df.iloc[i]
            if not pd.isna(row['Unnamed: 0']) and '市' in str(row['Unnamed: 0']):
                city_name = row['Unnamed: 0'].strip()
                if city_name.startswith('  '):  # 处理缩进的城市名
                    city_name = city_name.strip()
                books = row['Unnamed: 2'] if not pd.isna(row['Unnamed: 2']) else None
                museums = row['Unnamed: 4'] if not pd.isna(row['Unnamed: 4']) else None
                
                # 去掉城市名称中的"市"字
                if city_name.endswith('市'):
                    city_name = city_name[:-1]
                
                data.append([city_name, books, museums])
        
        return pd.DataFrame(data, columns=['城市', '图书馆藏书量(万册)', '博物馆数量'])
    except Exception as e:
        print(f"获取文化数据时出错: {e}")
        return pd.DataFrame(columns=['城市', '图书馆藏书量(万册)', '博物馆数量'])

def get_transportation_data():
    """从统计年鉴获取交通便利数据"""
    file_path = os.path.join(yearbook_dir, "2、地级以上城市统计资料", "2-27_城市交通状况.csv")
    try:
        df = pd.read_csv(file_path)
        # 处理数据：提取城市名称、公共交通客运总量、客运量
        data = []
        for i in range(len(df)):
            row = df.iloc[i]
            if not pd.isna(row['Unnamed: 0']) and '市' in str(row['Unnamed: 0']):
                city_name = row['Unnamed: 0'].strip()
                if city_name.startswith('  '):  # 处理缩进的城市名
                    city_name = city_name.strip()
                public_transport = row['Unnamed: 2'] if not pd.isna(row['Unnamed: 2']) else None
                passenger_volume = row['Unnamed: 4'] if not pd.isna(row['Unnamed: 4']) else None
                
                # 去掉城市名称中的"市"字
                if city_name.endswith('市'):
                    city_name = city_name[:-1]
                
                data.append([city_name, public_transport, passenger_volume])
        
        return pd.DataFrame(data, columns=['城市', '公共交通客运总量(万人次)', '客运量(万人次)'])
    except Exception as e:
        print(f"获取交通数据时出错: {e}")
        return pd.DataFrame(columns=['城市', '公共交通客运总量(万人次)', '客运量(万人次)'])

def get_economic_data():
    """从统计年鉴获取经济发展水平数据"""
    file_path = os.path.join(yearbook_dir, "2、地级以上城市统计资料", "2-6_地区生产总值.csv")
    try:
        df = pd.read_csv(file_path)
        # 处理数据：提取城市名称、地区生产总值、人均地区生产总值
        data = []
        for i in range(len(df)):
            row = df.iloc[i]
            if not pd.isna(row['Unnamed: 0']) and '市' in str(row['Unnamed: 0']):
                city_name = row['Unnamed: 0'].strip()
                if city_name.startswith('  '):  # 处理缩进的城市名
                    city_name = city_name.strip()
                gdp = row['Unnamed: 2'] if not pd.isna(row['Unnamed: 2']) else None
                gdp_per_capita = row['Unnamed: 4'] if not pd.isna(row['Unnamed: 4']) else None
                
                # 去掉城市名称中的"市"字
                if city_name.endswith('市'):
                    city_name = city_name[:-1]
                
                data.append([city_name, gdp, gdp_per_capita])
        
        return pd.DataFrame(data, columns=['城市', '地区生产总值(亿元)', '人均地区生产总值(元)'])
    except Exception as e:
        print(f"获取经济数据时出错: {e}")
        return pd.DataFrame(columns=['城市', '地区生产总值(亿元)', '人均地区生产总值(元)'])

# 2. 数据整合与清洗
def merge_all_data():
    """整合所有数据源"""
    # 获取各维度数据
    best_scores_df = get_best_spot_scores()
    population_df = get_city_population()
    env_df = get_environmental_data()
    cultural_df = get_cultural_data()
    transport_df = get_transportation_data()
    economic_df = get_economic_data()
    
    # 打印每个维度的城市数量
    print(f"最高景点评分数据: {len(best_scores_df)}个城市")
    print(f"城市规模数据: {len(population_df)}个城市")
    print(f"环境环保数据: {len(env_df)}个城市")
    print(f"人文底蕴数据: {len(cultural_df)}个城市")
    print(f"交通便利数据: {len(transport_df)}个城市")
    print(f"经济发展数据: {len(economic_df)}个城市")
    
    # 基于景点数据城市列表
    all_cities = pd.DataFrame(best_scores_df['城市'].unique(), columns=['城市'])
    
    # 按顺序合并所有数据
    merged_df = all_cities.copy()
    merged_df = pd.merge(merged_df, best_scores_df, on='城市', how='left')
    merged_df = pd.merge(merged_df, population_df, on='城市', how='left')
    merged_df = pd.merge(merged_df, env_df, on='城市', how='left')
    merged_df = pd.merge(merged_df, cultural_df, on='城市', how='left')
    merged_df = pd.merge(merged_df, transport_df, on='城市', how='left')
    merged_df = pd.merge(merged_df, economic_df, on='城市', how='left')
    
    print(f"整合后的数据集: {len(merged_df)}个城市")
    
    return merged_df

def clean_and_normalize_data(merged_df):
    """清洗数据、处理缺失值、标准化数据"""
    # 复制数据，避免修改原数据
    df = merged_df.copy()
    
    # 数值类型转换
    numeric_columns = df.columns.drop('城市')
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 打印缺失值信息
    missing_info = df.isna().sum()
    print("\n各指标缺失值数量:")
    print(missing_info)
    
    # 填充缺失值（使用均值填充）
    for col in numeric_columns:
        if df[col].isna().sum() > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"'{col}'的缺失值已用均值 {mean_val:.2f} 填充")
    
    # 处理逆向指标: PM2.5浓度 (越低越好)
    if 'PM2.5浓度' in df.columns:
        max_pm25 = df['PM2.5浓度'].max()
        df['PM2.5浓度'] = max_pm25 - df['PM2.5浓度']
        print(f"逆向指标'PM2.5浓度'已转换为正向指标")
    
    # 保存原始数据（填充缺失值后）
    original_data = df.copy()
    
    # 标准化数据 (Min-Max方法)
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    
    # 标准化所有数值列
    columns_to_normalize = numeric_columns
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    return original_data, df_normalized

# 3. AHP层次分析法权重计算
def construct_ahp_matrices():
    """构建AHP判断矩阵"""
    # 准则层判断矩阵 (6x6)
    # 景点吸引力(A)、城市规模(B)、环境环保(C)、人文底蕴(D)、交通便利(E)、经济发展水平(F)
    criterion_matrix = np.array([
        [1, 3, 4, 2, 2, 3],  # 景点吸引力相对其他因素
        [1/3, 1, 2, 1/2, 1/2, 1],  # 城市规模相对其他因素
        [1/4, 1/2, 1, 1/3, 1/3, 1/2],  # 环境环保相对其他因素
        [1/2, 2, 3, 1, 1, 2],  # 人文底蕴相对其他因素
        [1/2, 2, 3, 1, 1, 2],  # 交通便利相对其他因素
        [1/3, 1, 2, 1/2, 1/2, 1]   # 经济发展水平相对其他因素
    ])
    
    # 城市规模指标判断矩阵 (2x2)
    size_matrix = np.array([
        [1, 3],  # 常住人口相对城镇化率
        [1/3, 1]  # 城镇化率相对常住人口
    ])
    
    # 环境环保指标判断矩阵 (4x4)
    env_matrix = np.array([
        [1, 1/2, 3, 2],  # PM2.5浓度相对其他因素
        [2, 1, 4, 3],  # 空气质量优良天数比例相对其他因素
        [1/3, 1/4, 1, 1/2],  # 污水处理率相对其他因素
        [1/2, 1/3, 2, 1]   # 垃圾处理率相对其他因素
    ])
    
    # 人文底蕴指标判断矩阵 (2x2)
    cultural_matrix = np.array([
        [1, 1/2],  # 图书馆藏书量相对博物馆数量
        [2, 1]   # 博物馆数量相对图书馆藏书量
    ])
    
    # 交通便利指标判断矩阵 (2x2)
    transport_matrix = np.array([
        [1, 2],  # 公共交通客运总量相对客运量
        [1/2, 1]  # 客运量相对公共交通客运总量
    ])
    
    # 经济发展水平指标判断矩阵 (2x2)
    economic_matrix = np.array([
        [1, 1/2],  # 地区生产总值相对人均地区生产总值
        [2, 1]   # 人均地区生产总值相对地区生产总值
    ])
    
    return {
        'criterion': criterion_matrix,
        'size': size_matrix,
        'env': env_matrix,
        'cultural': cultural_matrix,
        'transport': transport_matrix,
        'economic': economic_matrix
    }

def calculate_weights(matrix):
    """计算AHP权重"""
    n = matrix.shape[0]
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    eigenvalue_max = eigenvalues[max_index].real
    eigenvector = eigenvectors[:, max_index].real
    
    # 归一化特征向量，得到权重
    weights = eigenvector / np.sum(eigenvector)
    
    # 一致性检验
    CI = (eigenvalue_max - n) / (n - 1)  # 一致性指标
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}  # 随机一致性指标
    RI = RI_dict[n]
    CR = CI / RI if RI != 0 else 0  # 一致性比率
    
    consistency = "通过" if CR < 0.1 or n <= 2 else "不通过"
    
    return weights, CI, CR, consistency

def print_ahp_results(matrices):
    """打印AHP结果"""
    print("\n===== AHP权重计算结果 =====")
    
    # 计算并打印准则层权重
    criterion_weights, ci_c, cr_c, cons_c = calculate_weights(matrices['criterion'])
    print(f"\n准则层判断矩阵一致性检验: CI={ci_c:.4f}, CR={cr_c:.4f}, {cons_c}")
    criterion_names = ['景点吸引力', '城市规模', '环境环保', '人文底蕴', '交通便利', '经济发展水平']
    for i, name in enumerate(criterion_names):
        print(f"{name}的权重: {criterion_weights[i]:.4f}")
    
    # 计算并打印指标层权重
    indicator_matrices = {
        '城市规模': {'matrix': matrices['size'], 'names': ['常住人口(万人)', '城镇化率(%)']},
        '环境环保': {'matrix': matrices['env'], 'names': ['PM2.5浓度', '空气质量优良天数比例(%)', '污水处理率(%)', '垃圾处理率(%)']},
        '人文底蕴': {'matrix': matrices['cultural'], 'names': ['图书馆藏书量(万册)', '博物馆数量']},
        '交通便利': {'matrix': matrices['transport'], 'names': ['公共交通客运总量(万人次)', '客运量(万人次)']},
        '经济发展水平': {'matrix': matrices['economic'], 'names': ['地区生产总值(亿元)', '人均地区生产总值(元)']}
    }
    
    print("\n指标层判断矩阵及权重:")
    for criterion, info in indicator_matrices.items():
        weights, ci, cr, cons = calculate_weights(info['matrix'])
        print(f"\n{criterion}指标判断矩阵一致性检验: CI={ci:.4f}, CR={cr:.4f}, {cons}")
        for i, name in enumerate(info['names']):
            print(f"  {name}的权重: {weights[i]:.4f}")
    
    return criterion_weights, indicator_matrices

def calculate_combined_weights(criterion_weights, indicator_matrices):
    """计算组合权重"""
    # 初始化组合权重字典
    combined_weights = {}
    
    # 景点吸引力（单指标，直接使用准则层权重）
    combined_weights['最高景点评分'] = criterion_weights[0]
    
    # 计算其他指标的组合权重
    criterion_indices = {
        '城市规模': 1,
        '环境环保': 2,
        '人文底蕴': 3,
        '交通便利': 4,
        '经济发展水平': 5
    }
    
    for criterion, info in indicator_matrices.items():
        criterion_weight = criterion_weights[criterion_indices[criterion]]
        indicator_weights, _, _, _ = calculate_weights(info['matrix'])
        
        for i, name in enumerate(info['names']):
            combined_weights[name] = criterion_weight * indicator_weights[i]
    
    # 打印组合权重
    print("\n===== 所有指标的最终组合权重 =====")
    for indicator, weight in combined_weights.items():
        print(f"{indicator}: {weight:.4f}")
    
    return combined_weights

# 4. 城市综合得分计算与筛选
def calculate_city_scores(df_normalized, combined_weights):
    """计算城市综合得分"""
    # 创建一个新的DataFrame来存储得分
    df_scores = df_normalized[['城市']].copy()
    
    # 计算每个维度的得分
    for indicator, weight in combined_weights.items():
        if indicator in df_normalized.columns:
            df_scores[f'{indicator}_加权得分'] = df_normalized[indicator] * weight
    
    # 计算综合得分
    score_columns = [col for col in df_scores.columns if col.endswith('_加权得分')]
    df_scores['综合得分'] = df_scores[score_columns].sum(axis=1)
    
    # 按综合得分排序
    df_scores = df_scores.sort_values('综合得分', ascending=False)
    
    return df_scores

def select_top_cities(df_scores, top_n=50):
    """选择前N个城市"""
    top_cities = df_scores.head(top_n)
    
    # 打印前N个城市
    print(f"\n===== 最令外国游客向往的前{top_n}个城市 =====")
    for i, (index, row) in enumerate(top_cities.iterrows(), 1):
        print(f"{i}. {row['城市']}: {row['综合得分']:.4f}")
    
    return top_cities

# 5. 可视化结果
def visualize_top_cities(top_cities, original_data):
    """可视化前50个城市的综合得分"""
    plt.figure(figsize=(12, 15))
    
    # 条形图显示前50个城市的综合得分
    plt.barh(top_cities['城市'][::-1], top_cities['综合得分'][::-1], color='skyblue')
    plt.xlabel('综合得分')
    plt.title('最令外国游客向往的50个城市综合得分排名')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(result_dir, 'problem2_top50_cities.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制前10个城市的雷达图
    top10 = top_cities.head(10)
    
    # 选择要在雷达图中显示的维度
    dimensions = ['最高景点评分', '常住人口(万人)', 'PM2.5浓度', 
                  '空气质量优良天数比例(%)', '博物馆数量', 
                  '公共交通客运总量(万人次)', '人均地区生产总值(元)']
    
    # 确保所有维度都是正向指标（值越大越好）
    # 对于原始数据中的PM2.5浓度，需要再次转换
    orig_top10 = original_data[original_data['城市'].isin(top10['城市'])]
    
    # 对数据进行标准化处理，使得每个维度的范围在0-1之间
    dim_data = orig_top10[dimensions].copy()
    for dim in dimensions:
        if dim == 'PM2.5浓度':  # PM2.5浓度是逆向指标
            dim_data[dim] = (dim_data[dim] - dim_data[dim].min()) / (dim_data[dim].max() - dim_data[dim].min())
        else:
            dim_data[dim] = (dim_data[dim] - dim_data[dim].min()) / (dim_data[dim].max() - dim_data[dim].min())
    
    # 准备雷达图数据
    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # 绘制每个城市的雷达图
    for i, city in enumerate(orig_top10['城市']):
        values = dim_data.iloc[i].tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, linewidth=1, label=city)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置雷达图的标签
    plt.xticks(angles[:-1], dimensions, size=10)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
    plt.ylim(0, 1)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('前10名城市在各维度的表现')
    
    # 保存雷达图
    plt.savefig(os.path.join(result_dir, 'problem2_top10_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return top_cities

# 6. 保存结果
def save_results(top_cities, df_scores):
    """保存评价结果"""
    # 保存前50个城市的详细信息
    top_cities.to_csv(os.path.join(result_dir, 'problem2_top50_cities.csv'), index=False, encoding='utf-8-sig')
    
    # 保存所有城市的综合得分
    df_scores[['城市', '综合得分']].to_csv(os.path.join(result_dir, 'problem2_all_cities_scores.csv'), index=False, encoding='utf-8-sig')
    
    # 保存文本结果
    with open(os.path.join(result_dir, 'problem2_results.txt'), 'w', encoding='utf-8') as f:
        f.write("问题2: 最令外国游客向往的50个城市\n\n")
        for i, (index, row) in enumerate(top_cities.iterrows(), 1):
            f.write(f"{i}. {row['城市']}: {row['综合得分']:.4f}\n")

def sensitivity_analysis(df_normalized, top_cities):
    """进行敏感性分析，通过小幅调整准则层权重"""
    print("\n===== 敏感性分析 =====")
    
    # 原始准则层权重
    original_matrices = construct_ahp_matrices()
    original_criterion_weights, _, _, _ = calculate_weights(original_matrices['criterion'])
    
    # 调整准则层权重的幅度
    adjustment = 0.1
    
    # 定义要调整权重的准则
    criteria = ['景点吸引力', '人文底蕴', '交通便利']
    
    # 原始前10名城市
    original_top10 = top_cities.head(10)['城市'].tolist()
    print(f"原始前10名城市: {', '.join(original_top10)}")
    
    # 分别调整每个准则的权重
    for criterion_idx, criterion_name in enumerate(['景点吸引力', '城市规模', '环境环保', '人文底蕴', '交通便利', '经济发展水平']):
        if criterion_name not in criteria:
            continue
            
        # 增加权重
        increased_weights = original_criterion_weights.copy()
        increased_weights[criterion_idx] += adjustment
        # 归一化
        increased_weights = increased_weights / np.sum(increased_weights)
        
        # 将修改后的权重应用于数据
        # 为简化分析，我们只考虑准则层权重的变化，指标层权重保持不变
        increased_scores = calculate_adjusted_scores(df_normalized, increased_weights, criterion_idx)
        increased_top10 = increased_scores.head(10)['城市'].tolist()
        
        # 计算排名变化率
        changes = sum(1 for city in increased_top10 if city not in original_top10)
        change_rate = changes / 10 * 100
        
        print(f"\n增加{criterion_name}权重 {adjustment:.1f} 后:")
        print(f"新的前10名城市: {', '.join(increased_top10)}")
        print(f"排名变化: {changes}个城市, 变化率: {change_rate:.1f}%")
    
    return "敏感性分析完成"

def calculate_adjusted_scores(df_normalized, adjusted_weights, adjusted_idx):
    """使用调整后的权重计算城市得分"""
    # 简化的得分计算，仅用于敏感性分析
    # 实际应用中应重新计算组合权重
    
    # 初始化得分DataFrame
    df_scores = df_normalized[['城市']].copy()
    
    # 定义各维度对应的指标
    dimensions = {
        0: ['最高景点评分'],
        1: ['常住人口(万人)', '城镇化率(%)'],
        2: ['PM2.5浓度', '空气质量优良天数比例(%)', '污水处理率(%)', '垃圾处理率(%)'],
        3: ['图书馆藏书量(万册)', '博物馆数量'],
        4: ['公共交通客运总量(万人次)', '客运量(万人次)'],
        5: ['地区生产总值(亿元)', '人均地区生产总值(元)']
    }
    
    # 计算每个维度的得分
    for dim_idx, indicators in dimensions.items():
        weight = adjusted_weights[dim_idx]
        # 简化：每个指标在其维度内权重相等
        indicator_weight = weight / len(indicators)
        
        for indicator in indicators:
            if indicator in df_normalized.columns:
                df_scores[f'{indicator}_得分'] = df_normalized[indicator] * indicator_weight
    
    # 计算综合得分
    score_columns = [col for col in df_scores.columns if col.endswith('_得分')]
    df_scores['综合得分'] = df_scores[score_columns].sum(axis=1)
    
    # 按综合得分排序
    df_scores = df_scores.sort_values('综合得分', ascending=False)
    
    return df_scores

# 主函数
def main():
    print("开始执行问题2: 城市综合吸引力评价模型...")
    
    # 1. 收集并整合数据
    merged_data = merge_all_data()
    
    # 2. 数据清洗与标准化
    original_data, normalized_data = clean_and_normalize_data(merged_data)
    
    # 3. AHP权重计算
    matrices = construct_ahp_matrices()
    criterion_weights, indicator_matrices = print_ahp_results(matrices)
    combined_weights = calculate_combined_weights(criterion_weights, indicator_matrices)
    
    # 4. 计算城市综合得分
    city_scores = calculate_city_scores(normalized_data, combined_weights)
    
    # 5. 选择前50个城市
    top_cities = select_top_cities(city_scores)
    
    # 6. 可视化结果
    visualize_top_cities(top_cities, original_data)
    
    # 7. 保存结果
    save_results(top_cities, city_scores)
    
    # 8. 敏感性分析
    sensitivity_analysis(normalized_data, top_cities)
    
    print("\n问题2已完成！结果已保存到文件中。")

if __name__ == "__main__":
    main() 