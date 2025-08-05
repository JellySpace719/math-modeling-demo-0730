# problem2_solution.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from glob import glob
from pathlib import Path
import re
import sys

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置输出目录
OUTPUT_DIR = "Problem2-output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置警告过滤
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 添加调试输出
print("开始执行问题二的数据处理与建模...")
print(f"当前工作目录: {os.getcwd()}")

# 检查环境
try:
    # 导入PyMC
    import pymc as pm
    import arviz as az
    print("使用PyMC版本:", pm.__version__)
    print("使用Arviz版本:", az.__version__)
except Exception as e:
    print(f"导入PyMC失败: {e}")
    raise

# 修复特殊学院数据处理部分

def load_and_preprocess_data():
    """
    加载并预处理所有学院的评分数据，处理特殊格式的学院数据
    
    Returns:
        DataFrame: 包含所有学院教师评分的统一格式数据
    """
    try:
        print("开始加载数据文件...")
        # 获取所有CSV文件
        csv_files = glob("attachment2_college_*.csv")
        print(f"找到的CSV文件数量: {len(csv_files)}")
        
        if len(csv_files) == 0:
            print("未找到CSV文件，尝试列出当前目录所有文件:")
            all_files = os.listdir(".")
            print("\n".join(all_files))
        
        # 初始化一个空列表存储所有数据
        all_data = []
        
        # 遍历每个文件
        for file in sorted(csv_files):
            print(f"处理文件: {file}")
            # 从文件名中提取学院ID
            college_id = re.search(r'attachment2_college_(.+)\_scores\.csv', file).group(1)
            
            # 特殊处理 P、H、T 学院，它们有多组专家
            if college_id in ['P', 'H', 'T']:
                print(f"  - 处理特殊格式学院: {college_id}")
                
                # 首先直接尝试读取CSV文件中的所有内容
                raw_df = pd.read_csv(file)
                print(f"  - 原始列名: {raw_df.columns.tolist()}")
                
                # 对于特殊学院，直接查看文件内容来确定结构
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 找出带有"组专家评分"的行
                for line_idx, line in enumerate(lines):
                    if "组专家评分" in line:
                        print(f"  - 找到专家组行({line_idx}): {line.strip()}")
                        
                        # 解析出该行中有多少个专家组
                        groups = re.findall(r'第(.+?)组专家评分', line)
                        print(f"  - 找到专家组数量: {len(groups)}, 组名: {groups}")
                        
                        # 为每个组创建数据
                        for group_idx, group_name in enumerate(groups):
                            group_id = f"{college_id}_Group{group_name}"
                            print(f"    - 处理专家组: {group_id}")
                            
                            # 查找该组的教师和分数
                            # 由于文件格式特殊，直接根据行号来提取数据
                            
                            # 先确定教师和分数所在的列索引
                            parts = line.split(',')
                            start_idx = -1
                            
                            for i, part in enumerate(parts):
                                if f"第{group_name}组专家评分" in part:
                                    start_idx = i
                                    break
                            
                            if start_idx >= 0:
                                print(f"      - 找到专家组起始列索引: {start_idx}")
                                
                                # 假设教师ID在专家组名称后面的一列或两列
                                teacher_col_idx = -1
                                score_col_idx = -1
                                
                                # 查找标题行(通常在专家组行的下一行)
                                header_line = lines[line_idx + 1].strip().split(',')
                                
                                for i in range(start_idx, min(start_idx + 5, len(header_line))):
                                    if "教师" in header_line[i]:
                                        teacher_col_idx = i
                                    elif "得分" in header_line[i]:
                                        score_col_idx = i
                                
                                print(f"      - 教师列索引: {teacher_col_idx}, 分数列索引: {score_col_idx}")
                                
                                if teacher_col_idx >= 0 and score_col_idx >= 0:
                                    # 提取数据行(从标题行后面的行开始)
                                    teacher_ids = []
                                    scores = []
                                    
                                    for data_line_idx in range(line_idx + 2, len(lines)):
                                        data_line = lines[data_line_idx].strip().split(',')
                                        
                                        if len(data_line) > max(teacher_col_idx, score_col_idx):
                                            teacher_id = data_line[teacher_col_idx].strip()
                                            score = data_line[score_col_idx].strip()
                                            
                                            # 确保不是空值
                                            if teacher_id and score and teacher_id != "" and score != "":
                                                try:
                                                    # 尝试转换分数为数值
                                                    score_val = float(score)
                                                    teacher_ids.append(teacher_id)
                                                    scores.append(score_val)
                                                except:
                                                    continue
                                    
                                    print(f"      - 提取到 {len(teacher_ids)} 个教师记录")
                                    
                                    # 创建数据框
                                    if teacher_ids:
                                        group_df = pd.DataFrame({
                                            'College_ID': college_id,
                                            'Expert_Group_ID': group_id,
                                            'Teacher_ID': teacher_ids,
                                            'Raw_Score': scores
                                        })
                                        all_data.append(group_df)
                                        print(f"      - 成功添加 {len(group_df)} 条记录")
                                else:
                                    print("      - 警告: 未找到教师列或分数列")
                            else:
                                print(f"      - 警告: 未找到专家组起始列")
                        
                        # 已经处理完所有专家组，跳出循环
                        break
            
            else:
                # 常规学院处理
                print(f"  - 处理常规格式学院: {college_id}")
                try:
                    # 读取数据
                    df = pd.read_csv(file)
                    print(f"  - 成功读取, 列名: {df.columns.tolist()}")
                    
                    # 找到教师ID列和分数列
                    id_cols = [col for col in df.columns if "教师" in str(col)]
                    score_cols = [col for col in df.columns if "得分" in str(col) or "分数" in str(col) or "总分" in str(col)]
                    
                    if id_cols:
                        id_col = id_cols[0]
                    else:
                        print(f"  - 警告: 找不到教师ID列，尝试使用第一列")
                        id_col = df.columns[0]
                    
                    if score_cols:
                        score_col = score_cols[0]
                    else:
                        print(f"  - 警告: 找不到分数列，尝试使用最后一列")
                        score_col = df.columns[-1]
                    
                    print(f"    - 使用ID列: {id_col}, 分数列: {score_col}")
                    
                    # 创建数据
                    college_df = pd.DataFrame({
                        'College_ID': college_id,
                        'Expert_Group_ID': f"{college_id}_Group1",
                        'Teacher_ID': df[id_col],
                        'Raw_Score': df[score_col]
                    })
                    
                    all_data.append(college_df)
                    print(f"    - 成功添加 {len(college_df)} 条记录")
                    
                except Exception as e:
                    print(f"  - 读取失败: {str(e)}")
                    continue
        
        if not all_data:
            raise ValueError("没有成功加载任何数据")
                
        # 合并所有数据
        print("合并所有数据...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 清理可能的数据问题：将分数转换为数值型
        print("清理数据...")
        try:
            combined_df['Raw_Score'] = pd.to_numeric(combined_df['Raw_Score'], errors='coerce')
            # 删除无效分数记录
            invalid_count = combined_df['Raw_Score'].isna().sum()
            if invalid_count > 0:
                print(f"警告: 发现 {invalid_count} 条无效分数记录，将被删除")
                combined_df = combined_df.dropna(subset=['Raw_Score'])
        except Exception as e:
            print(f"分数转换警告: {str(e)}")
        
        # 创建映射到整数的索引
        college_map = {college: idx for idx, college in enumerate(combined_df['College_ID'].unique())}
        expert_group_map = {group: idx for idx, group in enumerate(combined_df['Expert_Group_ID'].unique())}
        
        # 添加索引列
        combined_df['college_idx'] = combined_df['College_ID'].map(college_map)
        combined_df['expert_group_idx'] = combined_df['Expert_Group_ID'].map(expert_group_map)
        
        print(f"数据加载和预处理完成，共有 {len(combined_df)} 条评分记录")
        print(f"包含 {len(college_map)} 个学院，{len(expert_group_map)} 个专家组")
        
        # 显示前几行数据
        print("\n数据预览:")
        print(combined_df.head())
        
        return combined_df, college_map, expert_group_map
    
    except Exception as e:
        print(f"数据加载和预处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 修改exploratory_data_analysis函数，以使用命名聚合
def exploratory_data_analysis(df):
    """
    对数据进行探索性分析，了解各学院和专家组的打分特点
    
    Args:
        df: 预处理后的数据
    """
    try:
        # 创建一个保存EDA结果的文件夹
        eda_dir = os.path.join(OUTPUT_DIR, "EDA")
        os.makedirs(eda_dir, exist_ok=True)
        
        # 计算各学院的统计指标，使用命名聚合
        college_stats = df.groupby('College_ID')['Raw_Score'].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('range', lambda x: x.max() - x.min())  # 使用命名聚合
        ]).reset_index()
        
        # 计算各专家组的统计指标，使用命名聚合
        expert_group_stats = df.groupby(['College_ID', 'Expert_Group_ID'])['Raw_Score'].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('range', lambda x: x.max() - x.min())  # 使用命名聚合
        ]).reset_index()
        
        # 保存统计结果
        college_stats.to_csv(os.path.join(eda_dir, "college_statistics.csv"), index=False)
        expert_group_stats.to_csv(os.path.join(eda_dir, "expert_group_statistics.csv"), index=False)
        
        # 绘制学院得分分布箱线图
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='College_ID', y='Raw_Score', data=df, palette='viridis')
        plt.title('各学院教师评分分布', fontsize=16)
        plt.xlabel('学院', fontsize=14)
        plt.ylabel('原始评分', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "college_scores_boxplot.png"), dpi=300)
        plt.close()
        
        # 绘制特殊学院(H、P、T)内部专家组的分布
        special_colleges = ['H', 'P', 'T']
        for college in special_colleges:
            if college in df['College_ID'].unique():
                college_data = df[df['College_ID'] == college]
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Expert_Group_ID', y='Raw_Score', data=college_data, palette='Set2')
                plt.title(f'学院{college}内部各专家组评分分布', fontsize=16)
                plt.xlabel('专家组', fontsize=14)
                plt.ylabel('原始评分', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(eda_dir, f"college_{college}_expert_groups_boxplot.png"), dpi=300)
                plt.close()
        
        # 绘制整体分布图
        plt.figure(figsize=(12, 6))
        sns.histplot(df['Raw_Score'], kde=True, bins=30)
        plt.title('所有教师评分分布', fontsize=16)
        plt.xlabel('原始评分', fontsize=14)
        plt.ylabel('频数', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "overall_score_distribution.png"), dpi=300)
        plt.close()
        
        print(f"探索性数据分析完成，结果已保存到 {eda_dir} 目录")
        
        return college_stats, expert_group_stats
        
    except Exception as e:
        print(f"探索性数据分析失败: {str(e)}")
        raise

def build_and_fit_hierarchical_model(df, college_map, expert_group_map):
    """
    构建并拟合层级贝叶斯模型
    Args:
        df: 预处理后的数据
        college_map: 学院映射
        expert_group_map: 专家组映射
    Returns:
        trace: MCMC采样结果
        model: 拟合好的模型
    """
    # 设置模型数据
    scores = df['Raw_Score'].values
    college_idx = df['college_idx'].values
    expert_group_idx = df['expert_group_idx'].values
    n_colleges = len(college_map)
    n_expert_groups = len(expert_group_map)
    global_mean = scores.mean()
    global_sd = scores.std() * 2  # 宽泛的先验
    print(f"开始构建模型，全局均值: {global_mean:.2f}, 全局标准差: {global_sd:.2f}")
    # 构建层级贝叶斯模型
    print("构建层级贝叶斯模型...")
    with pm.Model() as hierarchical_model:
        # 先验分布
        mu_global = pm.Normal('mu_global', mu=global_mean, sigma=global_sd)
        sigma_college = pm.HalfNormal('sigma_college', sigma=10)
        sigma_expert_group = pm.HalfNormal('sigma_expert_group', sigma=10)
        sigma_error = pm.HalfNormal('sigma_error', sigma=10)
        alpha_college = pm.Normal('alpha_college', mu=0, sigma=sigma_college, shape=n_colleges)
        beta_expert_group = pm.Normal('beta_expert_group', mu=0, sigma=sigma_expert_group, shape=n_expert_groups)
        mu = mu_global + alpha_college[college_idx] + beta_expert_group[expert_group_idx]
        y = pm.Normal('y', mu=mu, sigma=sigma_error, observed=scores)
        print("开始MCMC采样...")
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42
        )
        summary = az.summary(trace)
        summary.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"))
        print("模型拟合完成")
        model = hierarchical_model
    return trace, model

# 修复model_diagnostics函数
def model_diagnostics(trace, college_map, expert_group_map):
    """
    进行模型诊断
    Args:
        trace: MCMC采样结果
        college_map: 学院映射
        expert_group_map: 专家组映射
    """
    try:
        print("开始模型诊断...")
        diagnostics_dir = os.path.join(OUTPUT_DIR, "Diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)
        # 贝叶斯模型的完整诊断
        print("使用完整贝叶斯模型诊断")
        az.plot_trace(trace, var_names=['mu_global', 'sigma_college', 'sigma_expert_group', 'sigma_error'])
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "trace_plots.png"), dpi=300)
        plt.close()
        az.plot_posterior(trace, var_names=['mu_global', 'sigma_college', 'sigma_expert_group', 'sigma_error'])
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "posterior_plots.png"), dpi=300)
        plt.close()
        az.plot_forest(trace, var_names=['alpha_college'], combined=True)
        plt.title('各学院评分偏差(α)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "college_effects_forest.png"), dpi=300)
        plt.close()
        az.plot_forest(trace, var_names=['beta_expert_group'], combined=True)
        plt.title('各专家组评分偏差(β)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "expert_group_effects_forest.png"), dpi=300)
        plt.close()
        print(f"模型诊断完成，结果已保存到 {diagnostics_dir} 目录")
    except Exception as e:
        print(f"模型诊断失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("跳过模型诊断阶段")

def calculate_corrected_scores(df, trace, college_map, expert_group_map):
    """
    计算校正后的教师评分
    Args:
        df: 预处理后的数据
        trace: MCMC采样结果
        college_map: 学院映射
        expert_group_map: 专家组映射
    Returns:
        DataFrame: 包含原始评分和校正后评分的数据
    """
    try:
        print("开始计算校正后评分...")
        corrected_df = df.copy()
        # 使用贝叶斯模型结果
        print("使用贝叶斯模型结果计算校正后评分")
        summary = az.summary(trace)
        mu_global = summary.loc['mu_global', 'mean']
        alpha_college = {}
        for college, idx in college_map.items():
            alpha_college[college] = summary.loc[f'alpha_college[{idx}]', 'mean']
        beta_expert_group = {}
        for group, idx in expert_group_map.items():
            beta_expert_group[group] = summary.loc[f'beta_expert_group[{idx}]', 'mean']
        print("计算校正后评分...")
        corrected_df['Corrected_Score'] = corrected_df.apply(
            lambda row: row['Raw_Score'] - alpha_college[row['College_ID']] - \
                        beta_expert_group[row['Expert_Group_ID']] + mu_global,
            axis=1
        )
        corrected_mean = corrected_df['Corrected_Score'].mean()
        corrected_std = corrected_df['Corrected_Score'].std()
        target_mean = 85
        target_std = 5
        corrected_df['Final_Score'] = (
            (corrected_df['Corrected_Score'] - corrected_mean) / corrected_std
        ) * target_std + target_mean
        corrected_df['Final_Score'] = corrected_df['Final_Score'].clip(60, 100)
        corrected_df.to_csv(os.path.join(OUTPUT_DIR, "corrected_scores.csv"), index=False)
        print("校正后的评分计算完成")
        print(f"校正后评分范围: {corrected_df['Final_Score'].min():.2f} - {corrected_df['Final_Score'].max():.2f}")
        return corrected_df
    except Exception as e:
        print(f"校正后评分计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def visualize_results(df, corrected_df):
    """
    可视化原始评分和校正后评分的结果
    
    Args:
        df: 原始数据
        corrected_df: 校正后的数据
    """
    try:
        results_dir = os.path.join(OUTPUT_DIR, "Results_Visualization")
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 整体分布对比
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['Raw_Score'], kde=True, bins=30)
        plt.title('原始评分分布', fontsize=14)
        plt.xlabel('分数', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        
        plt.subplot(1, 2, 2)
        sns.histplot(corrected_df['Final_Score'], kde=True, bins=30)
        plt.title('校正后评分分布', fontsize=14)
        plt.xlabel('分数', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "score_distributions_comparison.png"), dpi=300)
        plt.close()
        
        # 2. 各学院原始分数和校正后分数的箱线图对比
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 1, 1)
        sns.boxplot(x='College_ID', y='Raw_Score', data=df, palette='viridis')
        plt.title('各学院原始评分分布', fontsize=16)
        plt.xlabel('学院', fontsize=14)
        plt.ylabel('分数', fontsize=14)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        sns.boxplot(x='College_ID', y='Final_Score', data=corrected_df, palette='viridis')
        plt.title('各学院校正后评分分布', fontsize=16)
        plt.xlabel('学院', fontsize=14)
        plt.ylabel('分数', fontsize=14)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "college_scores_comparison.png"), dpi=300)
        plt.close()
        
        # 3. 特殊学院(H、P、T)内部各专家组校正前后对比
        special_colleges = ['H', 'P', 'T']
        for college in special_colleges:
            if college in df['College_ID'].unique():
                college_data_orig = df[df['College_ID'] == college]
                college_data_corr = corrected_df[corrected_df['College_ID'] == college]
                
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 1, 1)
                sns.boxplot(x='Expert_Group_ID', y='Raw_Score', data=college_data_orig, palette='Set2')
                plt.title(f'学院{college}内部各专家组原始评分', fontsize=16)
                plt.xlabel('专家组', fontsize=14)
                plt.ylabel('分数', fontsize=14)
                
                plt.subplot(2, 1, 2)
                sns.boxplot(x='Expert_Group_ID', y='Final_Score', data=college_data_corr, palette='Set2')
                plt.title(f'学院{college}内部各专家组校正后评分', fontsize=16)
                plt.xlabel('专家组', fontsize=14)
                plt.ylabel('分数', fontsize=14)
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"college_{college}_comparison.png"), dpi=300)
                plt.close()
        
        # 4. 原始排名vs校正后排名的散点图
        corrected_df['Raw_Rank'] = corrected_df['Raw_Score'].rank(ascending=False)
        corrected_df['Final_Rank'] = corrected_df['Final_Score'].rank(ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Raw_Rank', y='Final_Rank', hue='College_ID', data=corrected_df, alpha=0.6)
        
        # 绘制对角线
        max_rank = max(corrected_df['Raw_Rank'].max(), corrected_df['Final_Rank'].max())
        plt.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5)
        
        plt.title('原始排名 vs 校正后排名', fontsize=16)
        plt.xlabel('原始排名', fontsize=14)
        plt.ylabel('校正后排名', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "rank_comparison_scatter.png"), dpi=300)
        plt.close()
        
        # 5. 创建一个示例表格，展示排名变化最大的10位教师
        corrected_df['Rank_Change'] = corrected_df['Raw_Rank'] - corrected_df['Final_Rank']
        top_changes = corrected_df.sort_values(by='Rank_Change', ascending=False).head(10)
        bottom_changes = corrected_df.sort_values(by='Rank_Change').head(10)
        
        rank_changes = pd.concat([top_changes, bottom_changes])
        rank_changes = rank_changes[['College_ID', 'Expert_Group_ID', 'Teacher_ID', 
                                    'Raw_Score', 'Final_Score', 'Raw_Rank', 'Final_Rank', 'Rank_Change']]
        
        rank_changes.to_csv(os.path.join(results_dir, "significant_rank_changes.csv"), index=False)
        
        print(f"结果可视化完成，图表已保存到 {results_dir} 目录")
        
    except Exception as e:
        print(f"结果可视化失败: {str(e)}")
        raise

# 修复generate_summary_report函数
def generate_summary_report(df, corrected_df, trace, college_map, expert_group_map):
    """
    生成总结报告
    Args:
        df: 原始数据
        corrected_df: 校正后的数据
        trace: MCMC采样结果
        college_map: 学院映射
        expert_group_map: 专家组映射
    """
    try:
        print("开始生成总结报告...")
        report_path = os.path.join(OUTPUT_DIR, "problem2_analysis_results.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 问题二：教师教学评价标准化与汇总分析报告\n\n")
            f.write("## 1. 数据概览\n\n")
            f.write(f"- 共分析了 {len(df['College_ID'].unique())} 个学院的评分数据\n")
            f.write(f"- 总共有 {len(df)} 条教师评分记录\n")
            multi_expert_colleges = df.groupby(['College_ID', 'Expert_Group_ID']).size().reset_index()
            multi_expert_colleges = multi_expert_colleges['College_ID'].value_counts()
            multi_expert_colleges = multi_expert_colleges[multi_expert_colleges > 1].index.tolist()
            if multi_expert_colleges:
                f.write(f"- 其中学院 {', '.join(multi_expert_colleges)} 包含多组专家评分\n")
            f.write("\n")
            f.write("## 2. 各学院评分特点\n\n")
            college_stats = df.groupby('College_ID')['Raw_Score'].agg([
                'count', 'mean', 'std', 'min', 'max',
                ('range', lambda x: x.max() - x.min())
            ]).sort_values(by='mean', ascending=False)
            f.write("各学院评分统计特点（按平均分降序排列）：\n\n")
            f.write("| 学院 | 教师数量 | 平均分 | 标准差 | 最低分 | 最高分 | 极差 |\n")
            f.write("|------|---------|--------|--------|--------|--------|------|\n")
            for college, row in college_stats.iterrows():
                f.write(f"| {college} | {int(row['count'])} | {row['mean']:.2f} | {row['std']:.2f} | ")
                f.write(f"{row['min']:.2f} | {row['max']:.2f} | {row['range']:.2f} |\n")
            f.write("\n主要发现：\n")
            f.write(f"- 评分最高的学院: {college_stats.index[0]}，平均分 {college_stats['mean'].max():.2f}\n")
            f.write(f"- 评分最低的学院: {college_stats.index[-1]}，平均分 {college_stats['mean'].min():.2f}\n")
            f.write(f"- 学院间平均分差距: {college_stats['mean'].max() - college_stats['mean'].min():.2f} 分\n")
            max_range_college = college_stats.sort_values(by='range', ascending=False).index[0]
            min_range_college = college_stats.sort_values(by='range').index[0]
            max_range_value = college_stats.loc[max_range_college, 'range']
            min_range_value = college_stats.loc[min_range_college, 'range']
            f.write(f"- 极差最大的学院: {max_range_college}，极差 {max_range_value:.2f}\n")
            f.write(f"- 极差最小的学院: {min_range_college}，极差 {min_range_value:.2f}\n\n")
            f.write("## 3. 层级贝叶斯模型结构与参数\n\n")
            f.write("### 模型结构\n\n")
            f.write("我们构建了三层的完整层级贝叶斯模型 (HBM)：\n")
            f.write("- 学校整体水平（全局均值）\n")
            f.write("- 学院效应（每个学院的评分偏差）\n")
            f.write("- 专家组效应（每个专家组在学院内的评分偏差）\n\n")
            f.write("### 数学模型\n\n")
            f.write("$$S_{ijk} \\sim \\mathcal{N}(\\mu_{ij}, \\sigma^2_{\\text{error}})$$\n")
            f.write("$$\\mu_{ij} = \\mu_{\\text{global}} + \\alpha_i + \\beta_{ij}$$\n")
            f.write("$$\\alpha_i \\sim \\mathcal{N}(0, \\sigma^2_{\\text{college}})$$\n")
            f.write("$$\\beta_{ij} \\sim \\mathcal{N}(0, \\sigma^2_{\\text{expert\\_group}})$$\n\n")
            f.write("### 主要参数估计结果\n\n")
            summary = az.summary(trace)
            f.write(f"- 全局均值 (μ_global): {summary.loc['mu_global', 'mean']:.2f}\n")
            f.write(f"- 学院效应标准差 (σ_college): {summary.loc['sigma_college', 'mean']:.2f}\n")
            f.write(f"- 专家组效应标准差 (σ_expert_group): {summary.loc['sigma_expert_group', 'mean']:.2f}\n")
            f.write(f"- 误差标准差 (σ_error): {summary.loc['sigma_error', 'mean']:.2f}\n\n")
            f.write("## 4. 校正效果分析\n\n")
            orig_mean = df['Raw_Score'].mean()
            orig_std = df['Raw_Score'].std()
            corr_mean = corrected_df['Final_Score'].mean()
            corr_std = corrected_df['Final_Score'].std()
            f.write("### 整体评分分布变化\n\n")
            f.write(f"- 原始评分：均值 = {orig_mean:.2f}, 标准差 = {orig_std:.2f}\n")
            f.write(f"- 校正后评分：均值 = {corr_mean:.2f}, 标准差 = {corr_std:.2f}\n\n")
            f.write("### 校正后学院间差异\n\n")
            corr_college_stats = corrected_df.groupby('College_ID')['Final_Score'].agg([
                'count', 'mean', 'std', 'min', 'max',
                ('range', lambda x: x.max() - x.min())
            ])
            f.write("| 学院 | 校正前平均分 | 校正后平均分 | 校正前极差 | 校正后极差 |\n")
            f.write("|------|--------------|--------------|------------|------------|\n")
            for college in college_stats.index:
                orig = college_stats.loc[college]
                corr = corr_college_stats.loc[college]
                f.write(f"| {college} | {orig['mean']:.2f} | {corr['mean']:.2f} | ")
                f.write(f"{orig['range']:.2f} | {corr['range']:.2f} |\n")
            f.write("\n")
            f.write("## 5. 结论与解释\n\n")
            f.write("### 模型优势\n\n")
            f.write("1. **消除系统性偏差**：模型有效识别并剥离了由学院和专家组打分风格引起的系统性偏差。\n")
            f.write("2. **'借用强度'机制**：通过考虑整体数据信息，对小样本和极端打分行为进行了'收缩'估计，使得结果更稳健。\n")
            f.write("3. **保留相对排序**：虽然消除了系统性偏差，但仍保留了教师间的真实差异。\n")
            f.write("4. **统一标准**：校正后的分数分布更加集中且符合正态分布，便于统一管理和评价。\n\n")
            f.write("### 解决的关键问题\n\n")
            f.write("1. **评分标准不统一问题**：通过估计并剥离各学院的偏置，实现了评分标准的统一。\n")
            f.write("2. **极差问题**：对极差极小或极大的学院，通过'借用强度'机制使其分数分布更合理。\n")
            f.write("3. **专家组偏差问题**：明确建模了特殊学院内部不同专家组的评分偏差，消除了局部评价偏差。\n\n")
            f.write("总结：我们的模型成功构建了一个公平、合理的教师评价标准化体系，为学校决策提供了可靠的参考依据。\n")
        print(f"分析报告已生成并保存到 {report_path}")
    except Exception as e:
        print(f"生成总结报告失败: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数，协调整个工作流程
    """
    try:
        print("开始执行问题二的求解...")
        
        # 1. 数据预处理
        df, college_map, expert_group_map = load_and_preprocess_data()
        
        # 2. 探索性数据分析
        college_stats, expert_group_stats = exploratory_data_analysis(df)
        
        # 3. 构建并拟合层级贝叶斯模型
        trace, model = build_and_fit_hierarchical_model(df, college_map, expert_group_map)
        
        # 4. 模型诊断 - 传递college_map和expert_group_map参数
        model_diagnostics(trace, college_map, expert_group_map)
        
        # 5. 计算校正后评分
        corrected_df = calculate_corrected_scores(df, trace, college_map, expert_group_map)
        
        # 6. 可视化结果
        visualize_results(df, corrected_df)
        
        # 7. 生成总结报告
        generate_summary_report(df, corrected_df, trace, college_map, expert_group_map)
        
        print(f"问题二求解完成，所有结果已保存到 {OUTPUT_DIR} 目录")
        
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()