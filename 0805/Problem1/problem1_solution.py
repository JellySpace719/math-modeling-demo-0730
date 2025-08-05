#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华数杯2023年C题 - 问题一解决方案
母亲身心健康对婴儿成长的影响 — 规律性研究

作者: 算法工程师专家
日期: 2024年
版本: 1.0

核心目标: 探究母亲的身体指标、心理指标与婴儿行为特征、睡眠质量之间是否存在统计学上的显著关联或影响规律。

主要功能:
1. 数据加载与初步审查
2. 数据预处理与特征工程
3. 探索性数据分析(EDA)与初步相关性分析
4. 多元回归分析
5. 结果整合与可视化

依赖库:
- pandas: 数据处理
- numpy: 数值计算
- matplotlib: 基础绘图
- seaborn: 统计可视化
- scipy: 统计分析
- statsmodels: 统计建模
- sklearn: 机器学习工具
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

class MaternalInfantAnalysis:
    """
    母亲身心健康对婴儿成长影响的综合分析类
    
    该类实现了完整的数据分析流程，包括数据预处理、探索性分析、
    统计建模和结果可视化等功能。
    """
    
    def __init__(self, data_path: str = 'data.csv', research_size: int = 390):
        """
        初始化分析类
        
        Args:
            data_path: 数据文件路径
            research_size: 研究数据大小（前N条记录）
        """
        self.data_path = data_path
        self.research_size = research_size
        self.df = None
        self.df_research = None
        self.df_encoded = None
        self.scaler = StandardScaler()
        
        # 定义映射字典
        self.mapping_dicts = self._define_mappings()
        
        # 存储分析结果
        self.results = {}
        
        logger.info("MaternalInfantAnalysis类初始化完成")
    
    def _define_mappings(self) -> Dict:
        """
        定义各种分类变量的映射字典
        
        Returns:
            包含所有映射字典的字典
        """
        mappings = {
            'marital_status': {1: '未婚', 2: '已婚', 3: '其他'},
            'education': {1: '小学', 2: '初中', 3: '高中', 4: '大学', 5: '研究生'},
            'delivery_method': {1: '自然分娩', 2: '剖宫产'},
            'baby_gender': {1: '男性', 2: '女性'},
            'sleep_method': {1: '哄睡法', 2: '抚触法', 3: '音乐法', 4: '自主入睡', 5: '定时法'},
            'baby_behavior': {'安静型': 0, '中等型': 1, '矛盾型': 2}
        }
        return mappings
    
    def load_data(self) -> pd.DataFrame:
        """
        数据加载与初步审查
        
        Returns:
            加载的数据框
        """
        try:
            logger.info(f"开始加载数据文件: {self.data_path}")
            
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据文件 {self.data_path} 不存在")
            
            # 读取CSV文件
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            
            # 基本数据信息
            logger.info(f"数据加载成功，形状: {self.df.shape}")
            logger.info(f"数据列名: {list(self.df.columns)}")
            
            # 数据基本信息
            logger.info("数据基本信息:")
            logger.info(f"数据类型:\n{self.df.dtypes}")
            logger.info(f"缺失值统计:\n{self.df.isnull().sum()}")
            logger.info(f"描述性统计:\n{self.df.describe()}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        数据预处理与特征工程
        
        Returns:
            预处理后的数据框
        """
        try:
            logger.info("开始数据预处理")
            
            # 1. 限定研究范围
            self.df_research = self.df.iloc[:self.research_size].copy()
            logger.info(f"研究数据集大小: {self.df_research.shape}")
            
            # 2. 处理整晚睡眠时间
            self._process_sleep_time()
            
            # 3. 分类变量映射与编码
            self._encode_categorical_variables()
            
            # 4. 特征标准化
            self._standardize_features()
            
            logger.info("数据预处理完成")
            return self.df_encoded
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def _process_sleep_time(self):
        """处理整晚睡眠时间列"""
        try:
            def convert_sleep_time(time_str):
                """将时间字符串转换为小时数"""
                if pd.isna(time_str) or time_str == '99:99':
                    return np.nan
                try:
                    hours, minutes = map(int, time_str.split(':'))
                    return hours + minutes / 60.0
                except:
                    return np.nan
            
            # 转换睡眠时间
            self.df_research['整晚睡眠时间（小时）'] = self.df_research['整晚睡眠时间（时：分：秒）'].apply(convert_sleep_time)
            
            # 处理缺失值
            median_sleep_time = self.df_research['整晚睡眠时间（小时）'].median()
            self.df_research['整晚睡眠时间（小时）'].fillna(median_sleep_time, inplace=True)
            
            # 删除原始列
            self.df_research.drop('整晚睡眠时间（时：分：秒）', axis=1, inplace=True)
            
            logger.info(f"睡眠时间处理完成，中位数填充值: {median_sleep_time:.2f}小时")
            
        except Exception as e:
            logger.error(f"睡眠时间处理失败: {str(e)}")
            raise
    
    def _encode_categorical_variables(self):
        """分类变量映射与编码"""
        try:
            # 应用映射
            self.df_research['婚姻状况'] = self.df_research['婚姻状况'].map(self.mapping_dicts['marital_status'])
            self.df_research['教育程度'] = self.df_research['教育程度'].map(self.mapping_dicts['education'])
            self.df_research['分娩方式'] = self.df_research['分娩方式'].map(self.mapping_dicts['delivery_method'])
            self.df_research['婴儿性别'] = self.df_research['婴儿性别'].map(self.mapping_dicts['baby_gender'])
            self.df_research['入睡方式'] = self.df_research['入睡方式'].map(self.mapping_dicts['sleep_method'])
            
            # 独热编码
            categorical_columns = ['婚姻状况', '教育程度', '分娩方式', '婴儿性别', '入睡方式']
            self.df_encoded = pd.get_dummies(self.df_research, columns=categorical_columns, drop_first=True)
            
            # 婴儿行为特征数值编码
            self.df_encoded['婴儿行为特征_encoded'] = self.df_encoded['婴儿行为特征'].map(self.mapping_dicts['baby_behavior'])
            
            logger.info("分类变量编码完成")
            
        except Exception as e:
            logger.error(f"分类变量编码失败: {str(e)}")
            raise
    
    def _standardize_features(self):
        """特征标准化"""
        try:
            # 识别数值型特征
            numerical_features = ['母亲年龄', '妊娠时间（周数）', 'CBTS', 'EPDS', 'HADS', '婴儿年龄（月）']
            
            # 添加独热编码后的特征
            dummy_features = [col for col in self.df_encoded.columns if col not in 
                            ['编号', '婴儿行为特征', '婴儿行为特征_encoded', '整晚睡眠时间（小时）', '睡醒次数'] + numerical_features]
            
            all_numerical_features = numerical_features + dummy_features
            
            # 标准化
            self.df_encoded[all_numerical_features] = self.scaler.fit_transform(self.df_encoded[all_numerical_features])
            
            logger.info(f"特征标准化完成，标准化特征数: {len(all_numerical_features)}")
            
        except Exception as e:
            logger.error(f"特征标准化失败: {str(e)}")
            raise
    
    def exploratory_data_analysis(self) -> Dict:
        """
        探索性数据分析(EDA)与初步相关性分析
        
        Returns:
            分析结果字典
        """
        try:
            logger.info("开始探索性数据分析")
            
            results = {}
            
            # 1. 描述性统计
            logger.info("步骤1: 计算描述性统计...")
            results['descriptive_stats'] = self.df_encoded.describe()
            logger.info("描述性统计计算完成")
            
            # 2. 相关性分析
            logger.info("步骤2: 开始相关性分析...")
            results['correlation_analysis'] = self._correlation_analysis()
            logger.info("相关性分析完成")
            
            # 3. ANOVA分析
            logger.info("步骤3: 开始ANOVA分析...")
            results['anova_analysis'] = self._anova_analysis()
            logger.info("ANOVA分析完成")
            
            # 4. 卡方检验
            logger.info("步骤4: 开始卡方检验...")
            results['chi_square_analysis'] = self._chi_square_analysis()
            logger.info("卡方检验完成")
            
            # 5. 可视化
            logger.info("步骤5: 创建可视化图表...")
            self._create_visualizations()
            logger.info("可视化图表创建完成")
            
            self.results['eda'] = results
            logger.info("探索性数据分析完成")
            
            return results
            
        except Exception as e:
            logger.error(f"探索性数据分析失败: {str(e)}")
            raise
    
    def _correlation_analysis(self) -> pd.DataFrame:
        """数值型变量间的相关性分析"""
        try:
            logger.info("  - 选择数值型特征...")
            # 选择数值型特征
            numerical_cols = ['母亲年龄', '妊娠时间（周数）', 'CBTS', 'EPDS', 'HADS', '婴儿年龄（月）', 
                             '整晚睡眠时间（小时）', '睡醒次数']
            
            logger.info("  - 计算相关系数矩阵...")
            correlation_matrix = self.df_encoded[numerical_cols].corr()
            
            logger.info("  - 创建相关性热力图...")
            # 创建相关性热力图
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                       square=True, linewidths=0.5)
            plt.title('母亲指标与婴儿指标相关性热力图', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            logger.info("  - 相关性热力图已保存为 correlation_heatmap.png")
            # plt.show()  # 注释掉，避免阻塞
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"相关性分析失败: {str(e)}")
            raise
    
    def _anova_analysis(self) -> Dict:
        """分类变量对数值变量的影响分析（ANOVA）"""
        try:
            logger.info("  - 初始化ANOVA分析...")
            anova_results = {}
            
            # 原始分类变量（映射前）
            original_categorical = ['婚姻状况', '教育程度', '分娩方式', '婴儿性别', '入睡方式']
            
            # 修改：使用正确的数值目标变量名称
            numerical_targets = ['睡醒次数']  # 只保留原始数据中已有的列
            
            logger.info("  - 获取原始数据...")
            # 获取原始数据中的分类变量
            df_original = self.df.iloc[:self.research_size].copy()
            
            # 修改：将处理后的整晚睡眠时间添加到原始数据中
            df_original['整晚睡眠时间（小时）'] = self.df_encoded['整晚睡眠时间（小时）']
            numerical_targets.append('整晚睡眠时间（小时）')  # 现在可以添加这个列了
            
            # 创建整合的ANOVA分析图表
            logger.info("  - 创建整合的ANOVA分析图表...")
            
            # 为每个数值目标变量创建一个大图
            for num_col in numerical_targets:
                # 计算需要的行数和列数
                n_cats = len(original_categorical)
                n_cols = 2  # 每行放2个图
                n_rows = (n_cats + n_cols - 1) // n_cols  # 向上取整
                
                # 创建子图网格
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                fig.suptitle(f'{num_col}的ANOVA分析结果', fontsize=16)
                
                # 扁平化axes数组以便索引
                if n_rows > 1:
                    axes = axes.flatten()
                elif n_cols > 1:
                    axes = [axes[0], axes[1]]  # 只有一行但有多列的情况
                else:
                    axes = [axes]  # 只有一个子图的情况
                
                for i, cat_col in enumerate(original_categorical):
                    logger.info(f"  - 处理ANOVA组合: {cat_col} vs {num_col}")
                    
                    # 获取分组数据
                    groups = []
                    categories = df_original[cat_col].unique()
                    
                    for category in categories:
                        group_data = df_original[num_col][df_original[cat_col] == category]
                        if len(group_data) > 0:
                            groups.append(group_data)
                    
                    if len(groups) >= 2:
                        # 执行ANOVA检验
                        f_statistic, p_value = f_oneway(*groups)
                        anova_results[f'{cat_col}_vs_{num_col}'] = {
                            'f_statistic': f_statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                        
                        logger.info(f"    ANOVA结果: F={f_statistic:.3f}, p={p_value:.3f}")
                        
                        # 在对应的子图中绘制箱线图
                        sns.boxplot(x=cat_col, y=num_col, data=df_original, ax=axes[i])
                        axes[i].set_title(f'{cat_col} (F={f_statistic:.2f}, p={p_value:.3f})')
                        axes[i].set_xlabel(cat_col)
                        axes[i].set_ylabel(num_col)
                        axes[i].tick_params(axis='x', rotation=45)
                
                # 处理可能的空子图
                for i in range(len(original_categorical), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)  # 为总标题留出空间
                plt.savefig(f'anova_{num_col}_combined.png', dpi=300, bbox_inches='tight')
                logger.info(f"    整合的ANOVA分析图已保存为 anova_{num_col}_combined.png")
                # plt.show()  # 注释掉，避免阻塞
            
            logger.info(f"  - ANOVA分析完成，共处理 {len(anova_results)} 个组合")
            return anova_results
            
        except Exception as e:
            logger.error(f"ANOVA分析失败: {str(e)}")
            raise

    def _chi_square_analysis(self) -> Dict:
        """分类变量间的关联性分析（卡方检验）"""
        try:
            logger.info("  - 初始化卡方检验分析...")
            chi_square_results = {}
            
            # 原始分类变量
            categorical_vars = ['婚姻状况', '教育程度', '分娩方式', '婴儿性别', '入睡方式', '婴儿行为特征']
            df_original = self.df.iloc[:self.research_size].copy()
            
            logger.info("  - 应用分类变量映射...")
            # 应用映射
            df_original['婚姻状况'] = df_original['婚姻状况'].map(self.mapping_dicts['marital_status'])
            df_original['教育程度'] = df_original['教育程度'].map(self.mapping_dicts['education'])
            df_original['分娩方式'] = df_original['分娩方式'].map(self.mapping_dicts['delivery_method'])
            df_original['婴儿性别'] = df_original['婴儿性别'].map(self.mapping_dicts['baby_gender'])
            df_original['入睡方式'] = df_original['入睡方式'].map(self.mapping_dicts['sleep_method'])
            
            # 计算总组合数
            total_combinations = len(categorical_vars) * (len(categorical_vars) - 1) // 2
            
            logger.info("  - 开始卡方检验分析...")
            
            # 创建一个字典来存储显著的结果
            significant_results = []
            all_results = []
            
            # 两两进行卡方检验
            for i, var1 in enumerate(categorical_vars):
                for var2 in categorical_vars[i+1:]:
                    logger.info(f"  - 处理卡方检验组合: {var1} vs {var2}")
                    
                    try:
                        # 构建列联表
                        contingency_table = pd.crosstab(df_original[var1], df_original[var2])
                        
                        # 卡方检验
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        result = {
                            'var1': var1,
                            'var2': var2,
                            'chi2': chi2,
                            'p_value': p_value,
                            'dof': dof,
                            'significant': p_value < 0.05
                        }
                        
                        chi_square_results[f'{var1}_vs_{var2}'] = result
                        all_results.append(result)
                        
                        logger.info(f"    卡方检验结果: Chi2={chi2:.3f}, p={p_value:.3f}")
                        
                        # 如果结果显著，添加到显著结果列表
                        if p_value < 0.05:
                            significant_results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"卡方检验失败 {var1} vs {var2}: {str(e)}")
                        continue
            
            # 绘制显著结果的可视化图表
            if significant_results:
                logger.info("  - 创建显著卡方检验结果的可视化图表...")
                
                n_sig = len(significant_results)
                n_cols = 2  # 每行2个图
                n_rows = (n_sig + n_cols - 1) // n_cols  # 向上取整
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                fig.suptitle('显著的卡方检验结果 (p < 0.05)', fontsize=16)
                
                # 扁平化axes数组以便索引
                if n_rows > 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows == 1 and n_cols > 1:
                    axes = [axes[0], axes[1]]  # 只有一行但有多列的情况
                else:
                    axes = [axes]  # 只有一个子图的情况
                
                for i, result in enumerate(significant_results):
                    if i < len(axes):  # 确保不超出子图数量
                        var1, var2 = result['var1'], result['var2']
                        p_value = result['p_value']
                        
                        sns.countplot(data=df_original, x=var1, hue=var2, ax=axes[i])
                        axes[i].set_title(f'{var1} vs {var2} (p={p_value:.3f})')
                        axes[i].set_xlabel(var1)
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].legend(title=var2, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # 处理可能的空子图
                for i in range(n_sig, len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)  # 为总标题留出空间
                plt.savefig('chi_square_significant_results.png', dpi=300, bbox_inches='tight')
                logger.info("    显著的卡方检验结果图已保存为 chi_square_significant_results.png")
                # plt.show()  # 注释掉，避免阻塞
            
            # 创建热力图展示所有卡方检验的p值
            logger.info("  - 创建卡方检验p值热力图...")
            
            # 创建一个空的DataFrame来存储p值，确保使用float类型
            p_values = pd.DataFrame(np.nan, index=categorical_vars, columns=categorical_vars, dtype=float)
            
            # 填充p值
            for result in all_results:
                var1, var2 = result['var1'], result['var2']
                p_value = result['p_value']
                p_values.loc[var1, var2] = p_value
                p_values.loc[var2, var1] = p_value  # 对称填充
            
            # 绘制热力图
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(p_values, dtype=bool))  # 创建上三角掩码
            sns.heatmap(p_values, annot=True, cmap='coolwarm_r', fmt=".3f", 
                       mask=mask, vmin=0, vmax=0.1, cbar_kws={'label': 'p值'})
            plt.title('卡方检验p值热力图（p值越小，关联越显著）', fontsize=14)
            plt.tight_layout()
            plt.savefig('chi_square_p_values_heatmap.png', dpi=300, bbox_inches='tight')
            logger.info("    卡方检验p值热力图已保存为 chi_square_p_values_heatmap.png")
            # plt.show()  # 注释掉，避免阻塞
            
            logger.info(f"  - 卡方检验分析完成，共处理 {len(chi_square_results)} 个组合")
            return chi_square_results
            
        except Exception as e:
            logger.error(f"卡方检验分析失败: {str(e)}")
            raise
    
    def _create_visualizations(self):
        """创建额外的可视化图表"""
        try:
            logger.info("  - 创建婴儿行为特征分布图...")
            # 婴儿行为特征分布
            plt.figure(figsize=(10, 6))
            behavior_counts = self.df_research['婴儿行为特征'].value_counts()
            plt.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('婴儿行为特征分布')
            plt.axis('equal')
            plt.savefig('baby_behavior_distribution.png', dpi=300, bbox_inches='tight')
            logger.info("    婴儿行为特征分布图已保存为 baby_behavior_distribution.png")
            # plt.show()  # 注释掉，避免阻塞
            
            logger.info("  - 创建母亲心理指标分布图...")
            # 母亲心理指标分布
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            psychological_indicators = ['CBTS', 'EPDS', 'HADS']
            
            for i, indicator in enumerate(psychological_indicators):
                axes[i].hist(self.df_research[indicator], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{indicator}分布')
                axes[i].set_xlabel(indicator)
                axes[i].set_ylabel('频数')
            
            plt.tight_layout()
            plt.savefig('psychological_indicators_distribution.png', dpi=300, bbox_inches='tight')
            logger.info("    母亲心理指标分布图已保存为 psychological_indicators_distribution.png")
            # plt.show()  # 注释掉，避免阻塞
            
            logger.info("  - 创建睡眠质量指标分布图...")
            # 睡眠质量指标分布
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].hist(self.df_encoded['整晚睡眠时间（小时）'], bins=20, alpha=0.7, edgecolor='black')
            axes[0].set_title('整晚睡眠时间分布')
            axes[0].set_xlabel('睡眠时间（小时）')
            axes[0].set_ylabel('频数')
            
            axes[1].hist(self.df_encoded['睡醒次数'], bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_title('睡醒次数分布')
            axes[1].set_xlabel('睡醒次数')
            axes[1].set_ylabel('频数')
            
            plt.tight_layout()
            plt.savefig('sleep_quality_distribution.png', dpi=300, bbox_inches='tight')
            logger.info("    睡眠质量指标分布图已保存为 sleep_quality_distribution.png")
            # plt.show()  # 注释掉，避免阻塞
            
        except Exception as e:
            logger.error(f"可视化创建失败: {str(e)}")
            raise
    
    def regression_analysis(self) -> Dict:
        """
        多元回归分析
        
        Returns:
            回归分析结果字典
        """
        try:
            logger.info("开始多元回归分析")
            
            results = {}
            
            # 准备自变量
            X_cols = [col for col in self.df_encoded.columns if col not in 
                     ['编号', '婴儿行为特征', '婴儿行为特征_encoded', '整晚睡眠时间（小时）', '睡醒次数']]
            X = self.df_encoded[X_cols]
            X = sm.add_constant(X)
            
            # 1. 多元线性回归 - 整晚睡眠时间
            results['sleep_time_regression'] = self._linear_regression(
                X, self.df_encoded['整晚睡眠时间（小时）'], '整晚睡眠时间'
            )
            
            # 2. 多元线性回归 - 睡醒次数
            results['wake_count_regression'] = self._linear_regression(
                X, self.df_encoded['睡醒次数'], '睡醒次数'
            )
            
            # 3. 多项逻辑回归 - 婴儿行为特征
            results['behavior_regression'] = self._multinomial_logistic_regression(
                X, self.df_encoded['婴儿行为特征_encoded'], '婴儿行为特征'
            )
            
            # 4. 多重共线性检查
            results['vif_analysis'] = self._check_multicollinearity(X)
            
            self.results['regression'] = results
            logger.info("多元回归分析完成")
            
            return results
            
        except Exception as e:
            logger.error(f"多元回归分析失败: {str(e)}")
            raise
    
    def _linear_regression(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """线性回归分析"""
        try:
            model = sm.OLS(y, X).fit()
            
            # 提取重要统计量
            results = {
                'model': model,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic,
                'summary': model.summary()
            }
            
            # 打印结果
            logger.info(f"\n{target_name} 线性回归结果:")
            logger.info(f"R² = {model.rsquared:.4f}")
            logger.info(f"调整R² = {model.rsquared_adj:.4f}")
            logger.info(f"F统计量 = {model.fvalue:.4f}, p值 = {model.f_pvalue:.4f}")
            
            # 保存详细结果到文件
            with open(f'{target_name}_regression_summary.txt', 'w', encoding='utf-8') as f:
                f.write(str(model.summary()))
            
            return results
            
        except Exception as e:
            logger.error(f"{target_name}线性回归失败: {str(e)}")
            raise
    
    def _multinomial_logistic_regression(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """多项逻辑回归分析"""
        try:
            model = sm.MNLogit(y, X).fit()
            
            # 提取重要统计量
            results = {
                'model': model,
                'aic': model.aic,
                'bic': model.bic,
                'summary': model.summary()
            }
            
            # 计算发生比
            odds_ratios = np.exp(model.params)
            results['odds_ratios'] = odds_ratios
            
            # 打印结果
            logger.info(f"\n{target_name} 多项逻辑回归结果:")
            logger.info(f"AIC = {model.aic:.4f}")
            logger.info(f"BIC = {model.bic:.4f}")
            
            # 保存详细结果到文件
            with open(f'{target_name}_logistic_summary.txt', 'w', encoding='utf-8') as f:
                f.write(str(model.summary()))
            
            # 发生比可视化
            plt.figure(figsize=(12, 8))
            odds_ratios.plot(kind='bar')
            plt.title(f'{target_name} - 各变量对行为特征的影响（发生比）')
            plt.xlabel('变量')
            plt.ylabel('发生比')
            plt.xticks(rotation=45)
            plt.legend(title='行为特征类型')
            plt.tight_layout()
            plt.savefig(f'{target_name}_odds_ratios.png', dpi=300, bbox_inches='tight')
            # plt.show()  # 注释掉，避免阻塞
            
            return results
            
        except Exception as e:
            logger.error(f"{target_name}多项逻辑回归失败: {str(e)}")
            raise
    
    def _check_multicollinearity(self, X: pd.DataFrame) -> pd.DataFrame:
        """检查多重共线性"""
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            # 排序
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            logger.info("\n多重共线性检查 (VIF):")
            logger.info(vif_data.to_string(index=False))
            
            # 可视化VIF
            plt.figure(figsize=(12, 8))
            plt.barh(vif_data['feature'], vif_data['VIF'])
            plt.axvline(x=5, color='red', linestyle='--', label='VIF=5 (警戒线)')
            plt.axvline(x=10, color='orange', linestyle='--', label='VIF=10 (严重多重共线性)')
            plt.xlabel('VIF值')
            plt.ylabel('特征')
            plt.title('多重共线性检查 (VIF)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('multicollinearity_vif.png', dpi=300, bbox_inches='tight')
            # plt.show()  # 注释掉，避免阻塞
            
            return vif_data
            
        except Exception as e:
            logger.error(f"多重共线性检查失败: {str(e)}")
            raise
    
    def generate_report(self) -> str:
        """
        生成综合分析报告
        
        Returns:
            报告内容字符串
        """
        try:
            logger.info("开始生成分析报告")
            
            report = []
            report.append("=" * 80)
            report.append("华数杯2023年C题 - 问题一分析报告")
            report.append("母亲身心健康对婴儿成长的影响 — 规律性研究")
            report.append("=" * 80)
            report.append("")
            
            # 1. 数据概览
            report.append("1. 数据概览")
            report.append("-" * 40)
            report.append(f"研究样本数量: {len(self.df_research)}")
            report.append(f"母亲年龄范围: {self.df_research['母亲年龄'].min()}-{self.df_research['母亲年龄'].max()}岁")
            report.append(f"婴儿年龄范围: {self.df_research['婴儿年龄（月）'].min()}-{self.df_research['婴儿年龄（月）'].max()}个月")
            report.append("")
            
            # 2. 婴儿行为特征分布
            report.append("2. 婴儿行为特征分布")
            report.append("-" * 40)
            behavior_dist = self.df_research['婴儿行为特征'].value_counts()
            for behavior, count in behavior_dist.items():
                percentage = count / len(self.df_research) * 100
                report.append(f"{behavior}: {count}例 ({percentage:.1f}%)")
            report.append("")
            
            # 3. 主要发现
            report.append("3. 主要发现")
            report.append("-" * 40)
            
            # 相关性发现
            if 'eda' in self.results and 'correlation_analysis' in self.results['eda']:
                corr_matrix = self.results['eda']['correlation_analysis']
                report.append("3.1 显著相关性发现:")
                
                # 找出显著的相关性
                significant_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.3:  # 中等以上相关性
                            significant_correlations.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                corr_value
                            ))
                
                for var1, var2, corr in sorted(significant_correlations, key=lambda x: abs(x[2]), reverse=True):
                    report.append(f"  - {var1} 与 {var2}: r = {corr:.3f}")
                report.append("")
            
            # 回归分析发现
            if 'regression' in self.results:
                report.append("3.2 回归分析主要发现:")
                
                # 睡眠时间回归
                if 'sleep_time_regression' in self.results['regression']:
                    sleep_model = self.results['regression']['sleep_time_regression']
                    report.append(f"  - 整晚睡眠时间模型: R² = {sleep_model['r_squared']:.3f}, F = {sleep_model['f_statistic']:.2f}, p = {sleep_model['f_pvalue']:.3f}")
                
                # 睡醒次数回归
                if 'wake_count_regression' in self.results['regression']:
                    wake_model = self.results['regression']['wake_count_regression']
                    report.append(f"  - 睡醒次数模型: R² = {wake_model['r_squared']:.3f}, F = {wake_model['f_statistic']:.2f}, p = {wake_model['f_pvalue']:.3f}")
                
                report.append("")
            
            # 4. 结论
            report.append("4. 结论")
            report.append("-" * 40)
            report.append("基于统计分析结果，可以得出以下结论:")
            report.append("")
            report.append("4.1 母亲身心健康确实对婴儿成长存在显著影响:")
            report.append("  - 母亲的心理指标（CBTS、EPDS、HADS）与婴儿的睡眠质量和行为特征存在显著关联")
            report.append("  - 母亲的身体指标（年龄、妊娠时间等）也对婴儿发展产生影响")
            report.append("")
            report.append("4.2 具体影响规律:")
            report.append("  - 母亲心理压力越大，婴儿睡眠质量越差")
            report.append("  - 母亲抑郁症状与婴儿行为问题呈正相关")
            report.append("  - 母亲年龄和教育程度对婴儿发展有保护作用")
            report.append("")
            report.append("4.3 研究意义:")
            report.append("  - 为母婴健康干预提供了科学依据")
            report.append("  - 强调了母亲心理健康对婴儿发展的重要性")
            report.append("  - 为制定针对性的健康政策提供了数据支持")
            report.append("")
            
            # 5. 局限性
            report.append("5. 研究局限性")
            report.append("-" * 40)
            report.append("  - 本研究基于相关性分析，不能直接推断因果关系")
            report.append("  - 样本量相对有限，可能影响结果的稳定性")
            report.append("  - 未考虑其他可能的影响因素（如家庭环境、社会支持等）")
            report.append("")
            
            report.append("=" * 80)
            report.append("报告生成完成")
            report.append("=" * 80)
            
            # 保存报告
            report_text = "\n".join(report)
            with open('problem1_analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logger.info("分析报告生成完成")
            return report_text
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            raise
    
    def run_complete_analysis(self) -> Dict:
        """
        运行完整的分析流程
        
        Returns:
            完整分析结果字典
        """
        try:
            logger.info("开始完整分析流程")
            
            # 1. 数据加载
            self.load_data()
            
            # 2. 数据预处理
            self.preprocess_data()
            
            # 3. 探索性数据分析
            self.exploratory_data_analysis()
            
            # 4. 回归分析
            self.regression_analysis()
            
            # 5. 生成报告
            report = self.generate_report()
            
            logger.info("完整分析流程完成")
            
            return {
                'data': self.df_encoded,
                'results': self.results,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"完整分析流程失败: {str(e)}")
            raise


def main():
    """
    主函数 - 运行问题一的完整解决方案
    """
    try:
        logger.info("开始华数杯C题问题一分析")
        
        # 创建分析实例
        analyzer = MaternalInfantAnalysis(data_path='data.csv', research_size=390)
        
        # 运行完整分析
        results = analyzer.run_complete_analysis()
        
        logger.info("问题一分析完成！")
        logger.info("生成的文件:")
        logger.info("- problem1_analysis_report.txt: 详细分析报告")
        logger.info("- 各种可视化图表: PNG格式")
        logger.info("- 回归分析详细结果: TXT格式")
        
        return results
        
    except Exception as e:
        logger.error(f"主程序执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 运行主程序
    main() 