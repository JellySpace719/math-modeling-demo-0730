#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问题一解决方案：分析2023年教师教学评价专家组差异与可信度

本代码实现了对附件1中两组专家评分数据的详细分析，包括：
1. 数据预处理与描述性统计
2. 显著性差异检验
3. 效应量计算
4. 专家组可信度评估（基于ICC）
5. 结果可视化与综合判断

作者：AI算法工程师
日期：2023年7月31日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pingouin as pg
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
try:
    # 尝试加载系统中的中文字体
    font = FontProperties(fname=r'C:\Windows\Fonts\SimHei.ttf')
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：未能加载中文字体，图表中的中文可能无法正确显示")

class TeacherEvaluationAnalyzer:
    """教师教学评价分析器类"""

    def __init__(self):
        """初始化分析器"""
        self.group1_data = None  # 第一组专家评分数据
        self.group2_data = None  # 第二组专家评分数据
        self.teachers_scores = None  # 所有教师的平均分数据
        self.raw_expert_scores_g1 = None  # 第一组专家原始评分
        self.raw_expert_scores_g2 = None  # 第二组专家原始评分

    def load_data(self, file_group1, file_group2):
        """
        加载两组专家评分数据
        
        参数:
            file_group1 (str): 第一组专家评分文件路径
            file_group2 (str): 第二组专家评分文件路径
        """
        try:
            # 加载数据文件
            self.group1_data = pd.read_csv(file_group1, encoding='utf-8')
            self.group2_data = pd.read_csv(file_group2, encoding='utf-8')
            print(f"成功加载数据文件：{file_group1}和{file_group2}")
            return True
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return False

    def preprocess_data(self):
        """
        数据预处理：从原始评分表中提取每位教师的专家评分
        处理缺失值，计算每位专家对每位教师的总分
        """
        try:
            # 初始化结果存储
            teachers_count = 50  # 根据数据了解到共有50位教师
            expert_count = 10  # 每组10位专家
            
            # 初始化存储结构
            group1_total_scores = np.zeros((teachers_count, expert_count))
            group2_total_scores = np.zeros((teachers_count, expert_count))
            
            # 提取评分数据
            for teacher_idx in range(teachers_count):
                # 计算每位教师在数据中的起始行
                start_row = teacher_idx * 16 + 3  # 每位教师占16行，从第3行开始
                
                # 提取当前教师的评分数据
                teacher_data_g1 = self.group1_data.iloc[start_row:start_row+13, 2:12]  # 第3列到第12列为专家1-10的评分
                teacher_data_g2 = self.group2_data.iloc[start_row:start_row+13, 2:12]  # 第3列到第12列为专家11-20的评分
                
                # 处理缺失值（特别是教师27的专家6号"现代教学手段，板书设计"指标评分缺失）
                if teacher_idx == 26:  # 索引从0开始，教师27对应索引26
                    row_idx = start_row + 7  # "现代教学手段，板书设计"位于第8行
                    if pd.isna(self.group1_data.iloc[row_idx, 7]):  # 检查专家6号评分是否缺失
                        # 计算其他9位专家的平均分填充
                        other_experts = [2, 3, 4, 5, 6, 8, 9, 10, 11]  # 专家1-5,7-10对应的列索引
                        fill_value = self.group1_data.iloc[row_idx, other_experts].astype(float).mean()
                        print(f"缺失值填充：教师27的专家6号'现代教学手段，板书设计'指标使用其他专家平均值 {fill_value} 填充")
                        # 填充缺失值
                        self.group1_data.iloc[row_idx, 7] = fill_value
                
                # 计算每位专家对当前教师的总分
                for expert_idx in range(expert_count):
                    # 提取11个具体指标的评分并求和
                    expert_scores_g1 = pd.to_numeric(teacher_data_g1.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], expert_idx], errors='coerce')
                    expert_scores_g2 = pd.to_numeric(teacher_data_g2.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], expert_idx], errors='coerce')
                    
                    # 加上教学特色/整体评价分数
                    special_score_g1 = pd.to_numeric(self.group1_data.iloc[start_row + 11, expert_idx + 2], errors='coerce')
                    special_score_g2 = pd.to_numeric(self.group2_data.iloc[start_row + 11, expert_idx + 2], errors='coerce')
                    
                    # 计算总分
                    group1_total_scores[teacher_idx, expert_idx] = expert_scores_g1.sum() + special_score_g1
                    group2_total_scores[teacher_idx, expert_idx] = expert_scores_g2.sum() + special_score_g2
            
            # 保存原始专家评分数据（用于后续ICC计算）
            self.raw_expert_scores_g1 = group1_total_scores
            self.raw_expert_scores_g2 = group2_total_scores
            
            # 计算每位教师在两组专家中的平均总分
            teacher_ids = [f"教师{i+1}" for i in range(teachers_count)]
            group1_mean_scores = np.mean(group1_total_scores, axis=1)
            group2_mean_scores = np.mean(group2_total_scores, axis=1)
            
            # 创建包含教师ID和两组平均分的DataFrame
            self.teachers_scores = pd.DataFrame({
                '教师ID': teacher_ids,
                '第一组专家平均分': group1_mean_scores,
                '第二组专家平均分': group2_mean_scores
            })
            
            print("数据预处理完成：已计算每位教师在两组专家评分下的总分和平均分")
            return True
        
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            return False

    def descriptive_statistics(self):
        """
        对两组专家评分进行描述性统计分析
        返回描述性统计结果和箱线图
        """
        try:
            # 计算描述性统计指标
            desc_stats = pd.DataFrame({
                '第一组专家评分': self.teachers_scores['第一组专家平均分'].describe(),
                '第二组专家评分': self.teachers_scores['第二组专家平均分'].describe()
            })
            
            # 添加其他统计指标
            for col in ['第一组专家平均分', '第二组专家平均分']:
                desc_stats.loc['偏度', col.replace('平均分', '评分')] = stats.skew(self.teachers_scores[col])
                desc_stats.loc['峰度', col.replace('平均分', '评分')] = stats.kurtosis(self.teachers_scores[col])
                desc_stats.loc['极差', col.replace('平均分', '评分')] = self.teachers_scores[col].max() - self.teachers_scores[col].min()
            
            # 绘制描述性统计图形
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制直方图
            sns.histplot(self.teachers_scores['第一组专家平均分'], kde=True, ax=axes[0], color='blue', label='第一组')
            sns.histplot(self.teachers_scores['第二组专家平均分'], kde=True, ax=axes[0], color='red', alpha=0.6, label='第二组')
            axes[0].set_title('两组专家平均总分分布直方图')
            axes[0].set_xlabel('平均总分')
            axes[0].set_ylabel('频数')
            axes[0].legend()
            
            # 绘制箱线图
            boxplot_data = pd.melt(self.teachers_scores, id_vars=['教师ID'], 
                                  value_vars=['第一组专家平均分', '第二组专家平均分'],
                                  var_name='专家组', value_name='平均总分')
            sns.boxplot(x='专家组', y='平均总分', data=boxplot_data, ax=axes[1])
            axes[1].set_title('两组专家平均总分箱线图')
            
            plt.tight_layout()
            
            return desc_stats, fig
        
        except Exception as e:
            print(f"描述性统计分析失败: {str(e)}")
            return None, None

    def normality_test(self):
        """
        对两组评分的差值进行正态性检验
        返回检验结果和正态分布可视化
        """
        try:
            # 计算差值
            diff = self.teachers_scores['第一组专家平均分'] - self.teachers_scores['第二组专家平均分']
            
            # 进行Shapiro-Wilk正态性检验
            shapiro_test = stats.shapiro(diff)
            
            # 绘制差值的直方图和QQ图
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 直方图
            sns.histplot(diff, kde=True, ax=axes[0])
            axes[0].set_title('两组专家评分差值分布直方图')
            axes[0].set_xlabel('差值 (第一组 - 第二组)')
            axes[0].set_ylabel('频数')
            
            # QQ图
            stats.probplot(diff, plot=axes[1])
            axes[1].set_title('差值的QQ图')
            
            plt.tight_layout()
            
            # 判断是否符合正态分布
            is_normal = shapiro_test.pvalue > 0.05
            
            normality_result = {
                'statistic': shapiro_test.statistic,
                'p-value': shapiro_test.pvalue,
                'is_normal': is_normal,
                'conclusion': '差值服从正态分布' if is_normal else '差值不服从正态分布'
            }
            
            return normality_result, fig, diff
        
        except Exception as e:
            print(f"正态性检验失败: {str(e)}")
            return None, None, None

    def significance_test(self, diff, is_normal=True):
        """
        进行显著性差异检验
        
        参数:
            diff (Series): 两组评分的差值
            is_normal (bool): 差值是否服从正态分布
        
        返回:
            dict: 检验结果
        """
        try:
            if is_normal:
                # 使用配对样本t检验
                t_stat, p_value = stats.ttest_rel(
                    self.teachers_scores['第一组专家平均分'], 
                    self.teachers_scores['第二组专家平均分']
                )
                test_method = '配对样本t检验'
            else:
                # 使用Wilcoxon符号秩检验
                stat, p_value = stats.wilcoxon(
                    self.teachers_scores['第一组专家平均分'], 
                    self.teachers_scores['第二组专家平均分']
                )
                t_stat = stat  # 为了统一结果格式
                test_method = 'Wilcoxon符号秩检验'
            
            # 判断是否存在显著差异
            is_significant = p_value < 0.05
            
            test_result = {
                'test_method': test_method,
                'statistic': t_stat,
                'p-value': p_value,
                'is_significant': is_significant,
                'conclusion': '两组专家评分存在显著差异' if is_significant else '两组专家评分无显著差异'
            }
            
            return test_result
        
        except Exception as e:
            print(f"显著性差异检验失败: {str(e)}")
            return None

    def effect_size_analysis(self, diff):
        """
        计算效应量
        
        参数:
            diff (Series): 两组评分的差值
        
        返回:
            dict: 效应量分析结果
        """
        try:
            # 计算Cohen's d（配对样本）
            d = diff.mean() / diff.std()
            
            # 解释效应量大小
            if abs(d) < 0.2:
                interpretation = '效应量极小，差异在实际意义上可以忽略'
            elif abs(d) < 0.5:
                interpretation = '小效应，差异有限但可能在特定情境下有实际意义'
            elif abs(d) < 0.8:
                interpretation = '中等效应，差异在实际应用中有一定意义'
            else:
                interpretation = '大效应，差异具有较大的实际意义'
            
            effect_size_result = {
                'Cohen_d': d,
                'interpretation': interpretation
            }
            
            return effect_size_result
        
        except Exception as e:
            print(f"效应量分析失败: {str(e)}")
            return None

    def calculate_icc(self):
        """
        计算两组专家评分的组内相关系数(ICC)
        
        返回:
            dict: ICC计算结果
        """
        try:
            # 准备长格式数据
            teachers_count = self.raw_expert_scores_g1.shape[0]
            expert_count = self.raw_expert_scores_g1.shape[1]
            
            # 第一组专家ICC计算
            ratings_g1 = []
            teachers_ids_g1 = []
            raters_g1 = []
            
            for teacher_idx in range(teachers_count):
                for expert_idx in range(expert_count):
                    teachers_ids_g1.append(teacher_idx + 1)
                    raters_g1.append(expert_idx + 1)
                    ratings_g1.append(self.raw_expert_scores_g1[teacher_idx, expert_idx])
            
            icc_data_g1 = pd.DataFrame({
                'teacher': teachers_ids_g1,
                'rater': raters_g1,
                'score': ratings_g1
            })
            
            # 使用pingouin库计算ICC
            icc_g1 = pg.intraclass_corr(
                data=icc_data_g1,
                targets='teacher',
                raters='rater',
                ratings='score',
                nan_policy='omit'  # 忽略缺失值
            )
            
            # 第二组专家ICC计算
            ratings_g2 = []
            teachers_ids_g2 = []
            raters_g2 = []
            
            for teacher_idx in range(teachers_count):
                for expert_idx in range(expert_count):
                    teachers_ids_g2.append(teacher_idx + 1)
                    raters_g2.append(expert_idx + 1)
                    ratings_g2.append(self.raw_expert_scores_g2[teacher_idx, expert_idx])
            
            icc_data_g2 = pd.DataFrame({
                'teacher': teachers_ids_g2,
                'rater': raters_g2,
                'score': ratings_g2
            })
            
            icc_g2 = pg.intraclass_corr(
                data=icc_data_g2,
                targets='teacher',
                raters='rater',
                ratings='score',
                nan_policy='omit'  # 忽略缺失值
            )
            
            # 提取双向随机效应、绝对一致性、单次测量的ICC(2,1)值
            icc21_g1 = icc_g1.loc[icc_g1['Type'] == 'ICC2', 'ICC'].values[0]
            icc21_g2 = icc_g2.loc[icc_g2['Type'] == 'ICC2', 'ICC'].values[0]
            
            # 解释ICC值
            def interpret_icc(icc):
                if icc < 0.5:
                    return "差或不可接受的一致性"
                elif icc < 0.75:
                    return "中等一致性"
                elif icc < 0.9:
                    return "良好一致性"
                else:
                    return "优秀一致性"
            
            # 判断哪组更可信
            more_credible = "第一组专家" if icc21_g1 > icc21_g2 else "第二组专家"
            
            # 绘制ICC值对比图
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(['第一组专家', '第二组专家'], [icc21_g1, icc21_g2], color=['blue', 'red'])
            ax.set_title('两组专家评分的ICC值对比')
            ax.set_ylabel('ICC(2,1)值')
            ax.set_ylim(0, 1)
            
            # 在柱状图上方添加具体ICC值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            # 添加0.5和0.75的水平参考线
            ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.7)
            ax.text(0, 0.51, '0.5 - 中等一致性阈值', va='bottom', ha='left', alpha=0.7)
            
            ax.axhline(y=0.75, linestyle='--', color='gray', alpha=0.7)
            ax.text(0, 0.76, '0.75 - 良好一致性阈值', va='bottom', ha='left', alpha=0.7)
            
            plt.tight_layout()
            
            icc_result = {
                'ICC2_1_G1': icc21_g1,
                'ICC2_1_G2': icc21_g2,
                'interpretation_G1': interpret_icc(icc21_g1),
                'interpretation_G2': interpret_icc(icc21_g2),
                'more_credible': more_credible,
                'conclusion': f"基于ICC分析，{more_credible}的评分结果更可信，其ICC值为{icc21_g1 if more_credible == '第一组专家' else icc21_g2:.3f}，"
                              f"表明{interpret_icc(icc21_g1 if more_credible == '第一组专家' else icc21_g2)}"
            }
            
            return icc_result, fig
        
        except Exception as e:
            print(f"ICC计算失败: {str(e)}")
            return None, None

    def generate_comprehensive_result(self, normality_result, significance_test_result, 
                                      effect_size_result, icc_result):
        """
        生成综合分析结果
        
        参数:
            normality_result (dict): 正态性检验结果
            significance_test_result (dict): 显著性差异检验结果
            effect_size_result (dict): 效应量分析结果
            icc_result (dict): ICC计算结果
        
        返回:
            dict: 综合分析结果
        """
        try:
            # 整合所有结果
            comprehensive_result = {
                'normality_test': normality_result,
                'significance_test': significance_test_result,
                'effect_size': effect_size_result,
                'icc_analysis': icc_result,
            }
            
            # 构建最终结论文本
            conclusion = []
            
            # 1. 正态性检验结论
            conclusion.append(f"1. 正态性检验：{normality_result['conclusion']}（p值={normality_result['p-value']:.3f}）。")
            
            # 2. 显著性差异结论
            conclusion.append(f"2. 显著性差异检验（{significance_test_result['test_method']}）：{significance_test_result['conclusion']}（p值={significance_test_result['p-value']:.3f}）。")
            
            # 3. 效应量分析结论
            conclusion.append(f"3. 效应量分析：Cohen's d={effect_size_result['Cohen_d']:.3f}，{effect_size_result['interpretation']}。")
            
            # 4. ICC分析结论
            conclusion.append(f"4. ICC分析：第一组专家ICC={icc_result['ICC2_1_G1']:.3f}（{icc_result['interpretation_G1']}），"
                             f"第二组专家ICC={icc_result['ICC2_1_G2']:.3f}（{icc_result['interpretation_G2']}）。")
            
            # 5. 最终结论：哪一组更可信
            conclusion.append(f"5. 综合结论：基于ICC分析，{icc_result['more_credible']}的评分结果更可信，因为其内部一致性更高，"
                             f"表明评分标准更统一、评分更稳定。")
            
            # 如果有显著差异，补充说明
            if significance_test_result['is_significant']:
                mean_diff = self.teachers_scores['第一组专家平均分'].mean() - self.teachers_scores['第二组专家平均分'].mean()
                higher_group = "第一组" if mean_diff > 0 else "第二组"
                conclusion.append(f"   同时，两组专家评分存在统计显著差异，{higher_group}专家的平均评分更高，"
                                 f"但该差异的效应量为{effect_size_result['Cohen_d']:.3f}，{effect_size_result['interpretation']}。")
            
            comprehensive_result['conclusion_text'] = "\n".join(conclusion)
            
            return comprehensive_result
        
        except Exception as e:
            print(f"生成综合分析结果失败: {str(e)}")
            return None

    def save_results(self, results_dict, output_file):
        """
        保存分析结果到文件
        
        参数:
            results_dict (dict): 分析结果字典
            output_file (str): 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入标题
                f.write("# 2023年教师教学评价专家组差异与可信度分析报告\n\n")
                
                # 写入结论摘要
                f.write("## 分析结论\n\n")
                f.write(results_dict['conclusion_text'])
                f.write("\n\n")
                
                # 添加详细的描述性统计数据
                f.write("## 描述性统计分析\n\n")
                
                # 两组专家评分的基本统计量
                f.write("### 1. 两组专家评分的基本统计指标\n\n")
                
                # 创建描述性统计表格
                desc_stats = pd.DataFrame({
                    '第一组专家评分': self.teachers_scores['第一组专家平均分'].describe(),
                    '第二组专家评分': self.teachers_scores['第二组专家平均分'].describe()
                })
                
                # 添加其他统计指标
                for col in ['第一组专家平均分', '第二组专家平均分']:
                    desc_stats.loc['偏度', col.replace('平均分', '评分')] = stats.skew(self.teachers_scores[col])
                    desc_stats.loc['峰度', col.replace('平均分', '评分')] = stats.kurtosis(self.teachers_scores[col])
                    desc_stats.loc['极差', col.replace('平均分', '评分')] = self.teachers_scores[col].max() - self.teachers_scores[col].min()
                
                # 写入Markdown表格
                f.write("| 统计指标 | 第一组专家评分 | 第二组专家评分 |\n")
                f.write("|---------|------------|------------|\n")
                
                for idx in desc_stats.index:
                    f.write(f"| {idx} | {desc_stats.loc[idx, '第一组专家评分']:.4f} | {desc_stats.loc[idx, '第二组专家评分']:.4f} |\n")
                
                f.write("\n")
                
                # 各教师评分详情
                f.write("### 2. 50位教师的评分详情\n\n")
                f.write("| 教师ID | 第一组专家平均分 | 第二组专家平均分 | 差值(第一组-第二组) |\n")
                f.write("|--------|------------|------------|---------------|\n")
                
                # 计算差值并排序
                self.teachers_scores['差值'] = self.teachers_scores['第一组专家平均分'] - self.teachers_scores['第二组专家平均分']
                sorted_scores = self.teachers_scores.sort_values(by='第一组专家平均分', ascending=False)
                
                # 写入每位教师的评分数据
                for _, row in sorted_scores.iterrows():
                    f.write(f"| {row['教师ID']} | {row['第一组专家平均分']:.2f} | {row['第二组专家平均分']:.2f} | {row['差值']:.2f} |\n")
                
                f.write("\n")
                
                # 分布特征分析
                f.write("### 3. 评分分布特征分析\n\n")
                
                # 计算分数段分布
                score_bins = [70, 75, 80, 85, 90, 95, 100]
                
                g1_dist = pd.cut(self.teachers_scores['第一组专家平均分'], bins=score_bins).value_counts().sort_index()
                g2_dist = pd.cut(self.teachers_scores['第二组专家平均分'], bins=score_bins).value_counts().sort_index()
                
                f.write("#### 分数段分布\n\n")
                f.write("| 分数段 | 第一组专家(人数) | 第二组专家(人数) |\n")
                f.write("|-------|--------------|------------|\n")
                
                for i, bin_name in enumerate(g1_dist.index):
                    f.write(f"| {bin_name} | {g1_dist.iloc[i]} | {g2_dist.iloc[i]} |\n")
                
                f.write("\n")
                
                # 写入详细统计结果
                f.write("## 假设检验与分析结果\n\n")
                
                # 正态性检验
                f.write("### 1. 正态性检验（Shapiro-Wilk测试）\n\n")
                f.write(f"- 统计量: {results_dict['normality_test']['statistic']:.4f}\n")
                f.write(f"- p值: {results_dict['normality_test']['p-value']:.4f}\n")
                f.write(f"- 结论: {results_dict['normality_test']['conclusion']}\n\n")
                
                # 显著性差异检验
                f.write(f"### 2. 显著性差异检验（{results_dict['significance_test']['test_method']}）\n\n")
                f.write(f"- 统计量: {results_dict['significance_test']['statistic']}\n")
                f.write(f"- p值: {results_dict['significance_test']['p-value']:.4f}\n")
                f.write(f"- 结论: {results_dict['significance_test']['conclusion']}\n\n")
                
                # 效应量分析
                f.write("### 3. 效应量分析\n\n")
                f.write(f"- Cohen's d: {results_dict['effect_size']['Cohen_d']:.4f}\n")
                f.write(f"- 解释: {results_dict['effect_size']['interpretation']}\n\n")
                
                # ICC分析
                f.write("### 4. ICC分析（组内相关系数）\n\n")
                f.write(f"- 第一组专家ICC(2,1): {results_dict['icc_analysis']['ICC2_1_G1']:.4f}\n")
                f.write(f"- 第二组专家ICC(2,1): {results_dict['icc_analysis']['ICC2_1_G2']:.4f}\n")
                f.write(f"- 第一组专家ICC解释: {results_dict['icc_analysis']['interpretation_G1']}\n")
                f.write(f"- 第二组专家ICC解释: {results_dict['icc_analysis']['interpretation_G2']}\n")
                f.write(f"- 更可信组别: {results_dict['icc_analysis']['more_credible']}\n")
                
            print(f"分析结果已保存至 {output_file}")
            return True
            
        except Exception as e:
            print(f"保存分析结果失败: {str(e)}")
            return False

    def run_analysis(self, file_group1, file_group2, output_file="analysis_results.md"):
        """
        运行完整分析流程
        
        参数:
            file_group1 (str): 第一组专家评分文件路径
            file_group2 (str): 第二组专家评分文件路径
            output_file (str): 输出结果文件路径
        """
        try:
            print("开始分析...")
            
            # 1. 加载数据
            if not self.load_data(file_group1, file_group2):
                return False
            
            # 2. 数据预处理
            if not self.preprocess_data():
                return False
            
            # 3. 描述性统计
            desc_stats, desc_fig = self.descriptive_statistics()
            if desc_fig:
                desc_fig.savefig("descriptive_statistics.png", dpi=300, bbox_inches="tight")
                print("描述性统计图表已保存至 descriptive_statistics.png")
            
            # 4. 正态性检验
            normality_result, normality_fig, diff = self.normality_test()
            if normality_fig:
                normality_fig.savefig("normality_test.png", dpi=300, bbox_inches="tight")
                print("正态性检验图表已保存至 normality_test.png")
            
            is_normal = normality_result['is_normal'] if normality_result else True
            
            # 5. 显著性差异检验
            significance_test_result = self.significance_test(diff, is_normal)
            
            # 6. 效应量分析
            effect_size_result = self.effect_size_analysis(diff)
            
            # 7. ICC计算与分析
            icc_result, icc_fig = self.calculate_icc()
            if icc_fig:
                icc_fig.savefig("icc_analysis.png", dpi=300, bbox_inches="tight")
                print("ICC分析图表已保存至 icc_analysis.png")
            
            # 8. 生成综合分析结果
            comprehensive_result = self.generate_comprehensive_result(
                normality_result, significance_test_result, effect_size_result, icc_result
            )
            
            # 9. 保存结果
            self.save_results(comprehensive_result, output_file)
            
            # 10. 绘制教师评分散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(self.teachers_scores['第一组专家平均分'], self.teachers_scores['第二组专家平均分'], 
                       alpha=0.7, s=50)
            plt.plot([70, 100], [70, 100], 'r--', linewidth=2)  # 参考线：x=y
            
            # 添加教师标签
            for i, txt in enumerate(self.teachers_scores['教师ID']):
                plt.annotate(txt, 
                            (self.teachers_scores['第一组专家平均分'].iloc[i], 
                             self.teachers_scores['第二组专家平均分'].iloc[i]),
                            fontsize=8)
            
            plt.xlabel('第一组专家平均分')
            plt.ylabel('第二组专家平均分')
            plt.title('两组专家评分散点图')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig("scores_scatter.png", dpi=300, bbox_inches="tight")
            print("教师评分散点图已保存至 scores_scatter.png")
            
            print("分析完成！")
            return True
            
        except Exception as e:
            print(f"分析过程出错: {str(e)}")
            return False


def main():
    """主函数"""
    try:
        # 创建分析器实例
        analyzer = TeacherEvaluationAnalyzer()
        
        # 运行分析
        analyzer.run_analysis(
            file_group1="attachment1_group1_expert_scores.csv",
            file_group2="attachment1_group2_expert_scores.csv",
            output_file="problem1_analysis_results.md"
        )
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main() 