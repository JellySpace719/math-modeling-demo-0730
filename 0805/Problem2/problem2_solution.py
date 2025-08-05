#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问题二：婴儿行为特征分类与预测模型

本模型基于母亲的身体指标与心理指标，预测婴儿的行为特征（安静型、中等型、矛盾型）。
使用随机森林分类算法，通过特征工程、参数调优和模型评估，构建一个高精度的分类模型。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
import matplotlib as mpl

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    mpl.rcParams['font.family'] = ['SimHei']
except:
    warnings.warn("无法设置中文字体，图表中的中文可能无法正确显示")

# 忽略警告
warnings.filterwarnings('ignore')

class Config:
    """配置类，存储模型参数和路径"""
    # 随机种子，确保结果可复现
    RANDOM_STATE = 42
    # 数据路径
    DATA_PATH = "../data.csv"
    # 输出路径
    OUTPUT_DIR = "./output"
    # 图表输出路径
    FIGURE_DIR = "./figures"
    # 模型保存路径
    MODEL_DIR = "./models"

class BabyBehaviorClassifier:
    """婴儿行为特征分类模型"""
    
    def __init__(self, config=None):
        """
        初始化分类器
        
        参数:
            config: 配置对象，包含模型参数和路径
        """
        self.config = config or Config()
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_encoder = None
        self.create_directories()
        
    def create_directories(self):
        """创建必要的目录"""
        for directory in [self.config.OUTPUT_DIR, self.config.FIGURE_DIR, self.config.MODEL_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def load_data(self):
        """
        加载数据并进行初步处理
        
        返回:
            X_train: 训练特征
            y_train: 训练标签
            X_predict: 待预测数据
            predict_ids: 待预测数据的ID
        """
        try:
            # 加载数据
            df = pd.read_csv(self.config.DATA_PATH)
            
            # 分离训练集和预测集
            train_df = df[df['婴儿行为特征'].notna() & (df['婴儿行为特征'] != '')]
            predict_df = df[df['婴儿行为特征'].isna() | (df['婴儿行为特征'] == '')]
            
            # 提取特征和标签
            X_train = train_df[['母亲年龄', '婚姻状况', '教育程度', '妊娠时间（周数）', 
                               '分娩方式', 'CBTS', 'EPDS', 'HADS']]
            y_train = train_df['婴儿行为特征']
            
            # 提取待预测数据
            X_predict = predict_df[['母亲年龄', '婚姻状况', '教育程度', '妊娠时间（周数）', 
                                  '分娩方式', 'CBTS', 'EPDS', 'HADS']]
            predict_ids = predict_df['编号']
            
            # 打印数据集信息
            print(f"训练集大小: {len(X_train)}")
            print(f"预测集大小: {len(X_predict)}")
            
            return X_train, y_train, X_predict, predict_ids
        
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def analyze_target_distribution(self, y_train):
        """
        分析目标变量的分布
        
        参数:
            y_train: 训练标签
        """
        try:
            # 计算各类别的样本数和比例
            value_counts = y_train.value_counts()
            value_percentages = y_train.value_counts(normalize=True) * 100
            
            # 打印类别分布
            print("\n婴儿行为特征分布:")
            for category, count in value_counts.items():
                percentage = value_percentages[category]
                print(f"{category}: {count}例 ({percentage:.1f}%)")
            
            # 绘制类别分布图
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x=y_train, palette='viridis')
            plt.title('婴儿行为特征分布', fontsize=14)
            plt.xlabel('行为特征类型', fontsize=12)
            plt.ylabel('样本数量', fontsize=12)
            
            # 在柱状图上显示具体数值和百分比
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                percentage = 100 * height / len(y_train)
                ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                        f'{int(height)}例\n({percentage:.1f}%)',
                        ha="center", fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{self.config.FIGURE_DIR}/婴儿行为特征分布.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"分析目标分布时出错: {e}")
    
    def encode_target(self, y_train):
        """
        对目标变量进行编码
        
        参数:
            y_train: 训练标签
            
        返回:
            y_encoded: 编码后的标签
        """
        try:
            # 创建标签映射
            self.target_encoder = {
                '安静型': 0,
                '中等型': 1,
                '矛盾型': 2
            }
            
            # 反向映射（用于后续解码）
            self.target_decoder = {v: k for k, v in self.target_encoder.items()}
            
            # 编码标签
            y_encoded = y_train.map(self.target_encoder)
            
            return y_encoded
            
        except Exception as e:
            print(f"编码目标变量时出错: {e}")
            raise
    
    def create_preprocessor(self):
        """
        创建特征预处理器
        
        返回:
            preprocessor: 列转换器对象
        """
        try:
            # 定义数值型和类别型特征
            numeric_features = ['母亲年龄', '妊娠时间（周数）', 'CBTS', 'EPDS', 'HADS']
            categorical_features = ['婚姻状况', '教育程度', '分娩方式']
            
            # 创建转换器
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            # 创建列转换器
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            self.feature_names = numeric_features + categorical_features
            return preprocessor
            
        except Exception as e:
            print(f"创建预处理器时出错: {e}")
            raise
    
    def build_model(self, X_train, y_train):
        """
        构建和训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            
        返回:
            best_model: 训练好的最佳模型
        """
        try:
            # 创建预处理器
            self.preprocessor = self.create_preprocessor()
            
            # 创建管道
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(random_state=self.config.RANDOM_STATE))
            ])
            
            # 定义参数网格
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__max_features': ['sqrt', 'log2'],
                'classifier__class_weight': [None, 'balanced']
            }
            
            # 创建网格搜索
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.RANDOM_STATE),
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # 训练模型
            print("\n开始模型训练和参数调优...")
            grid_search.fit(X_train, y_train)
            
            # 获取最佳参数和分数
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"最佳参数: {best_params}")
            print(f"最佳交叉验证加权F1分数: {best_score:.4f}")
            
            # 返回最佳模型
            self.model = grid_search.best_estimator_
            return self.model
            
        except Exception as e:
            print(f"构建模型时出错: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        """
        try:
            # 预测测试集
            y_pred = self.model.predict(X_test)
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print("\n模型评估结果:")
            print(f"准确率: {accuracy:.4f}")
            print(f"加权F1分数: {f1:.4f}")
            
            # 打印分类报告
            print("\n详细分类报告:")
            target_names = ['安静型', '中等型', '矛盾型']
            report = classification_report(y_test, y_pred, target_names=target_names)
            print(report)
            
            # 保存分类报告到文件
            with open(f"{self.config.OUTPUT_DIR}/classification_report.txt", 'w', encoding='utf-8') as f:
                f.write(f"准确率: {accuracy:.4f}\n")
                f.write(f"加权F1分数: {f1:.4f}\n\n")
                f.write(report)
            
            # 绘制混淆矩阵
            self.plot_confusion_matrix(y_test, y_pred, target_names)
            
            # 提取并可视化特征重要性
            self.visualize_feature_importance()
            
        except Exception as e:
            print(f"评估模型时出错: {e}")
    
    def plot_confusion_matrix(self, y_test, y_pred, target_names):
        """
        绘制混淆矩阵
        
        参数:
            y_test: 真实标签
            y_pred: 预测标签
            target_names: 类别名称
        """
        try:
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.title('混淆矩阵', fontsize=14)
            plt.ylabel('真实类别', fontsize=12)
            plt.xlabel('预测类别', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.config.FIGURE_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")
    
    def visualize_feature_importance(self):
        """提取并可视化特征重要性"""
        try:
            # 获取特征重要性
            feature_importance = self.model.named_steps['classifier'].feature_importances_
            
            # 获取特征名称
            numeric_features = ['母亲年龄', '妊娠时间（周数）', 'CBTS', 'EPDS', 'HADS']
            
            # 获取OneHotEncoder的特征名称
            ohe = self.model.named_steps['preprocessor'].transformers_[1][1]
            categorical_features = []
            for i, category in enumerate(['婚姻状况', '教育程度', '分娩方式']):
                categories = ohe.categories_[i]
                for cat in categories:
                    categorical_features.append(f"{category}_{cat}")
            
            # 合并所有特征名称
            feature_names = numeric_features + categorical_features
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(feature_importance)],
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            # 保存特征重要性到CSV
            importance_df.to_csv(f"{self.config.OUTPUT_DIR}/feature_importance.csv", index=False, encoding='utf-8')
            
            # 绘制特征重要性图
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
            plt.title('特征重要性排名（前15）', fontsize=14)
            plt.xlabel('重要性', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.config.FIGURE_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 打印前10个重要特征
            print("\n特征重要性排名（前10）:")
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['Feature']}: {row['Importance']:.4f}")
            
        except Exception as e:
            print(f"可视化特征重要性时出错: {e}")
    
    def predict_new_data(self, X_predict, predict_ids):
        """
        预测新数据
        
        参数:
            X_predict: 待预测特征
            predict_ids: 待预测数据的ID
            
        返回:
            predictions_df: 预测结果DataFrame
        """
        try:
            # 预测
            y_pred_encoded = self.model.predict(X_predict)
            
            # 解码预测结果
            y_pred = [self.target_decoder[code] for code in y_pred_encoded]
            
            # 创建预测结果DataFrame
            predictions_df = pd.DataFrame({
                '编号': predict_ids,
                '预测婴儿行为特征': y_pred
            })
            
            # 保存预测结果
            predictions_df.to_csv(f"{self.config.OUTPUT_DIR}/婴儿行为特征预测结果.csv", index=False, encoding='utf-8')
            
            # 打印预测结果
            print("\n预测结果:")
            print(predictions_df)
            
            return predictions_df
            
        except Exception as e:
            print(f"预测新数据时出错: {e}")
            raise
    
    def analyze_predictions(self, predictions_df):
        """
        分析预测结果
        
        参数:
            predictions_df: 预测结果DataFrame
        """
        try:
            # 计算各类别的预测数量
            pred_counts = predictions_df['预测婴儿行为特征'].value_counts()
            
            # 绘制预测结果分布图
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x='预测婴儿行为特征', data=predictions_df, palette='viridis')
            plt.title('预测婴儿行为特征分布', fontsize=14)
            plt.xlabel('行为特征类型', fontsize=12)
            plt.ylabel('样本数量', fontsize=12)
            
            # 在柱状图上显示具体数值和百分比
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                percentage = 100 * height / len(predictions_df)
                ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                        f'{int(height)}例\n({percentage:.1f}%)',
                        ha="center", fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{self.config.FIGURE_DIR}/预测婴儿行为特征分布.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n预测结果分布:")
            for category, count in pred_counts.items():
                percentage = 100 * count / len(predictions_df)
                print(f"{category}: {count}例 ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"分析预测结果时出错: {e}")
    
    def run(self):
        """执行完整的模型训练和预测流程"""
        try:
            print("="*50)
            print("开始执行婴儿行为特征分类与预测模型")
            print("="*50)
            
            # 加载数据
            X_train, y_train, X_predict, predict_ids = self.load_data()
            
            # 分析目标变量分布
            self.analyze_target_distribution(y_train)
            
            # 编码目标变量
            y_train_encoded = self.encode_target(y_train)
            
            # 划分训练集和测试集
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_train, y_train_encoded, 
                test_size=0.2, 
                random_state=self.config.RANDOM_STATE,
                stratify=y_train_encoded  # 确保分层抽样
            )
            
            # 构建和训练模型
            self.build_model(X_train_split, y_train_split)
            
            # 评估模型
            self.evaluate_model(X_test, y_test)
            
            # 使用全部训练数据重新训练最终模型
            print("\n使用全部训练数据重新训练最终模型...")
            self.build_model(X_train, y_train_encoded)
            
            # 预测新数据
            predictions_df = self.predict_new_data(X_predict, predict_ids)
            
            # 分析预测结果
            self.analyze_predictions(predictions_df)
            
            print("\n模型执行完成！")
            print(f"预测结果已保存至: {self.config.OUTPUT_DIR}/婴儿行为特征预测结果.csv")
            
        except Exception as e:
            print(f"模型执行过程中出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 创建分类器实例
    classifier = BabyBehaviorClassifier()
    
    # 执行完整流程
    classifier.run() 