# city_data_validator.py
import pandas as pd
import numpy as np
import re
from datetime import datetime

class CityDataValidator:
    """城市数据验证和清洗"""
    
    def __init__(self):
        self.validation_rules = {
            'population': lambda x: self.validate_number(x, 0, 5000),  # 万人
            'area': lambda x: self.validate_number(x, 0, 200000),      # 平方公里
            'gdp': lambda x: self.validate_number(x, 0, 50000),        # 亿元
            'pm25': lambda x: self.validate_number(x, 0, 500),         # PM2.5值
            'avg_temperature': lambda x: self.validate_number(x, -50, 50),  # 温度
            'air_quality': lambda x: self.validate_number(x, 0, 500),  # AQI
            'subway_lines': lambda x: self.validate_number(x, 0, 30),  # 地铁线路数
            'history_years': lambda x: self.validate_number(x, 0, 5000) # 历史年数
        }
        
        self.text_rules = {
            'climate_type': lambda x: len(x) < 50 if x else True,
            'local_cuisine': lambda x: len(x) < 100 if x else True,
            'administrative_level': lambda x: x in ['直辖市', '副省级市', '地级市', '县级市', ''] if x else True
        }
    
    def validate_number(self, value, min_val, max_val):
        """验证数值范围"""
        if not value or value == '':
            return True  # 空值允许
        try:
            # 清理数值中的中文字符
            clean_value = re.sub(r'[万亿元人平方公里°C毫米]', '', str(value))
            num = float(clean_value)
            return min_val <= num <= max_val
        except:
            return False
    
    def clean_data(self, df):
        """清洗数据"""
        cleaned_df = df.copy()
        
        # 数值字段验证
        for column, rule in self.validation_rules.items():
            if column in cleaned_df.columns:
                mask = cleaned_df[column].apply(rule)
                invalid_count = (~mask).sum()
                if invalid_count > 0:
                    print(f"字段 {column} 有 {invalid_count} 个无效值被清空")
                    cleaned_df.loc[~mask, column] = ''
        
        # 文本字段验证
        for column, rule in self.text_rules.items():
            if column in cleaned_df.columns:
                mask = cleaned_df[column].apply(rule)
                invalid_count = (~mask).sum()
                if invalid_count > 0:
                    print(f"字段 {column} 有 {invalid_count} 个无效值被清空")
                    cleaned_df.loc[~mask, column] = ''
        
        return cleaned_df
    
    def generate_data_report(self, df):
        """生成数据质量报告"""
        report = {
            'total_cities': len(df),
            'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'field_completeness': {},
            'data_quality_summary': {}
        }
        
        # 计算各字段完整性
        for column in df.columns:
            non_empty = df[column].replace('', np.nan).notna().sum()
            completion_rate = (non_empty / len(df)) * 100 if len(df) > 0 else 0
            report['field_completeness'][column] = {
                'completed_count': int(non_empty),
                'completion_rate': f"{completion_rate:.2f}%"
            }
        
        # 数据质量汇总
        total_fields = len(df.columns)
        avg_completion = sum([float(v['completion_rate'].replace('%', '')) 
                             for v in report['field_completeness'].values()]) / total_fields
        
        report['data_quality_summary'] = {
            'average_completion_rate': f"{avg_completion:.2f}%",
            'total_fields': total_fields,
            'high_quality_cities': int((df.count(axis=1) / total_fields > 0.8).sum()),
            'low_quality_cities': int((df.count(axis=1) / total_fields < 0.5).sum())
        }
        
        return report