import pandas as pd
import os
import glob
import numpy as np
import re
from pathlib import Path
from collections import Counter

def extract_city_comprehensive_features(city_df, city_name):
    """
    基于外国游客偏好提取城市综合特征
    """
    features = {'city': city_name}
    
    # ===================== 1. 景点吸引力指标 =====================
    # 评分相关
    valid_scores = pd.to_numeric(city_df['评分'].replace('', np.nan), errors='coerce').dropna()
    if len(valid_scores) > 0:
        features['平均景点评分'] = valid_scores.mean()
        features['高质量景点数'] = len(valid_scores[valid_scores >= 4.5])
        features['顶级景点数'] = len(valid_scores[valid_scores >= 4.8])  # 顶级景点
    else:
        features['平均景点评分'] = 0
        features['高质量景点数'] = 0
        features['顶级景点数'] = 0
    
    features['景点总数'] = len(city_df)
    
    # 建议游玩时间（反映景点深度）
    play_times = city_df['建议游玩时间'].fillna('').astype(str)
    long_visit_count = 0  # 需要长时间游览的景点
    for time_str in play_times:
        if any(keyword in time_str for keyword in ['天', '半天', '一天', '2小时以上', '3小时以上']):
            long_visit_count += 1
    features['深度游景点比例'] = long_visit_count / len(city_df) if len(city_df) > 0 else 0
    
    # ===================== 2. 人文底蕴指标 =====================
    introductions = city_df['介绍'].fillna('').str.lower()
    
    # 历史文化关键词
    culture_keywords = ['古', '历史', '文化', '遗址', '博物', '寺', '庙', '宫', '故居', 
                       '古镇', '古城', '传统', '非遗', '文物', '古建筑', '皇家', '帝王']
    
    # 宗教文化关键词  
    religion_keywords = ['佛', '道', '寺庙', '道观', '教堂', '清真寺', '禅', '佛教', '道教']
    
    # 现代文化关键词
    modern_culture_keywords = ['博物馆', '美术馆', '艺术', '文化中心', '剧院', '音乐厅']
    
    culture_score = sum([intro.count(word) for intro in introductions for word in culture_keywords])
    religion_score = sum([intro.count(word) for intro in introductions for word in religion_keywords])
    modern_culture_score = sum([intro.count(word) for intro in introductions for word in modern_culture_keywords])
    
    features['历史文化指数'] = culture_score / len(city_df) if len(city_df) > 0 else 0
    features['宗教文化指数'] = religion_score / len(city_df) if len(city_df) > 0 else 0
    features['现代文化指数'] = modern_culture_score / len(city_df) if len(city_df) > 0 else 0
    
    # ===================== 3. 环境与生态指标 =====================
    nature_keywords = ['山', '水', '湖', '公园', '森林', '花园', '植物园', '湿地', 
                      '自然', '生态', '绿地', '风景', '景观', '花海', '竹林', '溪流']
    
    nature_score = sum([intro.count(word) for intro in introductions for word in nature_keywords])
    features['自然生态指数'] = nature_score / len(city_df) if len(city_df) > 0 else 0
    
    # 空气质量相关（通过景点类型间接反映）
    outdoor_keywords = ['户外', '登山', '徒步', '露营', '观景', '眺望']
    outdoor_score = sum([intro.count(word) for intro in introductions for word in outdoor_keywords])
    features['户外活动指数'] = outdoor_score / len(city_df) if len(city_df) > 0 else 0
    
    # ===================== 4. 城市规模与现代化指数 =====================
    urban_keywords = ['广场', '商业', '购物', '中心', '现代', '国际', '都市', '大型', 
                     '综合体', '地标', '摩天', '高楼', 'CBD']
    
    urban_score = sum([intro.count(word) for intro in introductions for word in urban_keywords])
    features['城市现代化指数'] = urban_score / len(city_df) if len(city_df) > 0 else 0
    
    # 景点分布广度（通过地址判断）
    addresses = city_df['地址'].fillna('').astype(str)
    districts = set()
    for addr in addresses:
        # 提取区/县名称
        district_match = re.findall(r'[\u4e00-\u9fff]+[区县市]', addr)
        districts.update(district_match)
    features['城市覆盖广度'] = len(districts)
    
    # ===================== 5. 交通便利性指标 =====================
    # 通过开放时间判断便利性
    opening_hours = city_df['开放时间'].fillna('').astype(str)
    convenient_hours = 0
    for hours in opening_hours:
        if any(keyword in hours for keyword in ['全天', '24小时', '全年', '无需预约']):
            convenient_hours += 1
    features['交通便利指数'] = convenient_hours / len(city_df) if len(city_df) > 0 else 0
    
    # 交通枢纽相关景点
    transport_keywords = ['站', '机场', '港口', '码头', '交通', '枢纽', '地铁']
    transport_score = sum([intro.count(word) for intro in introductions for word in transport_keywords])
    features['交通枢纽指数'] = transport_score / len(city_df) if len(city_df) > 0 else 0
    
    # ===================== 6. 气候舒适度指标 =====================
    season_recommendations = city_df['建议季节'].fillna('').astype(str)
    
    # 全年适宜游览的景点比例
    all_season_count = 0
    spring_summer_count = 0  # 春夏适宜
    
    for season in season_recommendations:
        if any(keyword in season for keyword in ['四季', '全年', '任何时候', '春夏秋冬']):
            all_season_count += 1
        elif any(keyword in season for keyword in ['春', '夏', '温暖', '4月', '5月', '6月', '7月', '8月', '9月']):
            spring_summer_count += 1
    
    features['全季节适宜指数'] = all_season_count / len(city_df) if len(city_df) > 0 else 0
    features['春夏适宜指数'] = spring_summer_count / len(city_df) if len(city_df) > 0 else 0
    
    # ===================== 7. 美食文化指数 =====================
    food_keywords = ['美食', '小吃', '特色', '餐', '料理', '菜', '茶', '酒', '名吃', 
                    '风味', '传统食', '当地', '特产', '小食', '夜市']
    
    food_score = sum([intro.count(word) for intro in introductions for word in food_keywords])
    features['美食文化指数'] = food_score / len(city_df) if len(city_df) > 0 else 0
    
    # 茶文化特别指标（外国游客感兴趣）
    tea_keywords = ['茶', '茶馆', '茶室', '茶文化', '品茶', '茶艺']
    tea_score = sum([intro.count(word) for intro in introductions for word in tea_keywords])
    features['茶文化指数'] = tea_score / len(city_df) if len(city_df) > 0 else 0
    
    # ===================== 8. 国际化与服务指数 =====================
    # 门票价格合理性
    ticket_prices = []
    free_attractions = 0
    
    for ticket_info in city_df['门票'].fillna('{}'):
        try:
            if isinstance(ticket_info, str) and '免费' in ticket_info:
                free_attractions += 1
            elif isinstance(ticket_info, str) and ticket_info.strip():
                price_nums = re.findall(r'¥(\d+)', str(ticket_info))
                if price_nums:
                    ticket_prices.extend([int(p) for p in price_nums])
        except:
            continue
    
    features['免费景点比例'] = free_attractions / len(city_df) if len(city_df) > 0 else 0
    features['平均门票价格'] = np.mean(ticket_prices) if ticket_prices else 50
    
    # 服务质量（通过小贴士完善度判断）
    tips_quality = len(city_df[city_df['小贴士'].fillna('').str.len() > 20])
    features['服务质量指数'] = tips_quality / len(city_df) if len(city_df) > 0 else 0
    
    return features

def process_all_cities_comprehensive(folder_path):
    """
    处理所有城市数据 - 外国游客综合评价版本
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    all_city_features = []
    
    print(f"发现 {len(csv_files)} 个城市数据文件")
    
    for csv_file in csv_files:
        city_name = Path(csv_file).stem
        
        try:
            city_df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"处理城市: {city_name}, 景点数量: {len(city_df)}")
            
            city_features = extract_city_comprehensive_features(city_df, city_name)
            all_city_features.append(city_features)
            
        except Exception as e:
            print(f"处理 {city_name} 时出错: {e}")
            continue
    
    return pd.DataFrame(all_city_features)

def comprehensive_evaluation_for_foreign_tourists(city_features_df):
    """
    基于外国游客偏好的综合评价和归一化
    """
    # 需要归一化的指标
    feature_cols = [
        '平均景点评分', '高质量景点数', '顶级景点数', '景点总数', '深度游景点比例',
        '历史文化指数', '宗教文化指数', '现代文化指数', '自然生态指数', '户外活动指数',
        '城市现代化指数', '城市覆盖广度', '交通便利指数', '交通枢纽指数',
        '全季节适宜指数', '春夏适宜指数', '美食文化指数', '茶文化指数',
        '免费景点比例', '平均门票价格', '服务质量指数'
    ]
    
    # 归一化处理
    for col in feature_cols:
        if col in city_features_df.columns:
            min_val = city_features_df[col].min()
            max_val = city_features_df[col].max()
            
            # 门票价格是负向指标
            if col == '平均门票价格':
                city_features_df[col + '_norm'] = (max_val - city_features_df[col]) / (max_val - min_val) if max_val != min_val else 0.5
            else:
                city_features_df[col + '_norm'] = (city_features_df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    # 外国游客偏好权重设计
    weights = {
        # 景点吸引力 (30%)
        '平均景点评分_norm': 0.12,
        '高质量景点数_norm': 0.10,
        '顶级景点数_norm': 0.08,
        
        # 人文底蕴 (25%) - 外国游客最感兴趣
        '历史文化指数_norm': 0.15,
        '宗教文化指数_norm': 0.06,
        '现代文化指数_norm': 0.04,
        
        # 自然环境 (15%)
        '自然生态指数_norm': 0.10,
        '户外活动指数_norm': 0.05,
        
        # 城市规模与便利性 (15%)
        '城市现代化指数_norm': 0.05,
        '城市覆盖广度_norm': 0.04,
        '交通便利指数_norm': 0.04,
        '交通枢纽指数_norm': 0.02,
        
        # 气候舒适度 (8%)
        '全季节适宜指数_norm': 0.05,
        '春夏适宜指数_norm': 0.03,
        
        # 美食文化 (5%)
        '美食文化指数_norm': 0.03,
        '茶文化指数_norm': 0.02,
        
        # 性价比与服务 (2%)
        '免费景点比例_norm': 0.01,
        '平均门票价格_norm': 0.005,
        '服务质量指数_norm': 0.005,
    }
    
    # 综合评分计算
    city_features_df['外国游客吸引力得分'] = 0
    for col, weight in weights.items():
        if col in city_features_df.columns:
            city_features_df['外国游客吸引力得分'] += city_features_df[col] * weight
    
    return city_features_df

def main_foreign_tourist_evaluation(folder_path):
    """
    主函数：外国游客视角的城市综合评价
    """
    print("开始基于外国游客偏好的城市综合评价...")
    print("评价维度：景点吸引力、人文底蕴、自然环境、城市规模、交通便利、气候、美食文化")
    print("="*80)
    
    # 处理所有城市
    city_features_df = process_all_cities_comprehensive(folder_path)
    
    if city_features_df.empty:
        print("没有找到有效的城市数据文件")
        return
    
    print(f"\n成功处理 {len(city_features_df)} 个城市")
    
    # 综合评价
    city_features_df = comprehensive_evaluation_for_foreign_tourists(city_features_df)
    
    # 排序并输出TOP50
    top50 = city_features_df.sort_values('外国游客吸引力得分', ascending=False).head(50)
    
    print("\n🌟 最令外国游客向往的50个中国城市 🌟")
    print("="*80)
    
    for i, (idx, row) in enumerate(top50.iterrows(), 1):
        print(f"{i:2d}. {row['city']:12s} - 综合得分: {row['外国游客吸引力得分']:.4f} "
              f"(景点{int(row['景点总数'])}, 评分{row['平均景点评分']:.1f}, "
              f"文化{row['历史文化指数']:.2f})")
    
    # 保存详细结果
    output_columns = ['city', '外国游客吸引力得分', '景点总数', '平均景点评分', '高质量景点数',
                     '历史文化指数', '自然生态指数', '美食文化指数', '免费景点比例']
    
    top50[output_columns].to_csv(
        'top50_cities_for_foreign_tourists_comprehensive.csv', 
        index=False, 
        encoding='utf-8-sig'
    )
    
    # 保存完整评价数据
    city_features_df.to_csv('all_cities_foreign_tourist_evaluation.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n📊 详细结果已保存到: top50_cities_for_foreign_tourists_comprehensive.csv")
    print(f"📊 完整数据已保存到: all_cities_foreign_tourist_evaluation.csv")
    
    return top50

# 使用示例
if __name__ == "__main__":
    # 指定包含所有"城市名.csv"文件的文件夹路径
    folder_path = "./cities_data"  # 请修改为实际路径
    
    top50_cities = main_foreign_tourist_evaluation(folder_path)
    
    # 输出简要分析
    if top50_cities is not None and not top50_cities.empty:
        print("\n📈 TOP10城市特色分析:")
        top10 = top50_cities.head(10)
        for idx, row in top10.iterrows():
            specialty = []
            if row['历史文化指数'] > 0.5: specialty.append("历史悠久")
            if row['自然生态指数'] > 0.5: specialty.append("生态优美") 
            if row['美食文化指数'] > 0.3: specialty.append("美食丰富")
            if row['免费景点比例'] > 0.3: specialty.append("性价比高")
            
            print(f"   {row['city']}: {', '.join(specialty) if specialty else '综合发展'}")