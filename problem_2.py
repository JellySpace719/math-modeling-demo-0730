import pandas as pd

# 指标列
index_cols = ['景点评分', '规模', '交通', '气候', '环保', '人文', '美食', '国际化']
# 其中，假如'环保'是值越小越好（比如pm2.5），其他值越大越好

df = pd.read_csv('city_data_raw.csv')

# 归一化
for col in index_cols:
    the_min = df[col].min()
    the_max = df[col].max()
    if col == '环保':  # 假如'环保'是极小型，反向归一化
        df[col+'_norm'] = (the_max - df[col]) / (the_max - the_min)
    else:
        df[col+'_norm'] = (df[col] - the_min) / (the_max - the_min)

# 权重
weight = {
    "景点评分_norm": 0.25,
    "规模_norm": 0.10,
    "交通_norm": 0.15,
    "气候_norm": 0.10,
    "环保_norm": 0.10,
    "人文_norm": 0.15,
    "美食_norm": 0.10,
    "国际化_norm": 0.05
}

# 计算综合得分
df['综合得分'] = (
    df['景点评分_norm'] * weight['景点评分_norm'] + 
    df['规模_norm']      * weight['规模_norm'] + 
    df['交通_norm']      * weight['交通_norm'] +
    df['气候_norm']      * weight['气候_norm'] +
    df['环保_norm']      * weight['环保_norm'] + 
    df['人文_norm']      * weight['人文_norm'] +
    df['美食_norm']      * weight['美食_norm'] +
    df['国际化_norm']    * weight['国际化_norm']
)

# 选出TOP 50
top50 = df.sort_values('综合得分', ascending=False).head(50)
print(top50[['city', '综合得分']])
top50.to_csv('top50_cities.csv', index=False, encoding='utf-8-sig')