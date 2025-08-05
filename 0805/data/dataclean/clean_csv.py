import pandas as pd
import numpy as np
import re

# 1. 读入数据
df = pd.read_csv('data.csv', encoding='utf-8')

# 2. 统一空值
df.replace(['', 'NA', 'NaN', 'nan', 'None', '--'], np.nan, inplace=True)

# 3. 检查“整晚睡眠时间(时：分：秒)”并处理
def fix_sleep_time(val):
    if pd.isnull(val):
        return np.nan
    # 合法格式检测
    m = re.match(r'^\s*(\d{1,2}):(\d{1,2}):?(\d{0,2})\s*$', str(val))
    if m:
        h, m_, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        # 睡眠小时不应大于14，分钟不大于59，秒不大于59
        if 0 <= h <= 14 and 0 <= m_ < 60 and 0 <= s < 60:
            return f"{h:02d}:{m_:02d}"
        else:
            # 异常时间全部设为缺失
            return np.nan
    else:
        return np.nan

df['整晚睡眠时间（时：分：秒）'] = df['整晚睡眠时间（时：分：秒）'].apply(fix_sleep_time)

# 4. 常见字段异常清理
# 母亲年龄（合理范围：15-55）
df.loc[~df['母亲年龄'].between(15, 55), '母亲年龄'] = np.nan

# 妊娠周数（合理范围：22-44）
df.loc[~df['妊娠时间（周数）'].between(22, 44), '妊娠时间（周数）'] = np.nan

# 婴儿性别（应为1或2），其它设为NaN
df.loc[~df['婴儿性别'].isin([1,2]), '婴儿性别'] = np.nan

# 婴儿年龄（月）（一般0-24）
df.loc[~df['婴儿年龄（月）'].between(0, 24), '婴儿年龄（月）'] = np.nan

# 睡醒次数、入睡方式（一般0-10）
for f in ['睡醒次数','入睡方式']:
    df.loc[~df[f].between(0,10), f] = np.nan

# 处理整数列的数据类型，避免一位小数显示
int_columns = ['母亲年龄', '婚姻状况', '教育程度',
               '分娩方式', 'CBTS', 'EPDS', 'HADS',
               '婴儿性别', '婴儿年龄（月）', '睡醒次数', '入睡方式']
for col in int_columns:
    if col in df.columns:
        df[col] = df[col].astype('Int64')

# 6. 输出异常报告
with open('error_data.txt', 'w', encoding='utf-8') as fout:
    for col in df.columns:
        wrong_cnt = df[col].isnull().sum()
        if wrong_cnt > 0:
            fout.write(f'【{col}】异常或缺失数量：{wrong_cnt}\n')

# 7. 导出清洗后新csv
df.to_csv('cleaned_data.csv', index=False, encoding='utf-8-sig')

print('数据清洗完成！生成文件：cleaned_data.csv，error_data.txt')