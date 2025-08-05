### Problem1红宝书

### 问题一：母亲身心健康对婴儿成长的影响 — 规律性研究

**核心目标：** 探究母亲的身体指标、心理指标与婴儿行为特征、睡眠质量之间是否存在统计学上的显著关联或影响规律。

#### 1. 数据加载与初步审查

*   **目的：** 将CSV数据读入内存，并对数据结构进行初步了解。
*   **Python 思路：**
    *   导入 `pandas` 库。
    *   使用 `pd.read_csv()` 读取 '数据.csv' 文件。
    *   使用 `df.head()` 查看前几行数据，`df.info()` 查看数据类型和非空值数量，`df.describe()` 获取数值列的描述性统计。
    *   `df.shape` 确认数据行数和列数。
    *   注意题目说明只使用前390条数据进行研究，所以后续处理应基于 `df.iloc[:390]`。

#### 2. 数据预处理与特征工程

*   **目的：** 清洗数据，将原始数据转换为适合数学模型处理的格式。
*   **Python 思路：**
    1.  **限定研究范围：**
        *   创建新的DataFrame，只包含前390行数据，因为题目问题一的研究只涉及已有的完整数据。
        *   `df_research = df.iloc[:390].copy()`
    2.  **处理“整晚睡眠时间（时：分：秒）”列：**
        *   定义一个函数，将“HH:MM”格式转换为浮点数小时，并处理“99:99”异常值。
        *   `def convert_sleep_time(time_str):`
        *   `if time_str == '99:99': return np.nan`
        *   `hours, minutes = map(int, time_str.split(':'))`
        *   `return hours + minutes / 60.0`
        *   对DataFrame的该列应用此函数。
        *   `df_research['整晚睡眠时间（小时）'] = df_research['整晚睡眠时间（时：分：秒）'].apply(convert_sleep_time)`
        *   处理转换后可能产生的缺失值（即原99:99的记录）：使用整列的**中位数**进行填充，因为它对异常值更不敏感。
        *   `median_sleep_time = df_research['整晚睡眠时间（小时）'].median()`
        *   `df_research['整晚睡眠时间（小时）'].fillna(median_sleep_time, inplace=True)`
        *   删除原始“时：分：秒”列。
        *   `df_research.drop('整晚睡眠时间（时：分：秒）', axis=1, inplace=True)`
    3.  **分类变量的映射与编码：**
        *   **定义映射字典：** 根据题目附件中的“补充说明”，为 `婚姻状况`、`教育程度`、`分娩方式`、`婴儿性别`、`入睡方式`、`婴儿行为特征`创建映射字典。
            *   `marital_status_map = {1: '未婚', 2: '已婚', 3: '其他'}` (注意：数据中婚姻状况有3，但说明中只有1,2，需根据数据实际情况调整或补充)
            *   `education_map = {1: '小学', 2: '初中', 3: '高中', 4: '大学', 5: '研究生'}`
            *   `delivery_method_map = {1: '自然分娩', 2: '剖宫产'}`
            *   `baby_gender_map = {1: '男性', 2: '女性'}`
            *   `sleep_method_map = {1: '哄睡法', 2: '抚触法', ..., 5: '定时法'}`
            *   `baby_behavior_map = {'安静型': 0, '中等型': 1, '矛盾型': 2}` (用于逻辑回归的数值编码)
        *   **应用映射：** 对对应列使用 `df.replace()` 或 `df.map()` 进行文本标签替换。
        *   `df_research['婚姻状况'].replace(marital_status_map, inplace=True)`
        *   **独热编码：**
            *   识别需要独热编码的特征列：`婚姻状况`、`教育程度`、`分娩方式`、`婴儿性别`、`入睡方式`。
            *   使用 `pd.get_dummies()` 对这些列进行独热编码。选择 `drop_first=True`。
            *   `df_encoded = pd.get_dummies(df_research, columns=[...], drop_first=True)`
        *   **因变量数值编码：** 将`婴儿行为特征`列映射为数值类型（0, 1, 2），作为逻辑回归的因变量。
        *   `df_encoded['婴儿行为特征_encoded'] = df_encoded['婴儿行为特征'].map(baby_behavior_map)`
    4.  **特征标准化：**
        *   识别所有要进入模型的数值型自变量：`母亲年龄`、`妊娠时间（周数）`、`CBTS`、`EPDS`、`HADS`、`婴儿年龄（月）`，以及所有独热编码后生成的列。
        *   使用 `StandardScaler` 对这些列进行拟合和转换。
        *   `from sklearn.preprocessing import StandardScaler`
        *   `scaler = StandardScaler()`
        *   `df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])`
        *   **注意：** 独热编码后的列也应视为数值型特征，一并进行标准化，但它们通常是0/1值，标准化对它们影响小，但统一处理更规范。

#### 3. 探索性数据分析 (EDA) 与初步相关性分析

*   **目的：** 通过统计量和可视化方法，直观地发现变量间的初步关联和数据分布特征。
*   **Python 思路：**
    1.  **描述性统计：**
        *   再次对预处理后的 `df_encoded` 使用 `df_encoded.describe()` 和 `df_encoded.info()` 进行审查，确认数据类型和统计量正确。
    2.  **数值型变量间的相关性分析（皮尔逊相关系数）：**
        *   选择所有数值型母亲指标（包括标准化的原始数值型和独热编码后的）与婴儿睡眠指标（`整晚睡眠时间（小时）`、`睡醒次数`）。
        *   计算相关系数矩阵。
        *   `correlation_matrix = df_encoded[selected_numerical_cols].corr()`
        *   使用 `seaborn.heatmap()` 可视化该矩阵。
        *   `import seaborn as sns; import matplotlib.pyplot as plt`
        *   `plt.figure(figsize=(...))`
        *   `sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")`
        *   `plt.title('相关系数热力图')`
        *   `plt.show()`
        *   **解释：** 报告哪些特征之间存在显著的正相关或负相关，并讨论其强度。
    3.  **分类变量对数值变量的影响（单因素方差分析 ANOVA）：**
        *   **目标：** `婚姻状况`、`教育程度`、`分娩方式`、`婴儿性别`、`入睡方式`对`整晚睡眠时间（小时）`和`睡醒次数`的影响。
        *   对于每个分类自变量，遍历其所有类别，将对应因变量的数据分组。
        *   使用 `scipy.stats.f_oneway()` 对各组数据进行ANOVA检验。
        *   `from scipy.stats import f_oneway`
        *   `for cat_col in ['婚姻状况', '教育程度', ...]:`
        *   `for num_col in ['整晚睡眠时间（小时）', '睡醒次数']:`
        *   `groups = [df_encoded[num_col][df_encoded[cat_col] == category] for category in df_encoded[cat_col].unique()]`
        *   `f_statistic, p_value = f_oneway(*groups)`
        *   `print(f'ANOVA for {cat_col} vs {num_col}: F={f_statistic:.2f}, p={p_value:.3f}')`
        *   **可视化：** 使用 `seaborn.boxplot()` 或 `violinplot` 展示不同类别下因变量的分布。
        *   `sns.boxplot(x=cat_col, y=num_col, data=df_encoded)`
        *   `plt.title(f'{num_col} by {cat_col}')`
        *   `plt.show()`
        *   **解释：** 报告P值小于0.05的显著影响，并结合箱线图解释均值差异。
    4.  **分类变量间的关联性（卡方检验）：**
        *   **目标：** `婚姻状况`、`教育程度`、`分娩方式`、`婴儿性别`、`入睡方式`与`婴儿行为特征`之间的关联。
        *   对于每对分类变量，构建列联表。
        *   `contingency_table = pd.crosstab(df_encoded[cat_col1], df_encoded[cat_col2])`
        *   使用 `scipy.stats.chi2_contingency()` 进行卡方检验。
        *   `from scipy.stats import chi2_contingency`
        *   `chi2, p_value, _, _ = chi2_contingency(contingency_table)`
        *   `print(f'Chi-squared for {cat_col1} vs {cat_col2}: Chi2={chi2:.2f}, p={p_value:.3f}')`
        *   **可视化：** 使用 `seaborn.countplot()` 绘制各分类变量与婴儿行为特征或入睡方式的分布。
        *   `sns.countplot(x=cat_col1, hue=cat_col2, data=df_encoded)`
        *   `plt.title(f'Distribution of {cat_col2} by {cat_col1}')`
        *   `plt.show()`
        *   **解释：** 报告P值，判断分类变量之间是否存在显著关联。

#### 4. 多元回归分析

*   **目的：** 建立母亲指标与婴儿指标之间的定量关系模型，量化每个母亲指标对婴儿指标的具体影响，并进行统计显著性检验。
*   **Python 思路：**
    1.  **定义自变量 (X) 和因变量 (Y)：**
        *   **X：** 从 `df_encoded` 中选择所有独热编码和标准化后的自变量列。
        *   `X_cols = [col for col in df_encoded.columns if col not in ['编号', '婴儿行为特征', '婴儿行为特征_encoded', '整晚睡眠时间（小时）', '睡醒次数']]`
        *   `X = df_encoded[X_cols]`
        *   **添加常数项：** `statsmodels` 需要手动添加截距项。
        *   `import statsmodels.api as sm`
        *   `X = sm.add_constant(X)`
    2.  **多元线性回归（针对连续型婴儿指标）：**
        *   **Y1 (`整晚睡眠时间（小时）`)：**
            *   `model_sleep_time = sm.OLS(df_encoded['整晚睡眠时间（小时）'], X).fit()`
            *   `print(model_sleep_time.summary())`
        *   **Y2 (`睡醒次数`)：**
            *   `model_wake_count = sm.OLS(df_encoded['睡醒次数'], X).fit()`
            *   `print(model_wake_count.summary())`
        *   **解释：**
            *   **模型整体显著性：** 查看 `F-statistic` 和其对应的 `Prob (F-statistic)`。
            *   **拟合优度：** 查看 `R-squared` 和 `Adj. R-squared`。
            *   **个体变量影响：** 逐一查看每个自变量的 `coef` (回归系数) 和 `P>|t|` (P值)。
                *   `coef` 的符号和大小表示其对因变量的量化影响。
                *   `P>|t|` 小于0.05（或0.01等）表示该影响在统计上是显著的。
            *   **多重共线性检查（可选但推荐）：** 计算VIF值。
                *   `from statsmodels.stats.outliers_influence import variance_inflation_factor`
                *   `vif_data = pd.DataFrame()`
                *   `vif_data["feature"] = X.columns`
                *   `vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]`
                *   `print(vif_data)`
                *   VIF值过高（如大于5或10）提示存在多重共线性，可能需要采取特征选择或降维措施。
    3.  **多项逻辑回归（针对分类型婴儿指标）：**
        *   **Y3 (`婴儿行为特征_encoded`)：**
            *   `model_baby_behavior = sm.MNLogit(df_encoded['婴儿行为特征_encoded'], X).fit()`
            *   `print(model_baby_behavior.summary())`
        *   **解释：**
            *   `MNLogit` 会选择因变量的第一个类别（即0，通常对应“安静型”）作为参考类别。
            *   模型输出会为每个非参考类别（“中等型”、“矛盾型”）提供一组回归系数和P值。
            *   **发生比 (Odds Ratio)：** 对系数取指数 `np.exp(coef)` 可以得到发生比。例如，如果 `EPDS` 对“矛盾型”的系数是0.1，则发生比为 `e^0.1`，表示EPDS每增加1分，婴儿是“矛盾型”（而非参考类别）的发生几率是原来的 `e^0.1` 倍。
            *   同样关注P值，判断统计显著性。

#### 5. 结果整合与报告撰写

*   **结构：** 按照数据预处理、探索性分析发现、线性回归结果、逻辑回归结果的逻辑顺序组织。
*   **可视化：** 插入所有生成的热力图、箱线图、柱状图等，并配以清晰的图题和说明。
*   **结论：**
    *   **总体规律：** 总结母亲的身体指标和心理指标**确实存在**对婴儿行为特征和睡眠质量的影响规律。
    *   **具体影响：** 详细阐述通过相关性分析、ANOVA和回归模型发现的具体规律。
        *   例如：“母亲的EPDS得分与婴儿的整晚睡眠时间呈显著负相关（r=X.XX, p<0.001），且在控制其他因素后，EPDS得分每增加1分，婴儿整晚睡眠时间平均减少0.YY小时（β=-0.YY, p<0.001）。”
        *   例如：“母亲的HADS得分显著增加了婴儿行为特征为矛盾型而非安静型的发生几率（Odds Ratio = Z.ZZ, p<0.01）。”
    *   **影响强度排序：** 根据标准化回归系数的绝对值（对于线性回归）或发生比的偏离1的程度（对于逻辑回归），对各个母亲指标的影响强度进行排序和讨论。
    *   **局限性：** 提及模型基于线性假设，且相关性不等于因果关系。
    *   **建议和展望：** 基于发现的规律，可提出一些初步的、有针对性的干预或未来研究方向的建议。

