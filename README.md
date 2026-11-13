# 数据分析与挖掘实训：心血管患病预测

## 技术栈
Python 3
pandas：数据读取、描述性统计、预处理、哑变量处理
numpy：数值计算（新增 BMI 特征）
matplotlib + seaborn：直方图、饼图、条形图、核密度曲线、箱线图、散点图、热力图
sklearn：逻辑回归、随机森林、梯度提升树、KNN、SVM，MinMaxScaler 离差标准化，GridSearchCV 网格搜索

## 数据来源
数据来自 Kaggle 公开的 Cardiovascular Disease 数据集。共 70000 条记录，其中心血管疾病患者 34979 人，未患病 35021 人，三类特征、12 个输入字段 + 1 个目标字段。

## 字段说明

| 列名 | 含义 |
| --- | --- |
| id | 患者编号 |
| age | 年龄，原始单位为天 |
| gender | 性别，1 为女性，2 为男性 |
| height | 身高，cm |
| weight | 体重，kg |
| ap_hi | 收缩压 |
| ap_lo | 舒张压 |
| cholesterol | 胆固醇水平，1 正常、2 超标、3 严重超标 |
| gluc | 血糖水平，取值含义同上 |
| smoke | 是否吸烟，1 是、0 否 |
| alco | 是否饮酒，1 是、0 否 |
| active | 是否运动，1 是、0 否 |
| cardio | 是否患病，1 患病、0 未患病，作为预测目标 |

## Notebook 分析流程

### 1. 数据读取与展示

在 Jupyter Notebook 中用 `pd.read_csv` 读取，查看字段类型、数据形状（70000 行 × 13 列）、describe 描述性统计。

### 2. 可视化探索性分析

使用多种图表从不同角度观察数据：

- **离散型特征分布直方图**：age、gender、cholesterol、gluc
- **胆固醇水平饼图**：正常人群占 74.8%
- **血糖浓度条形图**：超出和大量超出正常水平约占七分之一
- **年龄直方图**：区间 30–65 岁，50–60 岁人数最多
- **身高/体重/血压核密度曲线**
- **年龄+性别分组条形图**：随年龄增加患病比例上升，性别影响不大
- **身高-体重散点图、分组箱线图**
- **相关性系数热力图**：患病与收缩压、年龄、BMI、体重相关性较强

### 3. 数据预处理

- 年龄字段：`age = round(age / 365)` 天转年
- 剔除身高、体重异常值：身高 < 140cm 且体重 < 40kg，剔除 198 条
- 剔除血压异常值：收缩压限定 `[60, 250]`，舒张压限定 `[30, 180]`，剔除 1215 条
- 去重：`drop_duplicates()`
- 哑变量处理：gender、cholesterol、gluc、smoke、alco 做 `pd.get_dummies`
- 离差标准化：`MinMaxScaler`
- 新增特征：`BMI = weight / (height/100)²`
- 数据集划分：`train_test_split`，测试集占 20%，`random_state=2`

### 4. 模型构建

| 模型 | 实现 | 关键参数 |
| --- | --- | --- |
| 逻辑回归 | `LogisticRegression` | penalty=l2，C=0.6，solver=liblinear，class_weight=balanced |
| 随机森林 | `RandomForestClassifier` | n_estimators=100，random_state=42 |
| 梯度提升树 | `GradientBoostingClassifier` | n_estimators=100，learning_rate=0.1，max_depth=4 |
| KNN | `KNeighborsClassifier` | n_neighbors=10 |
| SVM | `SVC` | gamma=auto |

### 5. 模型评估

| 模型 | F1 | 准确率 |
| --- | --- | --- |
| 逻辑回归 | 0.73 | 72.83% |
| 随机森林 | 0.72 | 71.88% |
| 梯度提升树 | 0.74 | 73.58% |

梯度提升树效果最好，对其做进一步优化。

### 6. 网格搜索调参

使用 `GridSearchCV`（`scoring='roc_auc'`，五折交叉验证），依次搜索 n_estimators、max_features、max_depth 以及三者联合空间，找到最优参数后重新训练。

## 项目结论

- **GBDT 是三种模型中效果最好的**，准确率 73.58%，F1 为 0.74。
- 对患病诊断影响较大的特征：**收缩压 ap_hi、年龄 age、体重指数 BMI、体重 weight**。
- 可视化遇到离散型数据难以直接画图时，可以先用 `value_counts()` 分类，再用饼图、条形图展示。
- GridSearchCV 中 scoring 参数必须显式指定为 `roc_auc`，否则返回的准确率可能为负数。

## 运行方式

1. 安装依赖：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. 打开 Jupyter Notebook：

```bash
jupyter notebook 心血管患病预测.ipynb
```

