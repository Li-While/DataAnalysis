# 数据分析与挖掘实训：心血管患病预测

## 项目简介

数据文件 `cardio_train.csv` 是心血管疾病调查数据，共 1000 行（含表头）、13 列。`心血管患病预测.ipynb` 完成数据读取、可视化探索、清洗、特征处理、相关性分析和多模型分类对比，最后用网格搜索调参。

## 数据说明

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

1. 读取与概览：`head`、`shape`、`columns`、`dtypes`、`info`、`describe`。
2. 探索性可视化：年龄、身高、体重、血压的直方图；胆固醇分布饼图；血糖分布条形图；年龄、性别与患病情况的计数图。
3. 预处理：
   - 年龄由天换算为年（除以 365 后取整）。
   - 剔除身高小于 140 cm 或体重小于 40 kg 的记录。
   - 把收缩压限制在 60 至 250、舒张压限制在 30 至 180，剔除超出范围的记录。
   - 新增 BMI 特征：体重除以身高的平方米。
   - 检查缺失值并尝试去重、去缺失。
4. 特征编码：对 `gender`、`cholesterol`、`gluc`、`smoke`、`alco` 做哑变量处理（`pd.get_dummies`）。
5. 相关性分析：计算相关系数矩阵，绘制带掩码的热力图。
6. 建模：用 `MinMaxScaler` 归一化，按 8:2 划分训练集与测试集，依次训练并输出分类报告和准确率。
   - 逻辑回归（L2，C=0.6，liblinear，类别平衡）
   - 随机森林（100 棵树）
   - 梯度提升树（100 棵，学习率 0.1，最大深度 4）
   - K 近邻（k=10）
   - 支持向量机（RBF，gamma='auto'）
7. 调参：用 `GridSearchCV` 以 `roc_auc` 为评分、5 折交叉验证，依次搜索 `n_estimators`、`max_features`、`max_depth`，最后按最优参数重训梯度提升树并评估。


## 环境与依赖

- Python 3.x，Jupyter Notebook 或 JupyterLab
- `pandas`、`numpy`
- `matplotlib`、`seaborn`
- `scikit-learn`

## 运行方式

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook 心血管患病预测.ipynb
```

