
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer

'''
任务详情
在这个业务问题上，我们的首要任务是识别正在流失的客户。即使我们预测非流失客户已流失，也不会损害我们的业务。
但将流失客户预测为非流失客户就不可以。所以召回率（TP/TP+FN）需要更高。
到目前为止，我的召回率已经达到了 62%。需要更好的
数据集不平衡
数据集没有缺失值
数据集有 10127 个客户和 19 个特征
寻求更好的召回，确保您的目标是 Attrition_Flag = 1 的召回，许多其他笔记本报告错误的召回
'''
'''
功能描述
CLIENTNUM：客户编号。持有账户的客户的唯一标识符
Customer_Age：人口统计变量 - 客户的年龄（岁）
Gender：人口统计变量 - M=男性，F=女性
Dependent_count：人口统计变量 - 受抚养人数量
Education_Level：人口统计变量 - 账户持有人的教育资格（例如：高中、大学毕业生等）
Marital_Status：人口统计变量 - 已婚、单身、离婚、未知
Income_Category：人口统计变量 - 账户持有人的年收入类别（< 40K，40K - 60K, 60K−80K、80K−120k，> 120k，Unknown）
Card_Category：产品变量 - 卡类型（蓝色、银色、金色、白金）
Months_on_book：与银行的关系期限
Total_Relationship_Count：总数。客户持有的产品数量
Months_Inactive_12_mon：过去 12 个月内不活动的月份数
Contacts_Count_12_mon：过去 12 个月的联系人数量
Credit_Limit：信用卡的信用额度
Total_Revolving_Bal：信用卡上的总循环余额
Avg_Open_To_Buy：开放购买信用额度（过去 12 个月的平均值）
Total_Amt_Chng_Q4_Q1：交易金额变化（第 4 季度相对于第 1 季度）
Total_Trans_Amt：总交易金额（过去 12 个月）
Total_Trans_Ct：总交易计数（过去 12 个月）
Total_Ct_Chng_Q4_Q1：交易计数变化（第 4 季度相对于第 1 季度）
Avg_Utilization_Ratio：平均卡使用率
'''

'''
Attrition_Flag：内部事件（客户活动）变量 - 如果帐户关闭，则为 1（流失客户），否则为 0（现有客户）
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import shap
import lightgbm as lgb
import seaborn as sns


from matplotlib import pyplot
from pprint import pprint
from IPython.display import display
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import pi
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import RandomizedSearchCV

#初步了解训练数据
#读取原始数据
data_file = "CSV 文件/BankChurners.csv"
data = pd.read_csv(data_file)
#删除几个无关列
drop_columns = ['CLIENTNUM',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
#删除原数据的这三个属性
data_raw = data.drop(drop_columns, errors ="ignore" ,axis=1)
#初步浏览数据 verbose可以不简化看到的信息
display(data_raw.info(verbose= True))
#检查有没有缺失值
print("没有缺失值")
print(data_raw.isnull().sum())



# 数据可视化
# 绘制饼图 df=Pandas dataframe  categorical_features = 特征列表 dropna = 是否使用NaN的布尔变量
# 输出打印出多个px.pie()
def PlotMultiplePie(df,categorical_features = None,dropna = False):
    #设置 30 个唯一变量的阈值，超过 50 个会导致饼图难看
    threshold = 30
    # 如果用户没有设置 categorical_features
    if categorical_features == None:
        categorical_features = df.select_dtypes(['object', 'category']).columns.to_list()
        print(categorical_features)
    # 循环遍历 categorical_features 列表
    for cat_feature in categorical_features:
        num_unique = df[cat_feature].nunique(dropna=dropna)
        num_missing = df[cat_feature].isna().sum()
    # 如果唯一值低于阈值则打印饼图和信息
        if num_unique <= threshold:
            print("饼图对于",cat_feature)
            print("唯一值的数量",num_unique)
            print("缺失值的数量",num_missing)
            # 计算类别特征的值和计数，同时处理缺失值
            value_counts = df[cat_feature].value_counts(dropna=dropna)
            fig = px.pie(
                values=value_counts.values,  # 使用计数作为饼图的数值
                names=value_counts.index,  # 使用类别作为饼图的标签
                title=cat_feature,  # 设置饼图的标题
                template='ggplot2'  # 设置饼图的模板
            )
            fig.show()
        else:
            print('饼图对于', cat_feature, '由于其太多的唯一值导致其不合适')
            print('类别的数量：', num_unique)
            print('缺失值的数量: ', num_missing)
            print('\n')

#使用PlotMultiplePie查看分类变量的分布

PlotMultiplePie(data_raw)



def PlotMultiplePie_on_one_figure(df, categorical_features=None, dropna=False, ncols=3, nrows=None, palette='viridis'):
    if categorical_features is None:
        categorical_features = df.select_dtypes(['object', 'category']).columns.to_list()

    # 计算需要的行数
    if nrows is None:
        nrows = (len(categorical_features) + ncols - 1) // ncols  # 向上取整

    # 使用 seaborn 生成颜色调色板
    num_colors = max(3, len(categorical_features))  # 至少3种颜色，或根据特征数量
    colors = sns.color_palette(palette, num_colors)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15), squeeze=False)

    # 遍历每个分类特征
    for i, cat_feature in enumerate(categorical_features):
        num_unique = df[cat_feature].nunique(dropna=dropna)

        if num_unique <= 30:  # 设定一个阈值
            ax = axes[i // ncols, i % ncols]
            value_counts = df[cat_feature].value_counts(dropna=dropna, normalize=True) * 100  # 转换为百分比

            # 使用 matplotlib 绘制饼图，并应用颜色
            wedges, texts, autotexts = ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                                              startangle=140, colors=colors[:len(value_counts)])

            # 可选：设置饼图每块的颜色边框，使其更加突出
            plt.setp(wedges, edgecolor='white', linewidth=1)

            ax.set_title(cat_feature)
            ax.set_ylabel('')  # 去除 y 轴标签
        else:
            print(f'对于 {cat_feature}，由于其唯一值过多，不适合绘制饼图')

    plt.tight_layout()  # 调整子图布局
    plt.show()
PlotMultiplePie_on_one_figure(data_raw)



#观察连续型数据
continous_features = data_raw.select_dtypes(["float64"]).columns.to_list()
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(16,10))
axes = axes.flatten()
for i ,count_feature in enumerate(continous_features):
    ax = sns.histplot(data_raw[count_feature], ax=axes[i] ,color= "skyblue",kde=True)
    axes[i].set_title(f"{count_feature} histplot")
#调整布局以减少堆叠
axes[-1].set_visible(False)
plt.tight_layout()
#显示图形
plt.show()


#输出数据集中的离散特征
discrete_features = data_raw.select_dtypes(["int64"]).columns.to_list()
fig1,axes1 = plt.subplots(nrows=3,ncols=3,figsize=(27,15))
axes1 = axes1.flatten()
for i ,disc_feature in enumerate(discrete_features):
    ax = sns.histplot(data_raw[disc_feature], ax=axes1[i] ,color= "green",kde=False)
    axes1[i].set_title(f"{disc_feature} histplot")
#调整布局以减少堆叠
plt.tight_layout()
#显示图形
plt.show()


#绘制连续型数据的箱线图
continous_features = data_raw.select_dtypes(["float64"]).columns.to_list()
fig,axes3 = plt.subplots(nrows=2,ncols=3,figsize=(16,10))
axes3 = axes3.flatten()
for i ,count_feature in enumerate(continous_features):
    ax = sns.boxplot(data_raw[count_feature], ax=axes3[i] ,color= "lightgreen")
    axes3[i].set_title(f"{count_feature} boxplot")
#调整布局以减少堆叠
axes3[-1].set_visible(False)
plt.tight_layout()
#显示图形
plt.show()

#绘制离散型型数据的箱线图
discrete_features = data_raw.select_dtypes(["int64"]).columns.to_list()
fig1,axes2 = plt.subplots(nrows=3,ncols=3,figsize=(27,15))
axes2 = axes2.flatten()
for i ,disc_feature in enumerate(discrete_features):
    ax = sns.boxplot(data_raw[disc_feature],orient="v", ax=axes2[i] ,color= "yellow")
    axes2[i].set_title(f"{disc_feature} boxplot")
#调整布局以减少堆叠
plt.tight_layout()
#显示图形
plt.show()


# 数值型变量和Y的关系的图片
# 假设 data_raw 是您的 DataFrame，并且已经包含了连续型特征和分类目标变量
continous_features1 = data_raw.select_dtypes(["float64"]).columns.to_list()
fig, axes3 = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
axes3 = axes3.flatten()
for i, count_feature in enumerate(continous_features1):
    # 使用 hue 参数根据 Attrition_Flag 的值区分颜色
    ax = sns.scatterplot(data=data_raw, x=count_feature, y="Attrition_Flag", hue="Attrition_Flag", ax=axes3[i],
                         palette="Set1")
    axes3[i].set_title(f"{count_feature} Scatterplot by Attrition_Flag")
    # 移除 y 轴上的刻度线（因为 Attrition_Flag 只有两个值）
    axes3[i].set_yticks([0, 1])
    # 如果需要，可以添加图例
    if i == 0:  # 只在第一个图上添加图例以避免重复
        axes3[i].legend(title="Attrition Flag")
# 调整布局以减少堆叠，但注意这里不需要设置最后一个轴不可见，因为我们已经用 flatten() 处理了所有轴
axes3[-1].set_visible(False)
plt.tight_layout()
# 显示图形
plt.show()


discrete_features1 = data_raw.select_dtypes(["int64"]).columns.to_list()
fig, axes3 = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
axes3 = axes3.flatten()
for i, discreate_feature in enumerate(discrete_features1):
    # 使用 hue 参数根据 Attrition_Flag 的值区分颜色
    ax = sns.scatterplot(data=data_raw, x=discreate_feature, y="Attrition_Flag", hue="Attrition_Flag", ax=axes3[i],
                         palette="Set1")
    axes3[i].set_title(f"{discreate_feature} Scatterplot by Attrition_Flag")
    # 移除 y 轴上的刻度线（因为 Attrition_Flag 只有两个值）
    axes3[i].set_yticks([0, 1])
    # 如果需要，可以添加图例
    if i == 0:  # 只在第一个图上添加图例以避免重复
        axes3[i].legend(title="Attrition Flag")
# 调整布局以减少堆叠，但注意这里不需要设置最后一个轴不可见，因为我们已经用 flatten() 处理了所有轴
plt.tight_layout()
# 显示图形
plt.show()






# 假设data_raw是您的原始DataFrame
# 列出您想要检查的列名
data_want = data_raw.copy()
columns_to_check = ['Education_Level', 'Marital_Status', 'Gender','Income_Category','Card_Category']  # 添加所有相关列名
# 检查这些列中是否包含Unknown值，并对每一行计算一个布尔值（任何列包含Unknown则为True）
rows_with_unknown = data_want[columns_to_check].isin(['Unknown']).any(axis=1)
# 删除包含Unknown值的行
data_cleaned_want = data_want[~rows_with_unknown]
# 验证Unknown值是否已被删除（检查任意一列）
print(data_cleaned_want['Education_Level'].isin(['Unknown']).sum())  # 这应该返回0，对于其他列也同样

#绘制类别和Attrition_Flag的关系
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(15,10), dpi=120)
axes = axes.flatten()
duobaos = ['#AB82FF','salmon']
# 柱状图
for i , feature in enumerate(columns_to_check):
    plt.subplot(231)
    ax = sns.countplot(x=feature,hue="Attrition_Flag",data=data_cleaned_want,ax = axes[i],palette=duobaos, dodge=False)
    axes[i].set_title(f"{feature} count_plot")
#调整布局以减少堆叠
plt.tight_layout()
axes[-1].set_visible(False)
#显示图形
plt.show()

# for i ,disc_feature in enumerate(discrete_features):
#     ax = sns.boxplot(data_raw[disc_feature],orient="v", ax=axes2[i] ,color= "yellow")
#     axes2[i].set_title(f"{disc_feature} boxplot")
# #调整布局以减少堆叠
# plt.tight_layout()
# #显示图形
# plt.show()


'''
从这里开始是客户的分类
'''
columns = ['Card_Category','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon',
           'Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt',
           'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio','Attrition_Flag']

new_data = data_raw.loc[:,columns]
print(new_data.head())

#数据预处理
#查看谁是分类型自变量
s = (new_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables in the dataset:", object_cols)

#对数据进行编码
LE = LabelEncoder()
for i in object_cols:
    new_data[i] = new_data[[i]].apply(LE.fit_transform)
print("All features are now numerical")
print(new_data.head())

#分别按照Attrition_Flag的1，0分类
# 创建一个布尔序列，表示 Attrition_Flag 为1的行（即流失的客户）
lost_customers = new_data['Attrition_Flag'] == 1
# 创建一个布尔序列，表示 Attrition_Flag 为0的行（即没有流失的客户）
retained_customers = new_data['Attrition_Flag'] == 0
# 使用布尔索引提取流失的客户数据集
data_lost = new_data[lost_customers]
# 使用布尔索引提取没有流失的客户数据集
data_retained = new_data[retained_customers]


#创造一个新的与流失客户data_lost相同的数据集
new_data1 = data_lost.copy()
#数据放缩
cols_del = ['Card_Category','Attrition_Flag']
new_data1 = new_data1.drop(cols_del, axis=1)
#Scaling
scaler1 = StandardScaler()
scaler1.fit(new_data1)
scaled_new_data1 =  pd.DataFrame(scaler1.transform(new_data1),columns= new_data1.columns )
print("All features are now scaled")
print(scaled_new_data1.head())

pca = PCA()
pca.fit(scaled_new_data1)
print(pca.components_)
print(pca.explained_variance_ratio_)


#降维并获取降维后的数据PCA_ds
pca = PCA(n_components=7)
pca.fit(scaled_new_data1)
PCA_ds = pd.DataFrame(pca.transform(scaled_new_data1), columns=(["col1","col2", "col3",'col4','col5','col6','col7']))
print(PCA_ds.describe().T)

'''
聚类开始
'''
#利用Elbow 方法来检测聚类个数
Elbow_M = KElbowVisualizer(KMeans(),k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()
#聚4类是最好的
#使用凝聚式聚类（层次聚类的一种）
AC = AgglomerativeClustering(n_clusters=4)
#训练模型
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
data_lost["Clusters"] = yhat_AC
print(data_lost.head())

'''
聚类结束
'''
#模型评估
pl = sns.countplot(x=data_lost["Clusters"])
pl.set_title("Distribution Of The Clusters")
plt.show()
#模型不平衡
# pl = sns.scatterplot(data = data_lost,x=data_lost["Total_Trans_Amt"], y=data_lost["Total_Trans_Ct"],hue=data_lost["Clusters"])
# pl.set_title("Cluster's Profile Based On Total_Trans_Amt And Total_Trans_Ct")
# plt.legend()
# plt.show()


columns1 = ['Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon',
           'Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt',
           'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(15,15))
ax = ax.flatten()
for i , col in enumerate(columns1):
    sns.boxenplot(x=data_lost["Clusters"], y=data_lost[col] , ax = ax[i], color = "skyblue")
    ax[i].set_title(f"{col} boxplot")
plt.tight_layout()
plt.show()

# 复制标准化的数据集
data_scaled_new  = scaled_new_data1
print(data_scaled_new.head())

'''
结束
'''




#数据清洗
#清理数据集，并将适当的列转换为适当的 dtypes()
#将 bool 和 object 转换为 category
cat_types = ['bool' , 'object' ,'category']
data_clean = data_raw.copy()
data_clean[data_clean.select_dtypes(cat_types).columns] = data_clean.select_dtypes(cat_types).apply(lambda x: pd.Categorical(x))
#初步浏览清洗的数据
print(data_clean.info())

#将data_clean 拆分成两个数据集 x-自变量 ，y因变量
#映射 Attririon_Flag :Attrited Customer = 1 和 Existing Customer = 0
codes = {'Existing Customer':0, 'Attrited Customer':1}
data_clean["Attrition_Flag"] = data_clean["Attrition_Flag"].map(codes)

y = data_clean["Attrition_Flag"]
X = data_clean.drop("Attrition_Flag",errors = "ignore" ,  axis=1)

# print(X.head)
#Label Encoding
#分类自变量的标签和独热编码
# 序数变量的标签编码（例如：排名、尺度等）
# 名义变量的单热编码（例如：颜色、性别等）

# 打印每个分类列的类别
for col in X.select_dtypes('category').columns.to_list() :
    print(col + ':' + str(X[col].cat.categories.to_list()))

#对每个分类自变量使用 One Hot Eoncoding
# 因为 Income_Category 具有“未知”值，无法转换为序数变量以使用标签编码
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    dummies = dummies.astype('int8')
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return (res)
'''
函数功能：
使用 pd.get_dummies 函数对 original_dataframe 中指定的 feature_to_encode 列进行独热编码。pd.get_dummies 会为每个唯一值创建一个新的列，并在相应的行中标记为1（如果该行具有该唯一值），否则为0。
使用 pd.concat 函数将独热编码后的结果（dummies）与原始数据框（original_dataframe）沿着列方向（axis=1）合并，这样原始数据框和独热编码后的列都会被保留。
使用 drop 方法删除原始数据框中需要被编码的特征列（feature_to_encode），因为我们已经有了它的独热编码形式。
函数返回一个新的DataFrame对象，它包含了原始数据框的所有列（除了被编码的特征列），以及该特征列的独热编码后的新列。
'''

features_to_encode = X.select_dtypes('category').columns.to_list()
for feature in features_to_encode:
    X = encode_and_bind(X, feature)

X.info()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 简单实现方法
# df_dummies.corr()['是否流失'].sort_values(ascending = False).plot(kind='bar')
merged_df = pd.concat([X, y.to_frame(name="Attrition_Flag")], axis=1)
corr_data = merged_df.corr()['Attrition_Flag'].sort_values(ascending=False)
positive_data = corr_data[corr_data > 0][1:]
negative_data = corr_data[corr_data < 0]
positive_color = '#9F79EE'
negative_color = 'salmon'
fig, ax = plt.subplots(figsize=(24,12), dpi=150)
# 绘制大于0部分的柱状图
positive_bars = ax.bar(positive_data.index, positive_data.values, label='positive correlation', color=positive_color)
# 绘制小于0部分的柱状图
negative_bars = ax.bar(negative_data.index, negative_data.values, label='negative correlation', color=negative_color)
plt.xticks(rotation=90,fontsize=8)
ax.set_title('与是否流失的相关性分布')
# ax.set_xlabel('不同数值化后的字段')
ax.set_ylabel('相关性系数')
ax.legend()
plt.show()






#数据放缩
scaler = MinMaxScaler()
scaler.fit_transform(X)
scaled_features = scaler.transform(X)
scaled_features = pd.DataFrame(scaled_features, columns = X.columns)
print(scaled_features.head())


#分离训练集和测试集
#8 ： 2
X_train80, X_test20, y_train80, y_test20 = train_test_split(scaled_features, y, test_size=0.2, random_state = 0, shuffle= True,stratify = y)
print(y_train80.value_counts())
print(y_test20.value_counts())



#模型训练
#随机森林分类器
#使用 RandomForestClassifier 进行初始拟合
RFC = RandomForestClassifier(n_estimators = 100, random_state = 0)
RFC.fit(X_train80, y_train80)
print("Accuracy: %.2f%%" % ((RFC.score(X_test20,y_test20))*100.0))

#随机森林分类器的混淆矩阵
#绘制未归一化和已归一化的混淆矩阵
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        RFC, X_test20, y_test20,
        display_labels=RFC.classes_,
        cmap=plt.cm.Blues,
        normalize=normalize
    )
    disp.ax_.set_title(title)
    plt.show()
plt.show()

#随机森林模型的性能
#使用给定 X_test20 数据的 RFC 模型进行预测
y_pred20 = RFC.predict(X_test20)
#Precision , Recall  , F1_score
precision_recall_fscore_support(y_test20, y_pred20, average='binary',pos_label=1,beta = 1)
#流失客户分类报告
print(classification_report(y_test20,y_pred20))
print("Accuracy: %.2f%%" % (accuracy_score(y_test20, y_pred20)*100.0))
print("Recall: %.2f%%" % ((recall_score(y_test20,y_pred20))*100.0))

#虽然准确率为 95.66%，但召回率（76.00%）是一个更重要的指标。为了改进这个指标，我们使用 RandomSearchCV 来超级调整参数。


#随机搜索CV
#使用 RandomSearchCV 调整 RFC 参数
#需要一段时间计算并使用大量 CPU
print('Running RandomizedSearchCV')
#默认 RFClassifier 但设置 random_state = 0 以保持结果一致
MOD = RandomForestClassifier(random_state=0)
#实施随机搜索CV
#随机森林中的树木数量 [100,150,...,500]
n_estimators = [int(x) for x in np.arange(start = 100, stop = 501, step = 50)]
#每次分割时要考虑的特征数量
max_features = ['auto', 'sqrt']
#树的最大层数
max_depth = [int(x) for x in np.arange(start = 20, stop = 101, step = 20)]
max_depth.append(None)
#分裂节点所需的最小样本数
min_samples_split = [2, 5, 10]
#每个叶节点所需的最小样本数
min_samples_leaf = [1, 2, 4]
#选择训练每棵树的样本的方法
bootstrap = [True, False]
#创建随机网络
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
scoreFunction = {"recall": "recall"}
#运行 3 次折叠和 25 次迭代的 RandomizedSearchCV
random_search = RandomizedSearchCV(MOD,
                                   param_distributions = random_grid,
                                   n_iter = 25,
                                   scoring = scoreFunction,
                                   refit = "recall",
                                   return_train_score = False,
                                   random_state = 0,
                                   verbose = 2,
                                   cv = 3,
                                   n_jobs = -1)
#训练和优化模型
random_search.fit(X_train80, y_train80)

print('Finished RandomizedSearchCV ')

#评估测试模型性能的方法
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Model Performance')
    print(classification_report(test_labels,predictions))
    print("Accuracy: %.2f%%" % (accuracy_score(test_labels, predictions)*100.0))
    print("Recall: %.2f%%" % ((recall_score(test_labels,predictions))*100.0))


# 评估基本模型 RFC 与来自 RandomizedSearchCV 的改进模型 RFC_search
print("Improved Model from RandomizedSearchCV")
RFC_search = random_search.best_estimator_
RFC_search.set_params(random_state=0)
pprint(RFC_search.get_params())
evaluate(RFC_search,X_test20,y_test20)

#RFC_search 改进模型的混淆矩阵
titles_options1 = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options1:
    disp = ConfusionMatrixDisplay.from_estimator(
        RFC_search, X_test20, y_test20,
        display_labels=RFC.classes_,
        cmap=plt.cm.Blues,
        normalize=normalize
    )
    disp.ax_.set_title(title)
    plt.show()
plt.show()

'''
准确率和召回率均有所提高，准确率：96.10%，召回率：78.15%。虽然这只是一个微小的改进，
但我们可以尝试其他机器学习算法来获得更好的召回率，同时保持准确性。
'''

#随机森林的特征重要性
feat_importances = pd.Series(RFC_search.feature_importances_, index=X_train80.columns)
plt.figure(figsize=(15,10))
feat_importances.nlargest(10).sort_values().plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()



