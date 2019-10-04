#导入库 -----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score



#准备训练数据-------------------------------------------------------------------------------------------
#载入数据
data = pd.read_csv("C:/学习/python/creditcard/creditcard.csv")

#归一化某些特征并去掉无用特征
data["normAmount"] = StandardScaler().fit_transform(X=data["Amount"].values.reshape(-1, 1))
data = data.drop(["Amount", "Time"], axis=1)

#分解为训练集和测试集
X_data = data.loc[:, data.columns.values != "Class"]
y_data = data.loc[:, data.columns.values == "Class"]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)


#下采样
fraud_index = data.loc[data['Class'] == 1, :].index.values
norm_index = data.loc[data['Class'] == 0, :].index.values
norm_index_undersample = np.random.choice(a=norm_index, size=len(fraud_index))
index_undersample = np.concatenate([norm_index_undersample, fraud_index])
data_undersample = data.loc[index_undersample, :]
X_data_undersample = data_undersample.loc[:, data_undersample.columns.values != "Class"]
y_data_undersample = data_undersample.loc[:, data_undersample.columns.values == "Class"]
X_train_undersample, X_test_undersample, y_train_undersample,y_test_undersample = train_test_split(X_data_undersample,
                                                                                                   y_data_undersample,
                                                                                                   test_size=0.3)




#建立训练模型-------------------------------------------------------------------------------------------
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000)
lr.fit(X=X_train_undersample.values, y=y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)




#设置阈值，画混淆矩阵---------------------------------------------------------------------------------------
threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
j = 1
plt.figure()
plt.subplots_adjust(hspace=0.4)
for i in threshold:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i #元素为布尔值的行向量
    plt.subplot(3, 3, j)
    j = j+1
    cnf_matrix = confusion_matrix(y_true=y_test_undersample, y_pred=y_test_predictions_high_recall, labels=[0, 1])
    recall = recall_score(y_true=y_test_undersample.values.ravel(), y_pred=y_test_predictions_high_recall, pos_label=1)
    recall = round(recall, 4)
    title = "threshold:" + str(i) + " recall_score: " + str(recall)
    print(title)
    plt.title(title)

    sns.heatmap(data=cnf_matrix, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predict_label")
    plt.ylabel("True_label")

plt.show()



