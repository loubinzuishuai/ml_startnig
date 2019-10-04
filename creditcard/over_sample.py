#导入模块-----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix




#准备训练集和测试集---------------------------------------------------------------------------------------
data = pd.read_csv("C:/学习/python/creditcard/creditcard.csv")
data['normAmount'] = StandardScaler().fit_transform(X=data['Amount'].values.reshape(-1, 1))
data = data.drop(labels=['Time', 'Amount'], axis=1)
X = data.loc[:, data.columns.values != 'Class']
y = data.loc[:, data.columns.values == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



#过采样--------------------------------------------------------------------------------------------------
#生成样本
oversampler = SMOTE()
X_train_oversample, y_train_oversample = oversampler.fit_sample(X=X_train.values, y=y_train.values.ravel())


#开始训练
lr = LogisticRegression(penalty='l2', solver='lbfgs', C=100, max_iter=3000)
lr.fit(X=X_train_oversample, y=y_train_oversample)
y_pred = lr.predict(X=X_test)


#画混淆矩阵
matrix_oversample = confusion_matrix(y_true=y_test.values.ravel(), y_pred=y_pred, labels=[0, 1])
plt.figure()
plt.subplot(2, 2, 1)
plt.title("over_sample")
sns.heatmap(data=matrix_oversample, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0,1], cmap=plt.cm.Blues)
plt.xlabel("prediction label")
plt.ylabel("true label")




#下采样-----------------------------------------------------------------------------------------------
#生成数据
fraud_index = X_train.loc[y_train.values.ravel() == 1, :].index.values
norm_index = X_train.loc[y_train.values.ravel() == 0, :].index.values
norm_index_undersample = np.random.choice(a=norm_index, size=len(fraud_index))
index_undersample = np.concatenate([norm_index_undersample, fraud_index])
np.random.shuffle(index_undersample)
X_train_undersample = X_train.loc[index_undersample, :]
y_train_undersample = y_train.loc[index_undersample, :]


#开始训练
lr1 = LogisticRegression(penalty='l2', solver='lbfgs', C=100, max_iter=3000)
lr1.fit(X=X_train_undersample.values, y=y_train_undersample.values.ravel())
y_pred1 = lr1.predict(X=X_test.values)

#画混淆矩阵
matrix_undersample = confusion_matrix(y_true=y_test.values.ravel(), y_pred=y_pred1, labels=[0, 1])
plt.subplot(2, 2, 2)
plt.title("under_sample")
sns.heatmap(data=matrix_undersample, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0,1], cmap=plt.cm.Blues)
plt.xlabel("prediction label")
plt.ylabel("true label")




#直接训练---------------------------------------------------------------------------------------------
lr2 = LogisticRegression(penalty='l2', solver='lbfgs', C=100, max_iter=3000)
lr2.fit(X=X_train.values, y=y_train.values.ravel())
y_pred2 = lr2.predict(X_test)

#画混淆矩阵
matrix = confusion_matrix(y_true=y_test.values.ravel(), y_pred=y_pred2, labels=[0, 1])
plt.subplot(2, 2, 3)
plt.title("nothing")
sns.heatmap(data=matrix, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0,1], cmap=plt.cm.Blues)
plt.xlabel("prediction label")
plt.ylabel("true label")
plt.show()