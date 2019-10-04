#导入三大件----------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report




data = pd.read_csv("C:/学习/python/creditcard/creditcard.csv")
cout_classes = pd.value_counts(data['Class'], sort=True).sort_index()


#将Amount列归一化。--------------------------------------------------------------------------------------
data['normAmount'] = StandardScaler().fit_transform(X=data['Amount'].values.reshape(-1, 1))
data = data.drop(["Time", "Amount"], axis=1)


#下采样-------------------------------------------------------------------------------------------------
X = data.iloc[:, data.columns != "Class"]
y = data.iloc[:, data.columns == "Class"]

number_records_fraud = len(data[data["Class"] == 1])
fraud_indices = data[data["Class"] == 1].index.values           #ndarray类型
normal_indices = data[data["Class"] == 0].index         #index类型

random_normal_indices = np.random.choice(normal_indices, size=number_records_fraud, replace=False)#随机选出正样本
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']




#产生训练集和测试集--------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                    y_undersample,
                                                                                                    test_size=0.3)


#逻辑回归训练（k折交叉验证，正则化惩罚，召回率的运用)-----------------------------------------------------------
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)

    # Different C parameters
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range)), columns=['C_parameter', 'Mean_recall_score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')
        index = fold.split(x_train_data)
        recall_accs = []
        for iteration, indices in enumerate(index, start=1):
            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C=c_param, penalty='l2', solver='lbfgs', max_iter=3000)

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0], :].values, y_train_data.iloc[indices[0], :].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(X_train.values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train.values.ravel(), y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.loc[[j], ['Mean_recall_score']] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    results_table['Mean_recall_score'] = results_table['Mean_recall_score'].astype('float64')
    best_c = results_table.loc[results_table['Mean_recall_score'].idxmax(), ['C_parameter']]

    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c

best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)


#画出混淆矩阵，计算回召率-----------------------------------------------------------------------------
def print_confussion_matrix(matrix, cmp = plt.cm.Blues):
    sns.heatmap(data=matrix,cmap=cmp, annot=True, fmt='d')


lr = LogisticRegression(penalty='l2', C=best_c.values[0], solver='lbfgs', max_iter=3000) #best_c是Series类型
lr.fit(X_train.values, y_train.values.ravel())
y_pred = lr.predict(X_test.values)

matrix = confusion_matrix(y_true=y_test.values.ravel(), y_pred=y_pred)
print(matrix)
plt.figure()
print_confussion_matrix(matrix)
plt.show()