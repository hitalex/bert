# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:52:36 2018

@author: yuqi wang and kong
"""
import numpy as np
#import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

##需要导入train_featv2.cvs数据
##X为除label外的特征集，y为label列
#需要注意应转换y值为数值，函数声明如下
def label_encode(x):
    if x =="unrelated":
        return 0
    elif x =="agreed":
        return 1
    else:
        return 2  
#train['target'] = train['label'].apply(lambda x: label_encode(x)) 
        
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

label_index_map = {'unrelated': 0, 'agreed':1, 'disagreed': 2}
class_weight={0: 1/16, 1: 1/15, 2: 1/5, 'unrealted': 1/16, 'agreed': 1/15, 'disagreed':1/5}

def train_test_xgboost(X_train, X_test, y_train, y_test):

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {  
                'objective':'multi:softprob',
                'num_class':3,
                'eta':0.01,
                'colsample_bytree':0.3,
                'learning_rate':0.1, 
                'max_depth':50, 
                'min_child_weight': 3,
                'subsample': 0.9,
                }

    num_round = 20
    plst = params.items()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    print('Training with xgboost...')
    bst = xgb.train(plst, dtrain, num_round)
    # 对测试集进行预测
    print('Predicting with xgboost...')
    pred_prob = bst.predict(dtest)

    pred = np.argmax(pred_prob, axis = 1)

    acc = weightAccuracy(y_test, pred)

    return pred

def weightAccuracy(y,y_hat):
    y_true_sum=0
    y_hat_sum=0
    for i in range(len(y)):
        y_true_sum+= class_weight[y[i]]
        if y[i] == y_hat[i]:
            y_hat_sum += class_weight[y[i]]
        else:
            continue
    print('y_true_sum: %.3f'%y_true_sum)
    print('y_hat_sum: %.3f'%y_hat_sum)
    print(classification_report(list(y), list(y_hat)))
    print('weighted accuracy: %f'%(y_hat_sum/y_true_sum))
    
    return y_hat_sum/y_true_sum

def prepare_features(csv_path):
    ''' 根据csv文件，导出features和label
    Input:
        csv_path: csv文件
    Output:
        features, label: 特征和label
    '''
    import pandas as pd
    column_list = ['cnt', 'bigDis', 'jaccSim', 'masiSim', 'cosSim', 'w2v', 'topic_sim']
    for i in range(2):
        column_list.append('pos_' + str(i+1))
        column_list.append('neg_' + str(i+1))
        column_list.append('neu_' + str(i+1))
        column_list.append('compound_' + str(i+1))

    for i in range(2):
        for j in range(50):
            column_list.append('title' + str(i+1) + '_topic_' + str(j+1))

    data = pd.read_csv(csv_path)

    feature_df = data[column_list]

    features = feature_df.values
    labels = data['label'].values

    for i in range(len(labels)):
        labels[i] = label_index_map[labels[i]]

    return features, labels

if __name__ == '__main__':

    print("Weight Accuracy:",weightAccuracy(y_test.values, preds))
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))
