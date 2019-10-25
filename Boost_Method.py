import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


os.getcwd()
data = pd.read_csv("../Data/otto_train.csv") # Product Category


nCar = data.shape[0]
nVar = data.shape[1]

data = data.drop(['id'], axis = 1) # id 제거


mapping_dict = {"Class_1": 1,
                "Class_2": 2,
                "Class_3": 3,
                "Class_4": 4,
                "Class_5": 5,
                "Class_6": 6,
                "Class_7": 7,
                "Class_8": 8,
                "Class_9": 9}
after_mapping_target = data['target'].apply(lambda x: mapping_dict[x])

feature_columns = list(data.columns.difference(['target']))
X = data[feature_columns]
y = after_mapping_target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42) # 학습데이터와 평가데이터의 비율을 8:2 로 분할| 

# !pip install xgboost
import xgboost as xgb
import time
start = time.time()
xgb_dtrain = xgb.DMatrix(data = train_x, label = train_y)
xgb_dtest = xgb.DMatrix(data = test_x)
xgb_param = {'max_depth': 10,
         'learning_rate': 0.01,
         'n_estimators': 100,
         'objective': 'multi:softmax',
        'num_class': len(set(train_y)) + 1}
xgb_model = xgb.train(params = xgb_param, dtrain = xgb_dtrain) 
xgb_model_predict = xgb_model.predict(xgb_dtest)
print("Accuracy: %.2f" % (accuracy_score(test_y, xgb_model_predict) * 100), "%")
print("Time: %.2f" % (time.time() - start), "seconds")


## 2. LightGBM

# !pip install lightgbm
import lightgbm as lgb
start = time.time() 
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y)
lgb_param = {'max_depth': 10,
            'learning_rate': 0.01,
            'n_estimators': 100,
            'objective': 'multiclass',
            'num_class': len(set(train_y)) + 1} 
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) 
lgb_model_predict = np.argmax(lgb_model.predict(test_x), axis = 1) 
print("Accuracy: %.2f" % (accuracy_score(test_y, lgb_model_predict) * 100), "%") 
print("Time: %.2f" % (time.time() - start), "seconds") 


## 3. Catboost

# !pip install catboost
import catboost as cb
start = time.time()
cb_dtrain = cb.Pool(data = train_x, label = train_y)
cb_param = {'max_depth': 10,
            'learning_rate': 0.01,
            'n_estimators': 100, 
            'eval_metric': 'Accuracy', 
            'loss_function': 'MultiClass'} 
cb_model = cb.train(pool = cb_dtrain, params = cb_param) 
cb_model_predict = np.argmax(cb_model.predict(test_x), axis = 1) + 1 
print("Accuracy: %.2f" % (accuracy_score(test_y, cb_model_predict) * 100), "%")
print("Time: %.2f" % (time.time() - start), "seconds")