import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("../Data/kc_house_data.csv") 

# Remove Variable
data = data.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis = 1)

# Split Data
feature_columns = list(data.columns.difference(['price']))
X = data[feature_columns]
y = data['price']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) 

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Use Statistic Method to compare
sm_train_x = sm.add_constant(train_x, has_constant = 'add') 
sm_model = sm.OLS(train_y, sm_train_x) 
fitted_sm_model = sm_model.fit() 
fitted_sm_model.summary() 

# Statistic Method Result
sm_test_x = sm.add_constant(test_x, has_constant = 'add') 
sm_model_predict = fitted_sm_model.predict(sm_test_x)

# Use Bagging
bagging_predict_result = []
for _ in range(30):
    data_index = [data_index for data_index in range(train_x.shape[0])]
    random_data_index = np.random.choice(data_index, train_x.shape[0])
    sm_train_x = train_x.iloc[random_data_index, ] 
    sm_train_y = train_y.iloc[random_data_index, ]
    sm_train_x = sm.add_constant(sm_train_x, has_constant = 'add')
    sm_model = sm.OLS(sm_train_y, sm_train_x)
    fitted_sm_model = sm_model.fit()
    
    sm_test_x = sm.add_constant(test_x, has_constant = 'add')
    sm_model_predict = fitted_sm_model.predict(sm_test_x)
    bagging_predict_result.append(sm_model_predict)

# Bagging Result
bagging_predict = []
for lst2_index in range(test_x.shape[0]): 
    temp_predict = [] #
    for lst_index in range(len(bagging_predict_result)): 
        temp_predict.append(bagging_predict_result[lst_index].values[lst2_index]) 
    bagging_predict.append(np.mean(temp_predict))

# Use Scikit Learn without Bagging
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
linear_model1 = regression_model.fit(train_x, train_y) 
predict1 = linear_model1.predict(test_x) 

# Use scikit Learn with Bagging
from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(base_estimator = regression_model, 
                                 n_estimators = 5, 
                                 verbose = 1) 
linear_model2 = bagging_model.fit(train_x, train_y) 
predict2 = linear_model2.predict(test_x) 

# Sampling More!
bagging_model2 = BaggingRegressor(base_estimator = regression_model, 
                                  n_estimators = 30, 
                                  verbose = 1) 
linear_model3 = bagging_model2.fit(train_x, train_y)
predict3 = linear_model3.predict(test_x) 

# Using Decision Tree without Bagging
from sklearn.tree import DecisionTreeRegressor
decision_tree_model = DecisionTreeRegressor()
tree_model1 = decision_tree_model.fit(train_x, train_y)
predict1 = tree_model1.predict(test_x)

# Using Decision Tree with Bagging
bagging_decision_tree_model1 = BaggingRegressor(base_estimator = decision_tree_model,
                                                n_estimators = 5,
                                                verbose = 1)
tree_model2 = bagging_decision_tree_model1.fit(train_x, train_y)
predict2 = tree_model2.predict(test_x)

# Sampling More!
bagging_decision_tree_model2 = BaggingRegressor(base_estimator = decision_tree_model,
                                                n_estimators = 30,
                                                verbose = 1)
tree_model3 = bagging_decision_tree_model2.fit(train_x, train_y)
predict3 = tree_model3.predict(test_x)
print("RMSE: {}".format(sqrt(mean_squared_error(predict3, test_y))))