import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("../Data/kc_house_data.csv") 

'''
id: 집 고유아이디
date: 집이 팔린 날짜 
price: 집 가격 (타겟변수)
bedrooms: 주택 당 침실 개수
bathrooms: 주택 당 화장실 개수
floors: 전체 층 개수
waterfront: 해변이 보이는지 (0, 1)
condition: 집 청소상태 (1~5)
grade: King County grading system 으로 인한 평점 (1~13)
yr_built: 집이 지어진 년도
yr_renovated: 집이 리모델링 된 년도
zipcode: 우편번호
lat: 위도
long: 경도
'''

# Data Shape
nCar = data.shape[0] 
nVar = data.shape[1] 
data = data.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis = 1) # Remove id, date, zipcode, lat, long

feature_columns = list(data.columns.difference(['price']))
X = data[feature_columns]
y = data['price']

# Split Dataset
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) 

# Baseline with LGB
import lightgbm as lgb
from math import sqrt
from sklearn.metrics import mean_squared_error
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) 
lgb_param = {'max_depth': 10,
            'learning_rate': 0.01,
            'n_estimators': 1000, 
            'objective': 'regression'} 
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain)
lgb_model_predict = lgb_model.predict(test_x)
print("RMSE: {}".format(sqrt(mean_squared_error(lgb_model_predict, test_y))))

# !pip install shap
# import skimage -> skimage.__version__ (skimage version)
# skimage version upgrade -> !pip install --upgrade scikit-image
import shap
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(test_x)

# Sample
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], test_x.iloc[0,:]) 
# Higher effect with Red Color, Lower effect with Blue Color

shap.force_plot(explainer.expected_value, shap_values, test_x)
shap.summary_plot(shap_values, test_x)


shap.summary_plot(shap_values, test_x, plot_type = "bar")