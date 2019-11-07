import numpy as np
import pandas as pd
data = pd.read_csv("../Data/credit_card.csv") # 데이터 불러오기
data = data.drop(["Time", "Amount"], axis = 1) # Time, Amount 제거
data.head(10)

# 데이터 내 NA값 여부 확인
data.isnull().any() # 만약 존재한다면 0으로 대체 혹은, 해당 열을 제외하고 진행
print("Credit Card Fraud Detection data -  rows:",data.shape[0]," columns:", data.shape[1])

data.describe() # 요약 통계량

# 종속 변수의 분포 확인
from collections import Counter
Counter(data.Class)

# EDA
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('ggplot') # Using ggplot2 style visuals 
f, ax = plt.subplots(figsize = (11, 15)) # 그래프 사이즈

ax.set_facecolor('#fafafa') # 그래프 색상값
ax.set(xlim = (-5, 5)) # X축 범위
plt.ylabel('Variables') # Y축 이름
plt.title("Overview Data Set") # 그래프 제목
ax = sns.boxplot(data = data.drop(columns = ['Class']), # V1 ~ V28 확인
                 orient = 'h', 
                 palette = 'Set2')

var = data.columns.values[:-1] # V1 ~ V28
i = 0
t0 = data.loc[data['Class'] == 0] # Class : 0 인 행만 추출
t1 = data.loc[data['Class'] == 1] # Class : 1 인 행만 추출

sns.set_style('whitegrid') # 그래프 스타일 지정
plt.figure()
fig, ax = plt.subplots(8, 4, figsize = (16, 28)) # 축 지정

for feature in var:
    i += 1
    plt.subplot(7, 4, i) # 28개의 그래프
    sns.kdeplot(t0[feature], bw = 0.5, label = "Class = 0")
    sns.kdeplot(t1[feature], bw = 0.5, label = "Class = 1")
    plt.xlabel(feature, fontsize = 12) # 라벨 속성값
    locs, labels = plt.xticks()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.show();


X = np.array(data.iloc[:, data.columns != 'Class'])
y = np.array(data.iloc[:, data.columns == 'Class'])
print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

## Ir imbalanced ratio  # majarity data/# minority data

import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, classification_report
lgbm_clf = lgbm.LGBMClassifier(n_estimators = 50, random_state = 42) # LGB Classifier
lgbm_clf.fit(X_train, y_train) # 학습 진행
y_pred = lgbm_clf.predict(X_test) # 평가 데이터셋 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, y_pred)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, y_pred)) # 전체적인 성능 평가


# 기존의 X_train, y_train, X_test, y_test의 형태 확인
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) # y_train 중 레이블 값이 1인 데이터의 개수
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) # y_train 중 레이블 값이 0 인 데이터의 개수

sm = SMOTE(random_state = 42) # SMOTE 알고리즘
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) # Over Sampling 진행

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

print("Before OverSampling, the shape of X_train: {}".format(X_train.shape)) # SMOTE 적용 이전 데이터 형태
print("Before OverSampling, the shape of y_train: {}".format(y_train.shape)) # SMOTE 적용 이전 데이터 형태
print('After OverSampling, the shape of X_train: {}'.format(X_train_res.shape)) # SMOTE 적용 결과 확인
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_res.shape)) # # SMOTE 적용 결과 확인

lgbm_clf2 = lgbm.LGBMClassifier(n_estimators = 50, random_state = 42) # LGB Classifier
lgbm_clf2.fit(X_train_res, y_train_res) # 학습 진행
y_pred2 = lgbm_clf2.predict(X_test) # 평가 데이터셋 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, y_pred2)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, y_pred2)) # 전체적인 성능 평가

# BLSM (Borderline SMOTE)
from imblearn.over_sampling import BorderlineSMOTE
sm2 = BorderlineSMOTE(random_state = 42) # BLSM 알고리즘 적용
X_train_res2, y_train_res2 = sm2.fit_sample(X_train, y_train.ravel()) # Over Sampling 적용
lgbm_clf3 = lgbm.LGBMClassifier(n_estimators = 50, random_state = 42) # LGB Classifier
lgbm_clf3.fit(X_train_res2, y_train_res2) # 학습 진행
y_pred3 = lgbm_clf3.predict(X_test) # 평가 데이터셋 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, y_pred3)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, y_pred3)) # 전체적인 성능 평가

# SVMSMOTE
from imblearn.over_sampling import SVMSMOTE
sm3 = SVMSMOTE(random_state = 42) # SVMSMOTE 알고리즘 적용
X_train_res3, y_train_res3 = sm3.fit_sample(X_train, y_train.ravel()) # Over Sampling 적용
lgbm_clf4 = lgbm.LGBMClassifier(n_estimators = 50, random_state = 42) # LGB Classifier
lgbm_clf4.fit(X_train_res3, y_train_res3) # 학습 진행
y_pred4 = lgbm_clf4.predict(X_test) # 평가 데이터셋 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, y_pred4)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, y_pred4)) # 전체적인 성능 평가

# BLSM을 이용해서 Oversampling 한 학습 데이터 셋 : X_train_res2, y_train_res2
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(C = 1e+10) 
# sklearn 의 Logistic Regression은 기본적으로 Ridge 정규화가 포함되어 있기 때문에, 
# 정규화 텀을 억제하는 C를 크게 적용한다 (C:Inverse of regularization strength)
lr_model.fit(X_train_res2, y_train_res2) # 로지스틱 회귀 모형 학습
lr_predict = lr_model.predict(X_test) # 학습 결과를 바탕으로 검증 데이터를 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, lr_predict)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, lr_predict)) # 전체적인 성능 평가

# 라쏘 로지스틱 회귀모형 학습
lasso_model = LogisticRegression(penalty = 'l1') # Penalty = l1 Regularizer, C = 1.0 (Default))
lasso_model.fit(X_train_res2, y_train_res2) # 라쏘 로지스틱 회귀 모형 학습
lasso_predict = lasso_model.predict(X_test) # 학습 결과를 바탕으로 검증 데이터를 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, lasso_predict)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, lasso_predict)) # 전체적인 성능 평가

# 릿지 로지스틱 회귀모형 학습
ridge_model = LogisticRegression(penalty = 'l2') # Default = LogisticRegression()
ridge_model.fit(X_train_res2, y_train_res2) # 릿지 로지스틱 회귀 모형 학습
ridge_predict = ridge_model.predict(X_test) # 학습 결과를 바탕으로 검증 데이터를 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, ridge_predict)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, ridge_predict)) # 전체적인 성능 평가

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators = 50, # 50번 추정
                                             max_depth = 10, # 트리 최대 깊이 10
                                             random_state = 42) # 시드값 고정
rf_model = random_forest_model.fit(X_train_res2, y_train_res2) # 학습 진행
rf_predict = rf_model.predict(X_test) # 평가 데이터 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, rf_predict)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, rf_predict)) # 전체적인 성능 평가

# CatBoost
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(n_estimators = 50, # 50번 추정
                           max_depth = 10, # 트리 최대 깊이 10
                           random_state = 42, # 시드값 고정
                           verbose = True) # 학습 진행 과정 표시
cat_model.fit(X_train_res2, y_train_res2) # 학습 진행
cat_predict = cat_model.predict(X_test) # 평가 데이터 예측
print("Confusion_Matrix: \n", confusion_matrix(y_test, cat_predict)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, cat_predict)) # 전체적인 성능 평가


import random
bagging_predict_result = [] # 빈 리스트 생성
number_of_bagging = 10 # Bagging 횟수
for _ in range(number_of_bagging):
    data_index = [data_index for data_index in range(X_train_res2.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, X_train_res2.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점 무시
    cat_model = CatBoostClassifier(n_estimators = 50, # 50번 추정
                                   max_depth = 10, # 트리 최대 깊이 10
                                   random_state = 42, # 시드값 고정
                                   verbose = False) # 학습 진행 과정 생략v
    cat_model.fit(X = pd.DataFrame(X_train_res2).iloc[random_data_index, ],
                  y = pd.DataFrame(y_train_res2).iloc[random_data_index])  # 학습 진행 시 선택된 데이터들로만 진행
    cat_predict = cat_model.predict(X_test) # 평가 데이터 예측
    bagging_predict_result.append(cat_predict) # 예측 결과를 bagging_predict_result에 저장
    print(_+1, "Model Evaluation Result:", "\n", classification_report(y_test, cat_predict)) # 전체적인 성능 평가

bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(X_test.shape[0]): # 테스트 데이터 개수만큼 반복
    temp_predict = [] # 반복문 내 임시 빈 리스트 생성
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 개수 만큼 반복
        temp_predict.append(bagging_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    if np.mean(temp_predict) >= 0.5: # 0, 1 이진분류이므로, 예측값의 평균이 0.5보다 크면 1, 아니면 0으로 예측 다수결)
        bagging_predict.append(1)
    elif np.mean(temp_predict) < 0.5: # 예측값의 평균이 0.5보다 낮으면 0으로 결과 저장
        bagging_predict.append(0)
print("Confusion_Matrix: \n", confusion_matrix(y_test, bagging_predict)) # 혼돈행렬
print('\n')
print("Model Evaluation Result: \n", classification_report(y_test, bagging_predict)) # 전체적인 성능 평가


import shap
cat_model = CatBoostClassifier(n_estimators = 50, # 50번 추정
                           max_depth = 10, # 트리 최대 깊이 10
                           random_state = 42, # 시드값 고정
                           verbose = True) # 학습 진행 과정 표시
cat_model.fit(X_train_res2, y_train_res2) # 학습 진행
explainer = shap.TreeExplainer(cat_model) # 트리 모델 Shap Value 계산 객체 지정
shap_values = explainer.shap_values(X_test) # Shap Values 계산

shap.initjs() # 자바스크립트 초기화 (그래프 초기화)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:]) # 첫 번째 검증 데이터 인스턴스에 대해 Shap Value를 적용하여 시각화
# 빨간색이 영향도가 높으며, 파란색이 영향도가 낮음

shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type = "bar") # 각 변수에 대한 Shap Values의 절대값으로 중요도 파악