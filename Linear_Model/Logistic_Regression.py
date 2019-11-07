import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Read Dataset
data = pd.read_csv("../Data/SAheart.csv", sep = ",", index_col = 0)

# Normalize
data1 = data.iloc[:, :4]
data2 = data.iloc[:, 5:9]
data3 = pd.concat([data1, data2], axis = 1)

scaler = preprocessing.MinMaxScaler() # You can use Normalizer(), StandardScaler()
scaler.fit(data3)

data4 = scaler.transform(data3)
scaled_data = pd.DataFrame(data4, columns = data3.columns, index = data3.index)

# Dummy Variable
scaled_data['famhist_true'] = data['famhist'] == 'Present'
scaled_Data['famhist_false'] = data['famhist'] == 'Absent'

# Merge dataset
preprocessed_data = pd.concat([scaled_data, data['chd']], axis = 1)

# Split train, test dataset
train, test = model_selection.train_test_split(preprocessed_data)
x_train = train.loc[:, train.columns != "chd"]
y_train = train['chd']
x_test = test.loc[:, test.columns != "chd"]
y_test = test['chd']

# Logistic Regression - Basic
lr_model = LogisticRegression(C = 1e+10)
lr_model.fit(x_train, y_train)
lr_predict = lr_model.predict(x_test)

# Evaluate lr_model
lr_cm = confusion_matrix(y_test, lr_predict)
accuracy = (lr_cm[0][0] + lr_cm[1][1]) / sum(sum(lr_cm))
precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_test, lr_predict, average = 'binary')
print("Accuracy: %.2f" % accuracy)
print("Precision: %.2f" % precision)
print("Recall: %.2f" % recall)
print("F1 Score: %.2f" % accuracy)

# Logistic Regression with Lasso
lasso_model = LogisticRegression(penalty = 'l1')
lasso_model.fit(x_train, y_train)
lasso_predict = lasso_model.predict(x_test)

# Evalute lasso_model
lasso_cm = confusion_matrix(y_test, lasso_predict)
accuracy = (lasso_cm[0][0] + lasso_cm[1][1]) / sum(sum(lasso_cm))
precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_test, lasso_predict, average = 'binary')
print("Accuracy: %.2f" % accuracy)
print("Precision: %.2f" % precision)
print("Recall: %.2f" % recall)
print("F1 Score: %.2f" % accuracy)

# Logistic Regression with Ridge
ridge_model = LogisticRegression(penalty = 'l2')
ridge_model.fit(x_train, y_train)
ridge_predict = ridge_model.predict(x_test)

# Evalute ridge_model
ridge_cm = confusion_matrix(y_test, ridge_predict)
accuracy = (ridge_cm[0][0] + ridge_cm[1][1]) / sum(sum(ridge_cm))
precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_test, ridge_predict, average = 'binary')
print("Accuracy: %.2f" % accuracy)
print("Precision: %.2f" % precision)
print("Recall: %.2f" % recall)
print("F1 Score: %.2f" % accuracy)