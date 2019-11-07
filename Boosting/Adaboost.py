import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("../Data/otto_train.csv")

# Preprocessing
mapping_dict = {"Class_1" : 1,
				"Class_2" : 2,
				"Class_3" : 3,
				"Class_4" : 4,
				"Class_5" : 5,
				"Class_6" : 6,
				"Class_7" : 7,
				"Class_8" : 8,
				"Class_9" : 9}

after_mapping_target = data['target'].apply(lambda x: mapping_dict[x])
feature_columns = list(data.columns.different(['target']))
X = data[feature_columns]
y = after_mapping_target

# Split Train Test set
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Basic
tree_model = DecisionTreeClassifier(max_depth = 5) 
Adaboost_model1 = AdaBoostClassifier(base_estimator = tree_model, 
                                     n_estimators = 20, 
                                     random_state = 42) 
model1 = Adaboost_model1.fit(train_x, train_y) 
predict1 = model1.predict(test_x) 
print("Accuracy: %.2f" % (accuracy_score(test_y, predict1) * 100), "%")

# More Tree
Adaboost_model2 = AdaBoostClassifier(base_estimator = tree_model,
                                    n_estimators = 300,
                                    random_state = 42)
model2 = Adaboost_model2.fit(train_x, train_y)
predict2 = model2.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict2) * 100), "%")

# More Depth
tree_model2 = DecisionTreeClassifier(max_depth = 20)
Adaboost_model3 = AdaBoostClassifier(base_estimator = tree_model2,
                                     n_estimators = 300,
                                     random_state = 42)
model3 = Adaboost_model3.fit(train_x, train_y)
predict3 = model3.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict3) * 100), "%")

# More Tree with Depth
tree_model3 = DecisionTreeClassifier(max_depth = 100) 
Adaboost_model4 = AdaBoostClassifier(base_estimator = tree_model3,
                                     n_estimators = 300,
                                     random_state = 42)

model4 = Adaboost_model4.fit(train_x, train_y)
predict4 = model4.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict4) * 100), "%")