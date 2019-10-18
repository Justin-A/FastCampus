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

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Basic
random_forest_model1 = RandomForestClassifier(n_estimators = 20,
											max_depth = 5,
											random_state = 42)
model1 = random_forest_model1.fit(train_x, train_y)
predict1 = model1.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict1) * 100), "%")

# More Tree
random_forest_model2 = RandomForestClassifier(n_estimators = 300,
											max_depth = 5,
											random_state = 42)
model12 = random_forest_model2.fit(train_x, train_y)
predict2 = model1.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict2) * 100), "%")

# More Depth
random_forest_model3 = RandomForestClassifier(n_estimators = 300,
											max_depth = 20,
											random_state = 42)
model3 = random_forest_model3.fit(train_x, train_y)
predict3 = model3.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict3) * 100), "%")

# More Tree with Depth
random_forest_model4 = RandomForestClassifier(n_estimators = 300,
											max_depth = 100,
											random_state = 42)
model4 = random_forest_model4.fit(train_x, train_y)
predict4 = model4.predict(test_x)
print("Accuracy: %.2f" % (accuracy_score(test_y, predict4) * 100), "%")

# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html