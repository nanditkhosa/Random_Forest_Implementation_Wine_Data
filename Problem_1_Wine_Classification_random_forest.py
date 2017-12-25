# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 11:31:34 2017

@author: Nandzz
"""

# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

#read from the csv file and return a Pandas DataFrame.
wine = pd.read_csv('wine.csv')

# print the column names
original_headers = list(wine.columns.values)
print(original_headers)

#print the first three rows.
print(wine[0:3])

# "Quality" is the class attribute we are predicting. 
class_column = 'quality'

#include them as features. 
feature_columns = ['fixed acidity', 'volatile acidity', 
                   'citric acid', 'residual sugar', 'chlorides',
                   'free sulfur dioxide', 'total sulfur dioxide', 
                   'density', 'pH', 'sulphates', 'alcohol']

#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class. 
wine_feature = wine[feature_columns]
wine_class = wine[class_column]

# Printing first three values from dat set
print(wine_feature[0:3])
print(list(wine_class[0:3]))

# Splitting data set into training and test data
train_feature, test_feature, train_class, test_class = \
    train_test_split(wine_feature, wine_class, stratify=wine_class, \
    train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

# Implementing Random forest Classifier on the training set to create model
num_trees = 100
n_features=3
random_forest = RandomForestClassifier(n_estimators=num_trees,criterion='entropy',max_depth=19,max_features=n_features,n_jobs=-1)
random_forest.fit(train_feature, train_class)
print("Training set accuracy: {:.2f}".format(random_forest.score(train_feature, train_class)))

# Finding accuracy on test set on the basis of model created above
prediction = random_forest.predict(test_feature)
print("Test set predictions:\n{}".format(prediction))
print("Test set accuracy: {:.2f}".format(random_forest.score(test_feature, test_class)))

#Storing traing set model evaluation results in CSV file
train_class_df = pd.DataFrame(train_class,columns=[class_column])     
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)

#Storing testing set model evaluation results in CSV file
temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('test_data.csv', index=False)

# Printing Confusion Matrix for the test set prediction by the model
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#Implementing 10 fold cross validation by using StratifiedShuffleSplit folds and printing accuracy individual as well as average
skf=StratifiedShuffleSplit(n_splits=10, test_size='default', train_size=None, random_state=None)
#skf is 10 StratifiedShuffleSplit folds
scores = cross_val_score(random_forest,wine_feature,wine_class,cv=skf, scoring='accuracy')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))