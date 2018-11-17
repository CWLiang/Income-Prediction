# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd 
import os
import warnings
from sklearn import preprocessing, svm
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_regression
import time

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def labelEncoding(data, labels):
    for col in data.columns:
        label = preprocessing.LabelEncoder()
        label.fit(data[col].values)
        data[col] = label.transform(data[col].values)
        
    return data

'''
viz_df,target, _ = preprocess(training, testing)
viz_df['target'] = target
hmap = viz_df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.9,annot=True,cmap="BrBG", square=True);

fig, a = plt.subplots(1,1,figsize=(15,4))
sns.countplot(training['Age'],hue=training['target'], ax=a)

fig, a = plt.subplots(1,1,figsize=(15,4))
sns.countplot(training['Education'],hue=training['target'], ax=a)

fig, a = plt.subplots(1,1,figsize=(15,4))
sns.countplot(training['Sex'],hue=training['target'], ax=a)
'''


features = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
features_train = features + ['target']

labels = ['workclass', 'education', 'marital-status', 'sex', 'occupation', 'education-num','relationship', 'race', 'native-country']

trainData = pd.read_csv('train.csv', header=None, names = features_train)
testData = pd.read_csv('test.csv', header=None, names = features)

target = trainData['target'].values
trainData = trainData.drop(['target'], axis=1)

trainData = labelEncoding(trainData, labels)
testData = labelEncoding(testData, labels)

# Pipeline Anova SVM

kf = KFold(n_splits=8, random_state=42)
f1_scores = []

anova_filter = SelectKBest(f_regression, k=3)
clf = svm.SVC(kernel='linear')
anova_svm = make_pipeline(anova_filter, clf)

for train_idx, test_idx in kf.split(target):
    X_train, X_test = trainData.iloc[train_idx], trainData.iloc[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    anova_svm.fit(X_train, y_train)
    y_pred = anova_svm.predict(X_test)
    
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
print("F1 平均: %f, 標準差 : %f" % (np.mean(f1_scores), np.std(f1_scores)))


'''
kf = KFold(n_splits=8, random_state=42)
f1_scores = []

for train_idx, test_idx in kf.split(target):
    X_train, X_test = trainData.iloc[train_idx], trainData.iloc[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    clf = XGBClassifier(colsample_bylevel=0.6, 
            colsample_bytree= 0.9, gamma= 5, max_delta_step= 3, max_depth= 11,
            min_child_weight= 5, n_estimators= 209, subsample= 0.9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
print("F1 平均: %f, 標準差 : %f" % (np.mean(f1_scores), np.std(f1_scores)))
'''
