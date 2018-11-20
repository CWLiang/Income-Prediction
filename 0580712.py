import numpy as np 
import pandas as pd 
import warnings
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_regression
import time
from sklearn.utils import shuffle

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def labelEncoding(data, labels):
    for col in data.columns:
        label = preprocessing.LabelEncoder()
        label.fit(data[col].values)
        data[col] = label.transform(data[col].values)
        
    return data

def scaler(data):
    
    for col in data.columns:
        m = min(data[col].values)
        M = max(data[col].values)
        scale = M-m
        data[col] = data[col].values/scale
        
    return data

def oversampling(data):
    
    idx = data['target'] == 1
    idy = data['target'] == 0
    
    tmp1 = data[idx]
    tmp0 = data[idy]
    data = data.append([tmp1]*2,ignore_index=True)
    #data = data.append([tmp0]*0, ignore_index=True)

    
    return shuffle(data)


features = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
features_train = features + ['target']

labels = ['workclass', 'education', 'marital-status', 'sex', 'occupation', 'education-num','relationship', 'race', 'native-country']

trainData = pd.read_csv('train.csv', header=None, names = features_train)
testData = pd.read_csv('test.csv', header=None, names = features)

#trainData = oversampling(trainData)

trainData = shuffle(trainData)

target = trainData['target'].values

a0 = 0
a1 = 0

for t in target:
    if t == 0:
        a0 = a0+1
    else:
        a1 = a1+1
        
print("0:", a0, "a1:", a1)

trainData = trainData.drop(['target'], axis=1)

trainData = labelEncoding(trainData, labels)
testData = labelEncoding(testData, labels)

trainData = scaler(trainData)
testData = scaler(testData)

kf = KFold(n_splits=8, random_state=42)
tStart = time.time()
myfilter = SelectKBest(f_regression, k=13)
#clf = XGBClassifier()
clf = XGBClassifier(colsample_bylevel=0.6, 
            colsample_bytree= 0.9, gamma= 5, max_delta_step= 3, max_depth= 10,
            min_child_weight= 1, n_estimators= 209, subsample= 0.9)
myModel = make_pipeline(myfilter, clf)
f1_scores = []
for train_idx, test_idx in kf.split(target):
    X_train, X_test = trainData.iloc[train_idx], trainData.iloc[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    myModel.fit(X_train, y_train)
    y_pred = myModel.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
print("Mean : %f, sigma : %f" % (np.mean(f1_scores), np.std(f1_scores)))
tEnd = time.time()
print("Model training time:", tEnd - tStart, "seconds")

y_pred = myModel.predict(testData)
answer = pd.read_csv('sub.csv')
answer['ans'] = y_pred
answer.to_csv('answer.csv', index=False)


