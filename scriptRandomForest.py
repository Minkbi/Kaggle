# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from numpy import genfromtxt, savetxt
import pandas as pad
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt

def main():
    #create the training & test sets, skipping the header row with [1:]
    #dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    
    dataset = pad.read_csv("train.csv")
    Y = dataset.pop('Survived')
    dataset["Age"].fillna(dataset.Age.mean(), inplace=True)
    numeric=list(dataset.dtypes[dataset.dtypes != "object"].index)
    dataset[numeric].head()
    
    test = pad.read_csv('test.csv')
    test["Age"].fillna(test.Age.mean(), inplace=True)
    test.fillna(0, inplace=True)
    numeric_test=list(test.dtypes[test.dtypes != "object"].index)
#    

#    #create and train the random forest
    rf = RandomForestRegressor(n_estimators=100,oob_score=True, random_state=42)
    rf.fit(dataset[numeric],Y)
    print(rf.score(dataset[numeric],Y))
    
    rfc = RandomForestClassifier(n_estimators=100, oob_score=True)
    rfc.fit(dataset[numeric],Y)
    predForest = rfc.predict(test[numeric_test])
    

#    score = rfc.score(dataset[numeric],Y)
    score = rfc.oob_score_
    print("Score rfc :", rfc.score(dataset[numeric], Y))
 
    savetxt('submission.csv', rfc.predict(test[numeric_test]), delimiter=',', fmt='%f')
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predForest
    })
    submission.to_csv('titanicForestBasic.csv', index=False)


main()


