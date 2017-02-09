# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:46:33 2017

@author: ELF
"""
import pandas as pd
from MarcTest import * 

training = pd.read_csv("train.csv")
# the following will print out the first 5 observations

def clean_titanic(titanic, train):
    # fill in missing age and fare values with their medians
    nameAdd(titanic)
    naAge(titanic)
#    motherAdd(titanic)
    # make male = 0 and female = 1
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    # turn embarked into numerical classes
    naEmbarked(titanic)
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    if train == True:
        clean_data = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked','Title']
    else:
        clean_data = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked','Title']
    return titanic[clean_data]

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
data= clean_titanic(training,True)




# create Random Forest model that builds 1000 decision trees
forest = RandomForestClassifier(n_estimators=1000,oob_score=True)
X = data.ix[:, 1:]
y = data.ix[:, 0]
forest = forest.fit(X, y)
print("Random Forest score :",forest.score(X, y))


test = pd.read_csv("test.csv")
dataTest=clean_titanic(test,False)
Z = dataTest.ix[:, 0:]
print(test['Age'][88])
#for i in range(len(Z)):
#    if (not(test['Age'][j]>0)):
#        print(i)
#        print ('*****************************************************')
predForest=forest.predict(Z)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predForest
    })
submission.to_csv('titanicForest.csv', index=False)


#
#logreg = LogisticRegression()
#logreg.fit(X, y)
#predLog = logreg.predict(Z)
#print("Logistic regression score :", logreg.score(X, y))
#
#
#submission = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": predLog
#    })
#submission.to_csv('titanicLog.csv', index=False)
