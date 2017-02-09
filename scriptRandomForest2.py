# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:46:33 2017

@author: ELF
"""
import pandas as pd
training = pd.read_csv("train.csv")
# the following will print out the first 5 observations

def clean_titanic(titanic, train):
    # fill in missing age and fare values with their medians
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    # make male = 0 and female = 1
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    # turn embarked into numerical classes
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    if train == True:
        clean_data = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    else:
        clean_data = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
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
predForest=forest.predict(Z)

submission = pd.DataFrame({
<<<<<<< HEAD
        "PassengerId": test_df["PassengerId"],
=======
        "PassengerId": test["PassengerId"],
>>>>>>> da21083e96a4ac5bad2ee2db63096bdee03bf0c9
        "Survived": predForest
    })
submission.to_csv('titanicForest.csv', index=False)


logreg = LogisticRegression()
logreg.fit(X, y)
predLog = logreg.predict(Z)
print("Logistic regression score :", logreg.score(X, y))


submission = pd.DataFrame({
<<<<<<< HEAD
        "PassengerId": test_df["PassengerId"],
=======
        "PassengerId": test["PassengerId"],
>>>>>>> da21083e96a4ac5bad2ee2db63096bdee03bf0c9
        "Survived": predLog
    })
submission.to_csv('titanicLog.csv', index=False)
