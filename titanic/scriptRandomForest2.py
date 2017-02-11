# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:46:33 2017

@author: ELF
"""
import pandas as pd
from MarcTest import * 
import matplotlib.pyplot as plt

training = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
training.head()
test.head()
trainConst = training
testConst = test
training = training.set_index('PassengerId')
test = test.set_index('PassengerId')

# the following will print out the first 5 observations

def decoupageAlea9_10(train):
    #train
    
    lentot = len(train)
    lentrain = int(lentot/10*9)
#    train = train.sample(frac=1).reset_index(drop=True)
    test = copy.copy(train[lentrain:])
    train = copy.copy(train[:lentrain])
    return train,test

def validation(predict,truth):
    lenpredict = len(predict)
    sommeT = 0
    for i in range(lenpredict):
        if predict[i]==truth[i]:
            sommeT += 1
    return sommeT/lenpredict




def clean_titanic(titanic, train):
    # fill in missing age and fare values with their medians
    titleAdd(titanic)
    naAge(titanic)
    famillyAdd(titanic) #resultat en baisse
    motherAdd(titanic) #resultat en hausse
    # make male = 0 and female = 1
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    # turn embarked into numerical classes
    naEmbarked(titanic)
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    clean_data = ['Pclass','Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked','TitleId','FDim','Mother','Child']
    
    return titanic[clean_data]




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
trainLen = len(training)
testLen = len(test)
alldata = [training ,test]
alldata = pd.concat(alldata)


dataTot= clean_titanic(alldata,True)
data = dataTot[:trainLen]
dataTest= dataTot[trainLen:]


# create Random Forest model that builds 1000 decision trees
partTest = []

data = [data, trainConst.set_index('PassengerId')['Survived']]
data = pd.concat(data,axis=1)
data,partTest = decoupageAlea9_10(data)
partTest['Ligne'] = list(range(0,len(partTest)))
partTest = partTest.set_index('Ligne')
X = data.ix[:,:-1]
y = data.ix[:, -1]

def predict(data,nest):
    forest = RandomForestClassifier(n_estimators=nest,oob_score=True)
    forest = forest.fit(X, y)
    #print("Random Forest score :",forest.score(X, y))
    #
    
    #
    #dataTest=clean_titanic(test,False)
    Z = partTest.ix[:,:-1]
    Zy = partTest.ix[:,-1]
    predForest=forest.predict(Z)
    stat = validation(predForest,Zy)
    return stat,nest

tabStat = []
tabStatX =[]
tabStatY =[]
max = 0
for i in range(1,1000):
    a, b = predict(data,10+i)
    if max < a:
        max = a
    tabStatX += [a] 
    tabStatY += [b]

    

plt.plot(tabStatY,tabStatX)

    
#submission = pd.DataFrame({
#        "PassengerId": testConst["PassengerId"],
#        "Survived": predForest
#    })
#submission.to_csv('titanicForest.csv', index=False)
#
#
##
##logreg = LogisticRegression()
##logreg.fit(X, y)
##predLog = logreg.predict(Z)
##print("Logistic regression score :", logreg.score(X, y))
##
##
##submission = pd.DataFrame({
##        "PassengerId": testConst["PassengerId"],
##        "Survived": predLog
##    })
##submission.to_csv('titanicLog.csv', index=False)
