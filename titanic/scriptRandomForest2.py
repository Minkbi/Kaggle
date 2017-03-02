# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:46:33 2017

@author: ELF
"""
import pandas as pd
from MarcTest import * 
import matplotlib.pyplot as plt


def main() :
    training = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    training.head()
    test.head()
    trainConst = training
    testConst = test
    training = training.set_index('PassengerId')
    test = test.set_index('PassengerId')
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    trainLen = len(training)
    testLen = len(test)
    alldata = [training ,test]
    alldata = pd.concat(alldata)
    
    #alldata['Cabin'].fillna(0, inplace=True)
    #cabin =[]
    #for i in range(1,len(alldata)+1):
    #    if alldata["Cabin"][i] == 0:
    #        cabin += [0]
    #    elif alldata["Cabin"][i][0] == 'A' :
    #        cabin += [1]
    #    elif alldata["Cabin"][i][0] == 'B' :
    #        cabin += [2]
    #    elif alldata["Cabin"][i][0] == 'C' :
    #        cabin += [3]
    #    elif alldata["Cabin"][i][0] == 'D' :
    #        cabin += [4]
    #    elif alldata["Cabin"][i][0] == 'E' :
    #        cabin += [5]
    #    elif alldata["Cabin"][i][0] == 'F' :
    #        cabin += [6]
    #    elif alldata["Cabin"][i][0] == 'G' :
    #        cabin += [7]
    #
    #absX = alldata['Pclass'][:-1]
    #absY = cabin
    #plt.plot(absX,absY )
    
    dataTot= clean_titanic(alldata)
    data = dataTot[:trainLen]
    dataTest= dataTot[trainLen:]
    
    
    # create Random Forest model that builds 1000 decision trees
    partTest = []
    
    data = [data, trainConst.set_index('PassengerId')['Survived']]
    data = pd.concat(data,axis=1)
    
    
    def predictTest(data,nest):
        partTest = []
        data,partTest = decoupageAlea9_10(data)
        partTest['Ligne'] = list(range(0,len(partTest)))
        partTest = partTest.set_index('Ligne')
        X = data.ix[:,:-1]
        y = data.ix[:, -1]
        forest = RandomForestClassifier(n_estimators=nest,oob_score=True)
        forest = forest.fit(X, y)
    #    print("Random Forest score :",forest.score(X, y))
    #    print(partTest)
        Z = partTest.ix[:,:-1]
        Zy = partTest.ix[:,-1]
        predForest=forest.predict(Z)
        stat = validation(predForest,Zy)
    #    stat = 0
        return stat,nest
    
    def predictFinal(data,test):
        X = data.ix[:,:-1]
        y = data.ix[:, -1]
        forest = RandomForestClassifier(n_estimators=300,oob_score=True)
        forest = forest.fit(X, y)
    #    print("Random Forest score :",forest.score(X, y))
    
        predForest=forest.predict(test)
        return predForest
    
    
    tabStatX =[]
    tabStatY =[]
    max = 0
    #for i in range(1,10):
    a, b = predictTest(data,100)
    predForest = predictFinal(data,dataTest)
    #    if max < a:
    #        max = a
    #    tabStatX += [a] 
    #    tabStatY += [b]
    
    #print(a)
    #print(max(tabStatX))
    #plt.plot(tabStatY,tabStatX)
    
        
    submission = pd.DataFrame({
            "PassengerId": testConst["PassengerId"],
            "Survived": predForest
        })
    submission.to_csv('titanicForest.csv', index=False)
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
    

#,BTitle,BFam,BMot,BChi

def clean_titanic(titanic):
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
    titanic.loc[titanic["Embarked"] == 'S', "EmbarkedS"] = 1
    titanic.loc[titanic["Embarked"] != 'S', "EmbarkedS"] = 0
    titanic.loc[titanic["Embarked"] == "C", "EmbarkedC"] = 1
    titanic.loc[titanic["Embarked"] != "C", "EmbarkedC"] = 0
    titanic.loc[titanic["Embarked"] == "Q", "EmbarkedQ"] = 1
    titanic.loc[titanic["Embarked"] != "Q", "EmbarkedQ"] = 0

    clean_data = ['TitleMlle','TitleMme','TitleMr','TitleRare','TitleElse','FSingle','FSmall','FBig','Pclass','Age', 'Sex', 'SibSp', 'Parch', 'Fare',
 'EmbarkedS', 'EmbarkedC', 'EmbarkedQ',
'TitleMlle','TitleMme','TitleMr','TitleRare','TitleElse',
'FSingle','FSmall','FBig'
,'Mother','Child']

    titanic = deckAdd(titanic)
    clean_data = ['Cabin_0','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Pclass','Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'EmbarkedS', 'EmbarkedC', 'EmbarkedQ','TitleMlle','TitleMme','TitleMr','TitleRare','TitleElse','FSingle','FSmall','FBig']
#    ['Cabin_0','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G',
#'Pclass','Age', 'Sex', 'SibSp', 'Parch', 'Fare',
# 'EmbarkedS', 'EmbarkedC', 'EmbarkedQ',
#'TitleMlle','TitleMme','TitleMr','TitleRare','TitleElse',
#'FSingle','FSmall','FBig'
#,'Mother','Child']
    return titanic[clean_data]



from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
trainLen = len(training)
testLen = len(test)
alldata = [training ,test]
alldata = pd.concat(alldata)

#alldata['Cabin'].fillna(0, inplace=True)
#cabin =[]
#for i in range(1,len(alldata)+1):
#    if alldata["Cabin"][i] == 0:
#        cabin += [0]
#    elif alldata["Cabin"][i][0] == 'A' :
#        cabin += [1]
#    elif alldata["Cabin"][i][0] == 'B' :
#        cabin += [2]
#    elif alldata["Cabin"][i][0] == 'C' :
#        cabin += [3]
#    elif alldata["Cabin"][i][0] == 'D' :
#        cabin += [4]
#    elif alldata["Cabin"][i][0] == 'E' :
#        cabin += [5]
#    elif alldata["Cabin"][i][0] == 'F' :
#        cabin += [6]
#    elif alldata["Cabin"][i][0] == 'G' :
#        cabin += [7]
#
#absX = alldata['Pclass'][:-1]
#absY = cabin
#plt.plot(absX,absY )

dataTot= clean_titanic(alldata)
n_features=dim(dataTot)
data = dataTot[:trainLen]
dataTest= dataTot[trainLen:]


# create Random Forest model that builds 1000 decision trees
partTest = []

data = [data, trainConst.set_index('PassengerId')['Survived']]
data = pd.concat(data,axis=1)


def predictTest(data,nest):
    partTest = []
    data,partTest = decoupageAlea9_10(data)
    partTest['Ligne'] = list(range(0,len(partTest)))
    partTest = partTest.set_index('Ligne')
    X = data.ix[:,:-1]
    y = data.ix[:, -1]
    forest = RandomForestClassifier(n_estimators=100,oob_score=True, n_jobs = 2, max_depth=20, max_features=n_features)
    forest = forest.fit(X, y)
    
    
    
#    importance = forest.feature_importances_
#    importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
#    importance["Std"] = np.std([tree.feature_importances_
#                            for tree in forest.estimators_], axis=0)   
#    x = range(importance.shape[0])
#    y = importance.ix[:, 0]
#    yerr = importance.ix[:, 1]
#    plt.bar(x, y, yerr=yerr, align="center")
#    #plt.show()


#    print("Random Forest score :",forest.score(X, y))
#    print(partTest)
    Z = partTest.ix[:,:-1]
    Zy = partTest.ix[:,-1]
    predForest=forest.predict(Z)
    stat = validation(predForest,Zy)
#    stat = 0
    return stat,nest

def predictFinal(data,test):
    X = data.ix[:,:-1]
    y = data.ix[:, -1]
    forest = RandomForestClassifier(n_estimators=300,oob_score=True,n_jobs= -1)
    forest = forest.fit(X, y)
#    print("Random Forest score :",forest.score(X, y))

    predForest=forest.predict(test)
    return predForest


tabStatX =[]
tabStatY =[]
max = 0
#for i in range(1,20):
a, b = predictTest(data,i)
print("On obtiens avec ", i, a)

predForest = predictFinal(data,dataTest)
#    if max < a:
#        max = a
#    tabStatX += [a] 
#    tabStatY += [b]


#print(max(tabStatX))
#plt.plot(tabStatY,tabStatX)

    
submission = pd.DataFrame({
        "PassengerId": testConst["PassengerId"],
        "Survived": predForest
    })
submission.to_csv('titanicForest.csv', index=False)
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
