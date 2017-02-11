# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:21:11 2017

@author: Marc
"""
#import

# pandas
import pandas as pd
from pandas import Series,DataFrame
#copy
import copy
#display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

def dataAcq():
    trainData = pd.read_csv("train.csv")
    testData  = pd.read_csv("test.csv")
    trainFeature = copy.copy(trainData)
    testData.head()
    trainFeature.head()

#trainData = trainData + trainData[]

#==============================================================================
# Title
#==============================================================================
def titleAdd(trainFeature):
    titleId = trainFeature["Name"].str.split(',')
    titleMlle = trainFeature["Name"].str.split(',')
    titleMme = trainFeature["Name"].str.split(',')
    titleMr = trainFeature["Name"].str.split(',')
    titleRare = trainFeature["Name"].str.split(',')
    titleElse = trainFeature["Name"].str.split(',')
    title = []
    rareTitle = ['Dona', 'Lady', 'th','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
#    print("**********************")
#    print(title)
        
    for i in range(1,len(titleId)+1) :
#        print("**********************")
#        print(i)
#        print("**********************")
#        print(title[i][1])
#        print("**********************")
#        print( title[i][1].split(' ')[1][:-1])
        titleId[i] = titleId[i][1].split(' ')[1][:-1]
#        print("**********************")
#        print(title[i])
        title += [titleId[i]]
        if titleId[i] == 'Mlle' or titleId[i] == 'Ms' :
            titleMlle[i] = 1
            titleMme[i] = 0
            titleMr[i] = 0
            titleRare[i] = 0
            titleElse[i] = 0
        elif titleId[i] == 'Mme' :
            titleMlle[i] = 0
            titleMme[i] = 1
            titleMr[i] = 0
            titleRare[i] = 0
            titleElse[i] = 0
        elif titleId[i] in rareTitle:
            titleMlle[i] = 0
            titleMme[i] = 0
            titleMr[i] = 0
            titleRare[i] = 1
            titleElse[i] = 0
        elif titleId[i] == 'Mr':
            titleMlle[i] = 0
            titleMme[i] = 0
            titleMr[i] = 1
            titleRare[i] = 0
            titleElse[i] = 0
        else :
            titleMlle[i] = 0
            titleMme[i] = 0
            titleMr[i] = 0
            titleRare[i] = 0
            titleElse[i] = 1
    trainFeature['TitleMlle'] = titleMlle
    trainFeature['TitleMme'] = titleMme
    trainFeature['TitleMr'] = titleMr
    trainFeature['TitleRare'] = titleRare
    trainFeature['TitleElse'] = titleElse
    trainFeature['Title'] = title
#==============================================================================
# Name
#==============================================================================
def surnameAdd(trainFeature):
    name =  trainFeature["Name"].str.split(',')
    for i in range(len(name)) :
        name[i] = name[i][0]
    trainFeature['SurName'] = name 

            

#==============================================================================
# Family
#==============================================================================
#Unrelevant because some of the familly might be in the test part
#testFamily = trainFeature.groupby(['SurName'],as_index=False).count()
#trainFeature['Fcount'] = trainFeature['PassengerId']
#for i in range (0,tabLen):
#    trainFeature['Fcount'][i] = testFamily['PassengerId'][testFamily['SurName'] == trainFeature['SurName'][i]]
def famillyAdd(trainFeature):
    tabLen = len(trainFeature)
    familly = []
    for i in range(1,tabLen+1):
        familly += [trainFeature['SibSp'][i]+trainFeature['Parch'][i] +1]
    trainFeature['Fsize'] = familly

    fSmall=[]
    fSingle=[]
    fBig=[]            
    for i in range(tabLen):
        if  familly[i]<=1:
            fSingle += [1] #single
            fSmall += [0]
            fBig += [0]
        elif familly[i]<=4:
            fSingle += [0]
            fSmall += [1]# small
            fBig += [0]
        else :
            fSingle += [0]
            fSmall += [0]
            fBig += [1]# big
        
    trainFeature['FSingle'] = fSingle
    trainFeature['FSmall'] = fSmall
    trainFeature['FBig'] = fBig
            
#==============================================================================
# mother and children
#==============================================================================
def motherAdd(trainFeature):
    mother = []
    tabLen = len(trainFeature)
    for i in range(1,tabLen+1):
        if trainFeature['Parch'][i]>=1 and trainFeature['Age'][i]>18 and trainFeature['Title'][i]!='Miss' and trainFeature['Sex'][i]=='female':
            mother += [1]
        else:
            mother += [0]
    trainFeature['Mother']=mother

    child = []
    for i in range(1,tabLen+1):
        if trainFeature['Parch'][i]>=1 and trainFeature['Age'][i]<18 :
            child += [1]
        else:
            child += [0]
    trainFeature['Child']=child
          
            

#==============================================================================
# Deck        
#==============================================================================
def deckAdd(trainFeature):
    tabLen = len(trainFeature)
    Deck = []
    for i in range(tabLen):
        if not(np.isnan(trainFeature['Cabin'][i])):
            Deck += [trainFeature['Cabin'][i]]
    


#    test2 = trainFeature.groupby(["Name"],as_index=False).min()
#On enlève les colonnes inutiles \\todo voir quoi faire de ticket et name
#    trainData = trainFeature.drop(['PassengerId','Ticket','Name','Embarked'], axis=1)
#
#    trainData['Fare'] = trainData['Fare'].astype(int)


#    count_nan_age_titanic = trainData["Age"].isnull().sum()

#==============================================================================
# plot survived
#==============================================================================
#testSurvived = trainData['Ticket'][trainData['Survived']==1]
            
#==============================================================================
# plot Mother            
#==============================================================================
#testMother = trainFeature[trainFeature['Mother']==1]          
#testAlison = trainFeature[trainFeature['SurName']=='Richards']
#testAge = trainFeature[trainFeature['Age'] <2]

#==============================================================================
# vFcount ~= nb SibSP + parch ?
#==============================================================================
#tabFcount = trainFeature['Fcount','SibSp','Parch']
#testcount = trainFeature[trainFeature['Fcount']==6]
#countFSib = 0
#countFBoth = 0
#countFParch = 0
#countFZero = 0
#for i in range(tabLen):
#    if trainFeature['Fcount'][i] == trainFeature['SibSp'][i]:
#        countFSib+=1
#    if trainFeature['Fcount'][i] == trainFeature['SibSp'][i]+trainFeature['Parch'][i]:
#        countFBoth+=1        
#    if trainFeature['Fcount'][i] == trainFeature['Parch'][i]:
#        countFParch+=1
#    if trainFeature['Fcount'][i] == 1   :
#        countFZero+=1
#==============================================================================
# plot Family
#==============================================================================
#tabFcount = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #30
#for i in range(len(name)):   
#    tabFcount[(trainFeature['Fcount'][i])*2+(trainFeature['Survived'][i])] += 1
#
#tabRatioF=[]
#for i in range(0,15):
#    if tabFcount[i*2+1]!=0:
#        tabRatioF += [i,tabFcount[i*2+1]/(tabFcount[i*2+1]+tabFcount[i*2]),tabFcount[i*2+1]+tabFcount[i*2]]



#==============================================================================
# plot name
#==============================================================================
#test2 = trainData.groupby(["Name"],as_index=False).min()
##On enlève les colonnes inutiles \\todo voir quoi faire de ticket et name
#trainData = trainData.drop(['PassengerId','Ticket','Name','Embarked'], axis=1)
#
#trainData['Fare'] = trainData['Fare'].astype(int)
#
#
#count_nan_age_titanic = trainData["Age"].isnull().sum()
#
###==============================================================================
## plot Fare
##==============================================================================
# plot Fare
#==============================================================================
#def fareAdd(trainFeature):
#
#    fare_not_survived = trainFeature['Fare'][trainFeature["Survived"] == 0]
#    fare_survived     = trainFeature["Fare"][trainFeature["Survived"] == 1]
#
#
#    avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
#    std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])
#
#    # plot
#    trainFeature["Fare"].plot(kind='hist', figsize=(20,10),bins=100, xlim=(0,50))
#
#    avgerage_fare.index.names = std_fare.index.names = ["Survived"]
#    avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
#    #
#==============================================================================
# plot Age
#==============================================================================

#==============================================================================
# Complete les ages avec la moyenne des categories (on peut aussi tester avec la médiane)
#==============================================================================
def naAge(trainFeature):
    meanTitle = trainFeature.groupby(["Title"],as_index=True).mean()
    age =[]

    for i in range (1,len(trainFeature)+1) :
        if np.isnan(trainFeature['Age'][i]) :
            age += [meanTitle['Age'][trainFeature['Title'][i]]]
        else :
            age += [trainFeature['Age'][i]]

    trainFeature['Age'] = age
#==============================================================================
# Complete le lieu d'embarquement
#==============================================================================
def naEmbarked(trainFeature):
#countEmbarked = trainFeature.groupby(["Embarked","Pclass"],as_index=False).count()
    for i in range (1,len(trainFeature)+1) :
            if (trainFeature['Embarked'][i]) not in ['C','Q','S'] :
                trainFeature.loc[i,'Embarked'] = 'S'
            
                            
#==============================================================================
# Complete le pont
#==============================================================================

#fareCat = []

#for i in range (len(trainFeature)) :
#    if trainFeature["Fare"][i] < 10 :
#        fareCat.append("Cat1")
#    elif trainFeature["Fare"][i] < 20 :
#        fareCat.append("Cat2")
#    elif trainFeature["Fare"][i] < 30 :
#        fareCat.append("Cat3")
#    elif trainFeature["Fare"][i] < 60 :
#        fareCat.append("Cat4")
#    else :
#        fareCat.append("Cat5")
        
#trainFeature["FareCat"] = fareCat
def deckAdd(trainFeature):
    deck = trainFeature["Cabin"].str.split(' ')
    
    for i in range (len(deck)) :
        if (str(deck[i]) != 'nan') :
            deck[i] = str(deck[i])[2]
        
        trainFeature["Deck"] = deck
            
#deckExploration = trainFeature.groupby(["Deck","Embarked","Pclass"],as_index=False).count()
#trainData["Fare"].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,100))

        