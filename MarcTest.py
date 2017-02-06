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

trainData = pd.read_csv("train.csv")
testData  = pd.read_csv("test.csv")
trainFeature = copy.copy(trainData)
trainData.head()
trainFeature.head()

#trainData = trainData + trainData[]

#==============================================================================
# Title
#==============================================================================

title = trainData["Name"].str.split(',')
rareTitle = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
for i in range(len(title)) :
    title[i] = title[i][1].split(' ')[1][:-1]
    if title[i] == 'Mlle' or title[i] == 'Ms' :
        title[i] = 'Miss'
    elif title[i] == 'Mme' :
        title[i] = 'Mrs'
    elif title[i] in rareTitle:
        title[i] = 'Rare'
trainFeature['Title'] = title
#==============================================================================
# Name
#==============================================================================
         
name =  trainData["Name"].str.split(',')
for i in range(len(name)) :
    name[i] = name[i][0]
trainFeature['SurName'] = name 
    
            

#==============================================================================
# Family
#==============================================================================
testFamily = trainFeature.groupby(['SurName'],as_index=False).count()
trainFeature['Fcount'] = testFamily['PassengerId']
trainFeature['Fcount' == 1]



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
#fare_not_survived = trainData['Fare'][trainData["Survived"] == 0]
#fare_survived     = trainData["Fare"][trainData["Survived"] == 1]
#
#
#avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
#std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])
#
## plot
#trainData["Fare"].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))
#
#avgerage_fare.index.names = std_fare.index.names = ["Survived"]
#avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

#==============================================================================
# plot Age
#==============================================================================
