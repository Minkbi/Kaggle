# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:21:11 2017

@author: Marc
"""
#import
# pandas
import pandas as pd
from pandas import Series,DataFrame
#display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

trainData = pd.read_csv("train.csv")
testData  = pd.read_csv("test.csv")

trainData.head()

#trainData = trainData + trainData[]

test = trainData["Name"].str.split(',')
test_split = test

for i in range(len(test)) :
    test_split[i] = test[i][1].split(' ')[1][:-1]




test2 = trainData.groupby(["Name"],as_index=False).min()
#On enlève les colonnes inutiles \\todo voir quoi faire de ticket et name
trainData = trainData.drop(['PassengerId','Ticket','Name','Embarked'], axis=1)

trainData['Fare'] = trainData['Fare'].astype(int)


count_nan_age_titanic = trainData["Age"].isnull().sum()

##==============================================================================
# plot Fare
#==============================================================================
fare_not_survived = trainData['Fare'][trainData["Survived"] == 0]
fare_survived     = trainData["Fare"][trainData["Survived"] == 1]


avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
trainData["Fare"].plot(kind='hist', figsize=(20,10),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
#
#==============================================================================
# plot Age
#==============================================================================
