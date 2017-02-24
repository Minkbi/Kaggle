# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:31:45 2016

@author: christopherhamblin

TITANIC SURVIVOR ANALYSIS
This is a demonstative analysis of the Kaggle "Titanic Survivor" competition,
for which we will try to label passanger as survivors or not in the test data set
based on information provided in the labeled training data set. The two data sets can be found here;
 https://www.kaggle.com/c/titanic/data

"""


#import libraries
import pandas as pd
import numpy as np




#import data frames
dftrain=pd.read_csv('train.csv')
dftest=pd.read_csv('test.csv')

#Combine Training and Test data for easy manipulation
    #move Survived column to front of trian
cols = dftrain.columns.tolist()
cols=[cols[1]]+[cols[0]]+cols[2:]
dftrain=dftrain[cols]
    #add Survived column to front of test
dftest.insert(0,'Survived',np.nan,True)
    #combine
dfcombi=pd.concat([dftrain,dftest]) 



#Manipulate data into format useful to classifier
    #Get rid of unuseable variables
dfcombi=dfcombi.drop(['Ticket'],axis=1)

    #fill in missing values
        #Age (use average age)
age_mean=dfcombi['Age'].mean()
dfcombi['Age']=dfcombi['Age'].fillna(age_mean)
        #Embarked (use most common port)
from scipy.stats import mode
mode_embarked = mode(dfcombi['Embarked'].dropna())[0][0]
dfcombi['Embarked'] = dfcombi['Embarked'].fillna(mode_embarked)

        #Fare (use average for given Pclass)
meanfare=dfcombi.pivot_table('Fare',index='Pclass',aggfunc='mean')
dfcombi['Fare'] = dfcombi[['Fare', 'Pclass']].apply(lambda x:
                            meanfare[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'],axis=1)
       
        #Cabin (Use 0, we will convert cabin to numeric later, unlisted cabins
        # have lowest survival rate, so we'll give them there own number)      
dfcombi['Cabin']=dfcombi['Cabin'].fillna('0')
dfcombi['Cabin']=dfcombi['Cabin'].apply(lambda x: x[0])

      
    #Create numerical values for catagorical variables 
       #Sex (0 or1)
dfcombi['Sex']=dfcombi['Sex'].map({'female':1,'male':0}).astype(int)

       #Embarked (One hot encode, leaving out the mode 'S')
dfcombi = pd.concat([dfcombi, pd.get_dummies(dfcombi['Embarked'], \
 prefix='Embarked')], axis=1)
dfcombi=dfcombi.drop(['Embarked_S','Embarked'],axis=1)

       #Cabin (One hot encode, leaving out the mode '0')
dfcombi = pd.concat([dfcombi, pd.get_dummies(dfcombi['Cabin'], \
 prefix='Cabin')], axis=1)
dfcombi=dfcombi.drop(['Cabin_0','Cabin'],axis=1)
           
       #Name (Group names by title then one hot encode dropping 'Mr')        
import re
dfcombi['Title']=dfcombi['Name'].apply(lambda x: re.split('[,.]',x)[1])
dfcombi['Title']=dfcombi['Title'].apply(lambda x: re.sub(' ','',x,1))
             #combine unusual names
dfcombi['Title']=dfcombi['Title'].apply(lambda x: 'Miss' if x=='Ms' else x)
dfcombi['Title']=dfcombi['Title'].apply(lambda x: 'Lady' if x in ('Mlle','Mme','the Countess','Dona')
                                            else x)
dfcombi['Title']=dfcombi['Title'].apply(lambda x: 'Sir' if x in ('Col','Major','Capt','Jonkheer','Don')
                                            else x)
dfcombi = pd.concat([dfcombi, pd.get_dummies(dfcombi['Title'], \
 prefix='Title')], axis=1)
dfcombi=dfcombi.drop(['Title_Mr','Title','Name'],axis=1)
 
#Split combined data frame back into training and testing 
dftrain=dfcombi.head(891)
dftest=dfcombi.tail(418)
dftest=dftest.drop(['Survived'],axis=1)

#Scale data for non-tree classifiers
dfcombinorm=dfcombi
dfcombinorm.loc[:,['Pclass','Age','SibSp','Parch','Fare']]= \
(dfcombinorm - dfcombinorm.mean()) / (dfcombinorm.max() - dfcombinorm.min())
dftrainnorm=dfcombinorm.head(891)
dftestnorm=dfcombi.tail(418)
dftestnorm=dftestnorm.drop(['Survived'],axis=1)

#Train Classifiers on Training Data
    #Convert data frames to numpy arrays
train=dftrain.values
test=dftest.values
trainnorm=dftrainnorm.values
testnorm=dftestnorm.values


import random
random.seed(400)

    #SVM
        #Train on normalized training data
from sklearn import svm
print ('')
print ('Training SVM...')
svmmodel=svm.SVC(degree=3, gamma='auto', kernel='linear')
svmmodel.fit(trainnorm[:,2:],trainnorm[:,0].astype(int))
        #predict on normalized test data
print ('')
print ('Predicting using svm...')
svmoutput = svmmodel.predict(testnorm[:,1:])
svmoutput_train=svmmodel.predict(trainnorm[:,2:]) #for accuracy score

        #Accuracy Score
from sklearn import metrics
print ('')
svm_pscore= metrics.accuracy_score(trainnorm[:,0].astype(int), svmoutput_train)

#svm_pscore=svmmodel.score(trainnorm[:,2:],trainnorm[:,0].astype(int))
print ('Accuracy score: ', svm_pscore) 

         #Write results to csv
svmresult = np.c_[testnorm[:,0].astype(int), svmoutput.astype(int)]
dfsvmresult = pd.DataFrame(svmresult[:,0:2], columns=['PassengerId', 'Survived'])
dfsvmresult.to_csv('svm.csv', index=False) #Change pathname