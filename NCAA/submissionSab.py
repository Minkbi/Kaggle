#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:48:03 2017

@author: Sabrina
"""


import numpy as np
import pandas as pd
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

df_for_predictions = pd.read_csv('test_Sab.csv')

#x_train = pd.DataFrame()
#
#x_train['Team_Id'] = df_for_predictions['Team_Id']
#x_train['Ratio'] = df_for_predictions['Ratio']
#y_train = df_for_predictions['Result']
#x_train, y_train = shuffle(x_train, y_train)
#    
##On récupère les dataTest
#df_sample_sub = pd.read_csv('sample_submission.csv')
#
#x_test = np.zeros(shape=(len(df_sample_sub), 2))
#
#def get_year_t1_t2(id):
#    """Return a tuple with ints `year`, `team1` and `team2`."""
#    return (int(x) for x in id.split('_'))
#
##x_test = np.zeros(shape=(len(df_sample_sub), 1))
#for ii, row in df_sample_sub.iterrows():
#    year, t1, t2 = get_year_t1_t2(row.id)
#    
#    t1_ratio = df_for_predictions[df_for_predictions.Team_Id == t1].Ratio.values[0]
#    t2_ratio = df_for_predictions[df_for_predictions.Team_Id == t2].Ratio.values[0]
#
#    diff_ratio = t1_ratio - t2_ratio
#    print(t1_ratio)
#    x_test[ii,0] = diff_ratio
#    #x_test[ii,0] = seed_diff
#
#
## A CHANGER ABSOLUMENT !!!
#x_train = x_train.fillna(0.5)
#        
#model = LogisticRegression()
#model = model.fit(x_train,y_train)
##print(model.score(x_train,y_train))
#predicted = model.predict_proba(x_test)
#clipped_preds = np.clip(predicted, 0.05, 0.95)
#df_sample_sub.pred = 1-clipped_preds
#df_sample_sub.to_csv('SubmissionSab.csv', index=False)
df_for_predictions[np.isnan(df_for_predictions)]=0
#Pour fg
X_train = df_for_predictions.Ratio.values.reshape(-1,1)
y_train = df_for_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)

# Logistic regression
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)

X = np.arange(-16, 16).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]

#On récupère les dataTest
df_sample_sub = pd.read_csv('sample_submission.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

X_test = np.zeros(shape=(n_test_games, 1))

l_k =0
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    t1_ratio = df_for_predictions[(df_for_predictions.Team_Id == t1)].Ratio.values[0]
    t2_ratio = df_for_predictions[(df_for_predictions.Team_Id == t2)].Ratio.values[0]
    diff_ratio = t1_ratio - t2_ratio
    X_test[ii, 0] = diff_ratio  
#    if (not np.isnan(t1)):
#        t1_ratio = df_for_predictions[(df_for_predictions.Team_Id == t1)].Ratio.values[0]
#        diff_ratio = t1_ratio 
#        X_test[ii, 0] = diff_ratio
#    elif (not np.isnan(t2)):
#        t2_ratio = df_for_predictions[(df_for_predictions.Team_Id == t2)].Ratio.values[0]
#        diff_ratio = -t2_ratio 
#        X_test[ii, 0] = diff_ratio
#    else:
    t1_ratio = df_for_predictions[(df_for_predictions.Team_Id == t1)].Ratio.values[0]
    t2_ratio = df_for_predictions[(df_for_predictions.Team_Id == t2)].Ratio.values[0]
    diff_ratio = t1_ratio - t2_ratio
    X_test[ii, 0] = diff_ratio
                       

   # if ((t1 in df_for_predictions['Team_Id']) & (t2 in df_for_predictions['Team_Id'])) :
#        t1_ratio = df_for_predictions[(df_for_predictions.Team_Id == t1)].Ratio.values[0]
#        t2_ratio = df_for_predictions[(df_for_predictions.Team_Id == t2)].Ratio.values[0]
#        diff_ratio = t1_ratio - t2_ratio
#        print(diff_ratio)
#        X_test[ii, 0] = diff_ratio
#    if (t1 in df_for_predictions.Team_Id ) :
#        t1_ratio = df_for_predictions[(df_for_predictions.Team_Id == t1)].Ratio.values[0]
#        diff_ratio = t1_ratio 
#        X_test[ii, 0] = diff_ratio
#    if (t2 in df_for_predictions.Team_Id ) :
#        t2_ratio = df_for_predictions[(df_for_predictions.Team_Id == t2)].Ratio.values[0]
#        diff_ratio = -t2_ratio 
#        X_test[ii, 0] = diff_ratio
          
#Prédictions
preds = clf.predict_proba(X_test)[:,1]
clipped_preds = np.clip(preds, 0.05, 0.90)
df_sample_sub.pred = clipped_preds
df_sample_sub.to_csv('subtestSab.csv', index=False)


