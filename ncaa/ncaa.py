# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:03:43 2017

@author: Alice
"""

import numpy as np
import pandas as pd
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

results = pd.read_csv("RegularSeasonCompactResults.csv")
teams = pd.read_csv("Teams.csv")


def delete(year,res):
    for i in range (year,2017):
        res = res[res.Season != i]
        

results.drop(labels=['Daynum','Wloc','Lscore','Wscore','Numot'], inplace=True, axis=1)

def victory_last_season(year):
    res = copy.copy(results)
    if year > 1985:
        res = res[res.Season == year-1]
        nbWin = res.groupby(['Wteam'],as_index=True).count()
        nbWin = nbWin.rename(columns={'Lteam':'NbVictories'})
        nbWin = nbWin.drop('Season',axis=1)
        nbLosses = res.groupby(['Lteam'],as_index=True).count()
        nbLosses = nbLosses.rename(columns={'Wteam':'NbLosses'})
        nbLosses = nbLosses.drop('Season',axis=1)
        victories = pd.concat([nbWin,nbLosses],axis=1)
        victories['PVictory'] = victories['NbVictories']/(victories['NbVictories']+victories['NbLosses'])        
        victories = victories.drop('NbVictories',axis=1)
        victories = victories.drop('NbLosses',axis=1)
        #victories = victories.drop('Wteam',axis=1)
        #victories = victories.rename(columns={'Lteam':'Team'})
    return victories
    
victories = victory_last_season(1986)
victories['Season']=1986
trainFeature = copy.copy(results)
trainFeature = trainFeature[trainFeature.Season != 1985]
for i in range(1987,2017):
    tmp = victory_last_season(i)
    tmp['Season']=i
    victories = tmp.append(victories)
victories['Wteam'] = victories.index.values
trainFeature = pd.merge(trainFeature,victories, on=['Season','Wteam'])
trainFeature = trainFeature.rename(columns={'PVictory':'T1VictoriesLY'})
victories = victories.rename(columns={'Wteam':'Lteam'})
trainFeature = pd.merge(trainFeature,victories, on=['Season','Lteam'])
trainFeature = trainFeature.rename(columns={'PVictory':'T2VictoriesLY'})
trainFeature = trainFeature.rename(columns={'Wteam':'Team1','Lteam':'Team2'})
trainFeature["T1Victory"] = 1

tmp = copy.copy(trainFeature)
tmp = tmp.rename(columns={'Team1':'Team2','Team2':'Team1','T1VictoriesLY':'T2VictoriesLY','T2VictoriesLY':'T1VictoriesLY'})
tmp["T1Victory"] = 0
trainFeature = trainFeature.append(tmp,ignore_index=True)

x_train = pd.DataFrame()
x_train['T1VictoriesLY'] = trainFeature.T1VictoriesLY.values
x_train['T2VictoriesLY'] = trainFeature.T2VictoriesLY.values
y_train = trainFeature['T1Victory']
x_train, y_train = shuffle(x_train, y_train)

# Logistic regression
#logreg = LogisticRegression()
#params = {'C': np.logspace(start=-5, stop=3, num=9)}
#clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
#clf.fit(x_train, y_train)

#X = np.arange(-1, 1).reshape(-1, 1)
#preds = clf.predict_proba(X)[:,1]

#On récupère les dataTest
df_sample_sub = pd.read_csv('sample_submission.csv')


def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

#Testest = victory_last_season(2016)['PVictory'][1103]

x_test = np.zeros(shape=(len(df_sample_sub), 2))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    T1VictoriesLY = victory_last_season(year)['PVictory'][t1]
    T2VictoriesLY = victory_last_season(year)['PVictory'][t2]
    x_test[ii, 0] = T1VictoriesLY
    x_test[ii, 1] = T2VictoriesLY
          
# Prédictions
#preds = clf.predict_proba(x_test)[:,1]
#clipped_preds = np.clip(preds, 0.05, 0.95)
#df_sample_sub.pred = clipped_preds
#df_sample_sub.to_csv('subtest1.csv', index=False)


#for i in range (len(regularCompactResults)) :
#    nbWin[regularCompactResults["Wteam"][i]] += 1
#    nbLoose[regularCompactResults["Lteam"][i]] += 1


       
       
#firstYearCompetition = [3000] * (len(teamsId))
#for i in range (len(regularCompactResults)) :
#    z = regularCompactResults["Season"][i]
#    if (z < firstYearCompetition[regularCompactResults["Wteam"][i]]) :
#        firstYearCompetition[regularCompactResults["Wteam"][i]] = z
#    if (z < firstYearCompetition[regularCompactResults["Lteam"][i]]) :
#        firstYearCompetition[regularCompactResults["Lteam"][i]] = z
    
#teamsId['FirstYear'] = firstYearCompetition