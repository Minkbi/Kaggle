# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:03:43 2017

@author: Alice
"""

import numpy as np
import pandas as pd
import copy

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

results = pd.read_csv("RegularSeasonCompactResults.csv")
resultsTourney = pd.read_csv("TourneyCompactResults.csv")


results.drop(labels=['Daynum','Wloc','Lscore','Wscore','Numot'], inplace=True, axis=1)
resultsTourney.drop(labels=['Daynum','Wloc','Lscore','Wscore','Numot'], inplace=True, axis=1)

#==============================================================================
# Traitement du poucentage de victoires cette saison sur le nombre de match joués
#==============================================================================

def victories_this_season(year):
    res = copy.copy(results)
    #if year > 1985:
    res = res[res.Season == year]
    nbWin = res.groupby(['Wteam'],as_index=False).count()
    nbWin = nbWin.rename(columns={'Lteam':'NbVictories'})
    nbWin = nbWin.drop('Season',axis=1)
    nbLosses = res.groupby(['Lteam'],as_index=False).count()
    nbLosses = nbLosses.rename(columns={'Wteam':'NbLosses'})
    nbLosses = nbLosses.drop('Season',axis=1)
    victories = pd.concat([nbWin,nbLosses],axis=1)
    victories['PVictory'] = victories['NbVictories']/(victories['NbVictories']+victories['NbLosses'])        
    victories = victories.drop('NbVictories',axis=1)
    victories = victories.drop('NbLosses',axis=1)
        #victories = victories.drop('Wteam',axis=1)
        #victories = victories.rename(columns={'Lteam':'Team'})
    return victories
    
victories = victories_this_season(1985)
victories['Season']=1985
#trainFeature = copy.copy(results)
trainFeature = copy.copy(resultsTourney)
for i in range(1985,2017):
    tmp = victories_this_season(i)
    tmp['Season']=i
    victories = tmp.append(victories)
    
victories = victories.drop('Lteam',axis=1)
trainFeature = pd.merge(trainFeature, victories, on=['Season','Wteam'])
trainFeature = trainFeature.rename(columns={'PVictory':'T1VictoriesTY'})
victories = victories.rename(columns={'Wteam':'Lteam'})
trainFeature = pd.merge(trainFeature,victories, on=['Season','Lteam'])
trainFeature = trainFeature.rename(columns={'PVictory':'T2VictoriesTY'})
trainFeature = trainFeature.rename(columns={'Wteam':'Team1','Lteam':'Team2'})
trainFeature["T1Victory"] = 1

tmp = copy.copy(trainFeature)
tmp = tmp.rename(columns={'Team1':'Team2','Team2':'Team1','T1VictoriesTY':'T2VictoriesTY','T2VictoriesTY':'T1VictoriesTY'})
tmp["T1Victory"] = 0
trainFeature = trainFeature.append(tmp,ignore_index=True)
trainFeature['VictoriesTY_diff'] = trainFeature['T1VictoriesTY'] - trainFeature['T2VictoriesTY']
trainFeature = trainFeature.drop('T1VictoriesTY',axis=1)
trainFeature = trainFeature.drop('T2VictoriesTY',axis=1)

#==============================================================================
# Traitement de la différence de seed entre les 2 équipes de la saison courante
#============================================================================== 

def seed_to_int(seed):
    """Get just the digits from the seeding. Return as int"""
    s_int = int(seed[1:3])
    return s_int

df_seeds = pd.read_csv('TourneySeeds.csv')
df_seeds['n_seed'] = df_seeds.Seed.apply(seed_to_int)
df_seeds = df_seeds.drop('Seed',axis=1)
df_seeds = df_seeds.rename(columns={'Team':'Team1','n_seed':'SeedT1'})
trainFeature = pd.merge(trainFeature,df_seeds, on=['Season','Team1'])
df_seeds = df_seeds.rename(columns={'Team1':'Team2','SeedT1':'SeedT2'})
trainFeature = pd.merge(trainFeature,df_seeds, on=['Season','Team2'])
trainFeature['Seed_diff'] = trainFeature['SeedT2'] - trainFeature['SeedT1']
trainFeature = trainFeature.drop('SeedT2',axis=1)
trainFeature = trainFeature.drop('SeedT1',axis=1)


#==============================================================================
# Traitement de la moyenne de différence de points marqué entre les 2 équipes
#============================================================================== 

res = pd.read_csv("RegularSeasonCompactResults.csv")

def get_mean_score(i):
    df_Win = res[(res.Wteam == i)]  
    diffWin=df_Win.Wscore-df_Win.Lscore    
    df_Lose = res[(res.Lteam == i)]
    diffLose=df_Lose.Lscore-df_Lose.Wscore
    diff=pd.concat([diffWin,diffLose])
    return diff.mean()

df_mean=pd.DataFrame()
df_mean['Mean']=trainFeature['Seed_diff']
for ii,rows in train.iterrows():
    rows=get_mean_score(rows.Team1)-get_mean_score(rows.Team2)
    
df_mean['Mean'] = (df_mean['Mean'] - df_mean['Mean'].mean())/(df_mean['Mean'].max()-df_mean['Mean'].min())


#==============================================================================
# Mise en forme des données d'entrainement
#==============================================================================

x_train = pd.DataFrame()
#x_train['Mean_diffScore']=df_mean

x_train['VictoriesTY_diff'] = trainFeature['VictoriesTY_diff']
x_train['Seed_diff'] = trainFeature['Seed_diff']
#x_train['Mean Score'] = trainFeature['Mean_Score']
y_train = trainFeature['T1Victory']
x_train, y_train = shuffle(x_train, y_train)
trainFeature.to_csv('trainFeature.csv', index=False)
# Logistic regression
#logreg = LogisticRegression()
#params = {'C': np.logspace(start=-5, stop=3, num=9)}
#clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
#clf.fit(x_train, y_train)

#X = np.arange(-1, 1).reshape(-1, 1)
#preds = clf.predict_proba(X)[:,:]

#On récupère les dataTest
df_sample_sub = pd.read_csv('sample_submission.csv')


def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

#Testest = victory_last_season(2016)['PVictory'][1103]

#A DECOMMENTER
x_test = np.zeros(shape=(len(df_sample_sub), 2))
#x_test = np.zeros(shape=(len(df_sample_sub), 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    VictoriesTY_t1 = victories[(victories.Lteam == t1) & (victories.Season == year)].PVictory.values[0]
    if np.isnan(VictoriesTY_t1):
        VictoriesTY_t1 = victories[(victories.Lteam == t1) & (victories.Season == year-1)].PVictory.values[0]
    VictoriesTY_t2 = victories[(victories.Lteam == t2) & (victories.Season == year)].PVictory.values[0]
    if np.isnan(VictoriesTY_t2):
        VictoriesTY_t2 = victories[(victories.Lteam == t2) & (victories.Season == year-1)].PVictory.values[0]
    victories_diff = VictoriesTY_t2 - VictoriesTY_t1
    x_test[ii, 0] = victories_diff
    Seed_t2 = df_seeds[(df_seeds.Season == year) & (df_seeds.Team2 == t2)].SeedT2.values[0]
    Seed_t1 = df_seeds[(df_seeds.Season == year) & (df_seeds.Team2 == t1)].SeedT2.values[0]
    seed_diff = Seed_t2 - Seed_t1
    x_test[ii,1] = seed_diff
#    x_test[ii,2] = get_mean_score(t1)-get_mean_score(t2)
    #x_test[ii,0] = seed_diff


# A CHANGER ABSOLUMENT !!!
x_train = x_train.fillna(0.5)
params = {'C': np.logspace(start=-5, stop=3, num=9)}
model = LogisticRegressionCV()
model = model.fit(x_train,y_train)
#print(model.score(x_train,y_train))
predicted = model.predict_proba(x_test)
clipped_preds = np.clip(predicted, 0.05, 0.95)
df_sample_sub.pred = 1-clipped_preds
df_sample_sub.to_csv('testAlice.csv', index=False)

x_norm_train=copy.copy(x_train)
x_norm_train['Seed_diff'] = (x_norm_train['Seed_diff'] - x_norm_train['Seed_diff'].mean())/(x_norm_train['Seed_diff'].max()-x_norm_train['Seed_diff'].min())



# Prédictions
#preds = clf.predict_proba(x_test)[:,:]
#clipped_preds = np.clip(preds, 0.05, 0.95)
#df_sample_sub.pred = clipped_preds
#df_sample_sub.to_csv('subtest2.csv', index=False)


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