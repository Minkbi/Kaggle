# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:37:56 2017

@author: ELF
"""

train = pd.read_csv("trainFeature.csv")
res = pd.read_csv("RegularSeasonCompactResults.csv")

def get_mean_score(i):
        
    df_Win = res[(res.Wteam == i)]  
    diffWin=df_Win.Wscore-df_Win.Lscore    
    df_Lose = res[(res.Lteam == i)]
    diffLose=df_Lose.Lscore-df_Lose.Wscore
    diff=pd.concat([diffWin,diffLose])
    return diff.mean()

df_mean=pd.DataFrame()
df_mean['Mean']=train['Seed_diff']
for ii,rows in train.iterrows():
    rows=get_mean_score(rows.Team1)-get_mean_score(rows.Team2)
    
df_mean['Mean'] = (df_mean['Mean'] - df_mean['Mean'].mean())/(df_mean['Mean'].max()-df_mean['Mean'].min())

train['Mean_diffScore']=df_mean
     