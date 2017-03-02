# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:37:56 2017

@author: ELF
"""

results = pd.read_csv("TourneyDetailedResults.csv")



res=copy.copy(results)

df_Win = res[(res.Wteam == 1412)]  
diffWin=df_Win.Wscore-df_Win.Lscore

df_Lose = res[(res.Lteam == 1412)]
diffLose=df_Lose.Lscore-df_Lose.Wscore

diff=pd.concat([diffWin,diffLose])
m=diff.mean()