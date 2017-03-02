# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:35:45 2017

@author: ELF
"""


import numpy as np
import pandas as pd
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

results = pd.read_csv("TourneyDetailedResults.csv")
results=results.drop('Wloc',axis=1)

def get_win(i):
    res = copy.copy(results)
    df_Win=pd.DataFrame()
    df_Win = res[res.Wteam == i]
    df_Win['Win']=1
    df_Win=df_Win.drop('Wteam',axis=1)
    df_Win.rename(index=str, columns={"Lteam": "adv"},inplace=True)   
    return df_Win

def get_lose(i):
    res = copy.copy(results)
    df_Lose=pd.DataFrame()
    df_Lose= res[res.Lteam == i]
    df_Lose['Win']=0           
    df_Lose=df_Lose.drop('Lteam',axis=1)
    df_Lose.rename(index=str, columns={"Wteam": "adv"},inplace=True)
    return df_Lose
    
def get_total_w():
    team = pd.read_csv("Teams.csv")
    res=pd.DataFrame()    
    for i,row in team.iterrows() :   
        dW=get_win(row.Team_Id)
        res=pd.concat([res,dW])
    return res

def get_total_l():
    team = pd.read_csv("Teams.csv")
    res=pd.DataFrame()    
    for i,row in team.iterrows() :   
        dL=get_lose(row.Team_Id)
        res=pd.concat([res,dL])
    return res

total=pd.concat(get_total_l(),get_total_w())
total.to_csv('dataTestPredictorsTotal.csv', index=False)
        
    
    
            


