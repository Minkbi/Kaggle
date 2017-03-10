#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:02:04 2017

@author: Sabrina
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

df_teams = pd.read_csv('TourneyDetailedResults.csv')

df_teams_base = pd.read_csv('Teams.csv')

df_team_Wratio = pd.read_csv('Teams.csv')

df_result = pd.read_csv('TourneyDetailedResults.csv')

df_teams.drop(labels=['Wscore','Lscore','Lteam','Wfgm','Wfga','Wfgm3','Wfga3','Wftm','Wfta','Lfgm','Lfga','Lfgm3','Lfga3','Lftm','Lfta','Daynum', 'Wloc', 'Numot', 'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf'], inplace=True, axis=1)
df_teams = df_teams.rename(columns={'Season':'NbGamesPlay','Wteam':'Team_Id'})

df_teams_base = df_teams_base.rename(columns={'Wteam':'Team_Id'})
df_teams_base.drop(labels=['Team_Name'], inplace=True, axis=1)
#df_teams_base['Result'] = 0
df_teams_base['NbGamesPlay'] = 0
#df_teams_base['Ratio'] = 0.0

df_teams=df_teams.groupby(['Team_Id'],as_index=False).count()
df_result.drop(labels=['Daynum', 'Wloc', 'Numot', 'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf'], inplace=True, axis=1)



df_result['Wratio_fg']= (df_result['Wfga']-df_result['Wfgm'])/df_result['Wfga']
        
df_result['Wratio_fg3']= (df_result['Wfga3']-df_result['Wfgm3'])/df_result['Wfga3']
         
df_result['Wratio_ft']= (df_result['Wfta']-df_result['Wftm'])/df_result['Wfta']

df_result['Lratio_fg']= (df_result['Lfga']-df_result['Lfgm'])/df_result['Lfga']
         
df_result['Lratio_fg3']= (df_result['Lfga3']-df_result['Lfgm3'])/df_result['Lfga3']
         
df_result['Lratio_ft']= (df_result['Lfta']-df_result['Lftm'])/df_result['Lfta']

df_team_Wratio.drop(labels=['Team_Name'], inplace=True, axis=1)


df_team_Wratio = pd.merge(left=df_team_Wratio, right=df_teams_base, on=['Team_Id'])
#Ajout de colonnes

df_team_Wratio['fg']=df_result['Wratio_fg']-df_result['Wratio_fg']
df_team_Wratio['fg3']=df_team_Wratio['fg']
df_team_Wratio['ft']=df_team_Wratio['fg']

df_team_Lratio = df_team_Wratio

df_team_Wratio=df_team_Wratio.sort_values( ["Team_Id"])

l_k =0
l_i =0
j = 0

for k in df_team_Wratio['Team_Id'] :
    for i in df_result['Wteam'] :    
        if(i==k):
            j+=1
            df_team_Wratio['fg'][l_k] += df_result['Wratio_fg'][l_i]
            df_team_Wratio['fg3'][l_k] += df_result['Wratio_fg3'][l_i]
            df_team_Wratio['ft'][l_k] += df_result['Wratio_ft'][l_i]
        l_i +=1
    df_team_Wratio['fg'][l_k] = df_team_Wratio['fg'][l_k]/ j#df_team_Wratio['NbGamesPlay'][l_k]
    df_team_Wratio['fg3'][l_k] = df_team_Wratio['fg3'][l_k]/j
    df_team_Wratio['ft'][l_k] = df_team_Wratio['ft'][l_k]/j
    df_team_Wratio['NbGamesPlay'][l_k]= j
    j=0
    l_i=0
    l_k+=1
    
l_k =0
l_i =0
j = 0    
    
for k in df_team_Lratio['Team_Id'] :
    for i in df_result['Lteam'] :
        if(i==k):
            j+=1
            df_team_Lratio['fg'][l_k] += df_result['Lratio_fg'][l_i]
            df_team_Lratio['fg3'][l_k] += df_result['Lratio_fg3'][l_i]
            df_team_Lratio['ft'][l_k] += df_result['Lratio_ft'][l_i]
        l_i +=1
    df_team_Lratio['fg'][l_k] = df_team_Lratio['fg'][l_k]/j
    df_team_Lratio['fg3'][l_k] = df_team_Lratio['fg3'][l_k]/j
    df_team_Lratio['ft'][l_k] = df_team_Lratio['ft'][l_k]/j
    df_team_Lratio['NbGamesPlay'][l_k] = j
    j=0
    l_i=0
    l_k+=1
    
df_team_ratio = df_team_Wratio

#df_team_ratio.drop(labels=['NbGamesPlay'], inplace=True, axis=1)

df_team_ratio['fg'] = (df_team_ratio['fg3']+df_team_Lratio['fg3'])/2 #(df_team_ratio['fg']+df_team_Lratio['fg'])/2            
df_team_ratio['fg3'] = (df_team_ratio['fg3']+df_team_Lratio['fg3'])/2
df_team_ratio['ft'] = (df_team_ratio['ft']+df_team_Lratio['ft'])/2
             
#df_team_ratio['fg'] = (df_team_ratio['fg']+df_team_ratio['fg3']+df_team_ratio['ft'])/3


df_team_Wratio.drop(labels=['fg3','ft'], inplace=True, axis=1)
df_team_ratio = df_team_ratio.rename(columns={'fg':'Ratio'})
     
df_team_ratio['NbGamesPlay'] =  df_team_Wratio['NbGamesPlay'] -df_team_Lratio['NbGamesPlay']
#df_team_ratio['Result'] = df_team_ratio['NbGamesPlay']
df_team_ratio = df_team_ratio.rename(columns={'NbGamesPlay':'Result'})
   
#df_team_ratio.drop(labels=['NbGamesplay'], inplace=True, axis=1)
l_k =0

for k in df_team_ratio['Result'] :
        if(k<0):
            df_team_ratio['Result'][l_k] = 0
        else :
            df_team_ratio['Result'][l_k] = 1
        l_k+=1

#df_final = pd.concat((df_team_ratio, df_teams_base))
#df_final = df_team_ratio.append(df_teams_base)
#df_final.to_csv('test_Sab.csv', index=False)
df_team_ratio.to_csv('test_Sab.csv', index=False)




