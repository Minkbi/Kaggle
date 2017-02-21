# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:03:43 2017

@author: Alice
"""

import pandas as pd
import copy

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
    return victories
    
victories = victory_last_season(2010)


#teamsId = teamsId.set_index('Team_Id')

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