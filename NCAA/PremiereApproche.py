    # -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:05:03 2017

@author: Marc
"""
import pandas as pd
import csv
import numpy as np



tourneyresults = pd.read_csv('TourneyCompactResults.csv')
tourneyseeds = pd.read_csv('TourneySeeds.csv')
regularseasoncompactresults = pd.read_csv('RegularSeasonCompactResults.csv')
sample = pd.read_csv('sample_Submission.csv')
#predictionGame = sample.split()



results = pd.DataFrame()
results['year'] = tourneyresults.Season
results['team1'] = np.minimum(tourneyresults.Wteam, tourneyresults.Lteam)
results['team2'] = np.maximum(tourneyresults.Wteam, tourneyresults.Lteam)
results['result'] = (tourneyresults.Wteam <
 tourneyresults.Lteam).astype(int)
merged_results = pd.merge(left=results,
  right=tourneyseeds,
  left_on=['year', 'team1'],
  right_on=['Season', 'Team'])
merged_results.drop(['Season', 'Team'], inplace=True, axis=1)
merged_results.rename(columns={'Seed': 'team1Seed'}, inplace=True)
merged_results = pd.merge(left=merged_results,
  right=tourneyseeds,
  left_on=['year', 'team2'],
  right_on=['Season', 'Team'])
merged_results.drop(['Season', 'Team'], inplace=True, axis=1)
merged_results.rename(columns={'Seed': 'team2Seed'}, inplace=True)
merged_results['team1Seed'] = \
merged_results['team1Seed'].apply(lambda x: str(x)[1:3])
merged_results['team2Seed'] = \
merged_results['team2Seed'].apply(lambda x: str(x)[1:3])
merged_results = merged_results.astype(int)
winsbyyear = regularseasoncompactresults[['Season', 'Wteam']].copy()
winsbyyear['wins'] = 1
wins = winsbyyear.groupby(['Season', 'Wteam']).sum()
wins = wins.reset_index()
lossesbyyear = regularseasoncompactresults[['Season', 'Lteam']].copy()
lossesbyyear['losses'] = 1
losses = lossesbyyear.groupby(['Season', 'Lteam']).sum()
losses = losses.reset_index()
winsteam1 = wins.copy()
winsteam1.rename(columns={'Season': 'year',
  'Wteam': 'team1',
  'wins': 'team1wins'}, inplace=True)
winsteam2 = wins.copy()
winsteam2.rename(columns={'Season': 'year',
  'Wteam': 'team2',
  'wins': 'team2wins'}, inplace=True)
lossesteam1 = losses.copy()
lossesteam1.rename(columns={'Season': 'year',
'Lteam': 'team1',
'losses': 'team1losses'}, inplace=True)
lossesteam2 = losses.copy()
lossesteam2.rename(columns={'Season': 'year',
'Lteam': 'team2',
'losses': 'team2losses'}, inplace=True)
merged_results = pd.merge(how='left',
  left=merged_results,
  right=winsteam1,
  left_on=['year', 'team1'],
  right_on=['year', 'team1'])
merged_results = pd.merge(how='left',
  left=merged_results,
  right=lossesteam1,
  left_on=['year', 'team1'],
  right_on=['year', 'team1'])
merged_results = pd.merge(how='left',
  left=merged_results,
  right=winsteam2,
  left_on=['year', 'team2'],
  right_on=['year', 'team2'])
merged_results = pd.merge(how='left',
  left=merged_results,
  right=lossesteam2,
  left_on=['year', 'team2'],
  right_on=['year', 'team2'])
teamcompactresults1 = merged_results[['year', 'team1']].copy()
teamcompactresults2 = merged_results[['year', 'team2']].copy()