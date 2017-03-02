# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:13:42 2017

@author: ELF
"""

import numpy as np
import pandas as pd
import copy

from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

df_sample = pd.read_csv('sample_submission.csv')

def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

l_year=[]
l_t1=[]
l_t2=[]

for ii, row in df_sample.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    l_year.append(year)
    l_t1.append(t1)
    l_t2.append(t2)
    
df_year=pd.DataFrame({'year': l_year})
df_t1=pd.DataFrame({'team 1': l_t1})
df_t2=pd.DataFrame({'team 2': l_t2})

df_donnee=pd.DataFrame()
df_donnee=pd.concat([df_year, df_t1,df_t2], axis=1, join='inner')
df_donnee.to_csv('matchs.csv',index=False)