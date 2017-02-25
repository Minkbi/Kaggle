# -*- coding: utf-8 -*-

"""
Created on Thu Feb 16 16:53:03 2017

@author: Marc
"""
#copy
import copy
#numpy
import numpy as np
#panda
import pandas as pd

# fonction de découpage de la donnée (return partie train, partie test)
def decoupageDataTest(allData):
    
    lentot = len(allData)
    lentrain = int(lentot/10*9)
#    train = train.sample(frac=1).reset_index(drop=True)
    test = copy.copy(allData[lentrain:])
    train = copy.copy(allData[:lentrain])    
    
    return train, test


#mélange les Lignes de la donnée pour éviter les tris initiaux
#mets les index de 0 à len(arg)-1
def shuffleRows(allData):
    allData = allData.reindex(np.random.permutation(allData.index))
    allData = indiceA0(allData)
    return allData


#donne la statistique de réussite 
def evaluation(predict,truth):
    predict = indiceA0(predict)
    truth = indiceA0(truth)
    lenpredict = len(predict)
    sommeT = 0
    for i in range(lenpredict):
        if predict[i]==truth[i]:
            sommeT += 1
    return sommeT/lenpredict

#fait commencer les indices de la dataframe à 0
def indiceA0(allData):
    ligne = list(range(0,len(allData)))
    allData["LigneIndex"] = ligne
    allData = allData.set_index('LigneIndex')
    return allData