# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:57:34 2020

@author: Esteban

Creacion de Features para el dataset de entrenamiento de un modelo de ML
que se usara en un sistema de trading diario automatizado
pip install oandapyV20
Se necesita tener Diferenciacion_fraccional en la misma ruta
"""
import pandas as pd
import Data
from Diferenciacion_fraccional import least_diff

# Download prices from Oanda into df_pe
instrumento = "EUR_USD"
granularidad = "D"

f_inicio = pd.to_datetime("2010-01-01 17:00:00").tz_localize('GMT')
f_fin = pd.to_datetime("2020-10-19 17:00:00").tz_localize('GMT')
token = '40a4858c00646a218d055374c2950239-f520f4a80719d9749cc020ddb5188887'

df_pe= Data.getPrices(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran="D",
                            p3_inst=instrumento,p4_oatk=token, p5_ginc=4900)
df_pe = df_pe.set_index('TimeStamp')

for col in df_pe:
    for i in range(len(df_pe[col])):
        df_pe[col][i] = float(df_pe[col][i])
        

def add_fracdiff_features(df, threshold=1e-4):
    '''
    Takes every column of a DataFrame, fractionally differentiates it to the
    least required order to make it stationary and joins them to the original
    DataFrame
    
    
    Parameters
    ----------
    df : pd.DataFrame
    threshold : float(), optional
        DESCRIPTION. The default is 1e-4.
        If length of df is small, use a bigger threshold, such as 1e-3

    Returns
    -------
    df : Same df as input but with every column duplicated and fractionally
         differentiated to the least required order to make it stationary

    '''
    for col in df.columns:
        _,series = least_diff(df[col], dRange = (0,1), step=0.1, 
                              threshold=threshold, confidence='1%') #threshold menor por ser una serie pequeña
        df[col+'fdiff'] = series
    return df

df_pe = add_fracdiff_features(df_pe, threshold=1e-4)

# Labeling: 1 for positive next day return, 0 for negative next day return
def next_day_ret(df):
    '''
    Given a DataFrame with one column named 'Close' label each row according to
    the next day's return. If it is positive, label is 1. If negative, label is 0
    Designed to label a dataset used to train a ML model for trading
    
    RETURNS
    next_day_ret: pd.DataFrame
    label: list
    
    Implementation on df_pe:
        _, label = next_day_ret(df_pe)
        df_pe['Label'] = label
    '''
    next_day_ret = df.Close.pct_change().shift(-1)
    label = []
    for i in range(len(next_day_ret)):
        if next_day_ret[i]>0:
            label.append(1)
        else:
            label.append(0)
    return next_day_ret, label

_, label = next_day_ret(df_pe)
df_pe['Label'] = label
