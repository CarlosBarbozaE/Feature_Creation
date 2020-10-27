# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:06:04 2020

@author: Esteban
"""
import pandas as pd
import numpy as np
import Data
import Add_features
import Math_transformations

def download(f_inicio = '2016-01-01', f_fin = '2020-10-26'):
    # Download prices from Oanda into df_pe
    instrumento = "EUR_USD"
    
    f_inicio = pd.to_datetime(f_inicio+' 17:00:00').tz_localize('GMT')
    f_fin = pd.to_datetime(f_fin+' 17:00:00').tz_localize('GMT')
    
    token = '40a4858c00646a218d055374c2950239-f520f4a80719d9749cc020ddb5188887'
    
    df_pe = Data.getPrices(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran = "D",
                           p3_inst = instrumento, p4_oatk = token, p5_ginc = 4900)
    df_pe = df_pe.set_index('TimeStamp') #set index to date
    
    df_pe = pd.DataFrame(df_pe.values.astype(float), 
                         columns = df_pe.columns, 
                         index = df_pe.index)
    return df_pe


def add_all_features(df_pe):
    # Add fracdiff features
    df_pe = Add_features.add_fracdiff_features(df_pe, threshold = 1e-4)
    # Technical Indicators
    df_pe['CCI'] = Add_features.CCI(df_pe, 14) # Add CCI
    df_pe['SMA_5'] = Add_features.SMA(df_pe, 5)
    df_pe['SMA_10'] = Add_features.SMA(df_pe, 10)
    df_pe['MACD'] = Add_features.df_pe['SMA_10']-df_pe['SMA_5']
    df_pe['Upper_BB'], df_pe['Lower_BB'] = Add_features.BBANDS(df_pe, 10)
    df_pe['Range_BB'] = (df_pe['Close']-df_pe['Lower_BB'])/(df_pe['Upper_BB']-df_pe['Lower_BB'])
    df_pe['RSI'] = Add_features.RSI(df_pe, 10)
    df_pe['Max_range'] = Add_features.price_from_max(df_pe, 20)
    df_pe['Min_range'] = Add_features.price_from_min(df_pe, 20)
    df_pe['Price_Range'] = Add_features.price_range(df_pe, 50)
    df_pe['returna'], df_pe['returna_acums'], df_pe['returnlog'], df_pe['returnlog_acum'], df_pe['binary'] = Add_features.ret_div(df_pe)
    df_pe['zscore'] = Add_features.z_score(df_pe)
    df_pe['diff1'] , df_pe['diff2'] , df_pe['diff3'] , df_pe['diff4'] , df_pe['diff5'] = Add_features.int_diff(df_pe,np.arange(1,6))
    df_pe['mova1'] , df_pe['movaf2'] , df_pe['mova3'] , df_pe['mova4'] , df_pe['mova5'] = Add_features.mov_averages(df_pe,np.arange(1,6))
    df_pe['quartiles'] = Add_features.quartiles(df_pe,10)
    # Change Point Detection
    #df_pe['Windows'] = Add_features.window(df_pe)[2]
    #df_pe['binary_c'] = Add_features.binary(df_pe)[2]
    return df_pe

def create(inicio: str, fin: str):
    '''
    inicio, fin are str dates in format 'yyyy-mm-dd'
    '''
    df_pe = download(inicio, fin)
    df_pe = add_all_features(df_pe)
    df_pe = df_pe.dropna()
    df_pe = Math_transformations.math_transformations(df_pe)
    df_pe['Label'] = Add_features.next_day_ret(df_pe)[1]
    return df_pe
    
dataset = create('2015-01-01', '2020-10-26')