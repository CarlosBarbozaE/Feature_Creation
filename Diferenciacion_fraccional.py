# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:09:49 2020

@author: Esteban
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 

msft = yf.Ticker('MSFT')
hist = msft.history(period='10y')
close = pd.DataFrame(np.log(hist.Close))


def findWeights_FFD(d, length, threshold):
    #set first weight to be a 1 and k to be 1
    w, k = [1.], 1
    w_curr = 1
    
    #while we still have more weights to process, do the following:
    while(k < length):
        w_curr = (-w[-1]*(d-k+1))/k
        #if the current weight is below threshold, exit loop
        if(abs(w_curr) <= threshold):
            break
        #append coefficient to list if it passes above threshold condition
        w.append(w_curr)
        #increment k
        k += 1
    #make sure to convert it into a numpy array and reshape from a single row to a single
        # column so they can be applied to time-series values easier
    w = np.array(w[::-1]).reshape(-1,1)
    
    return w


def fracdiff_threshold(series, d, threshold):
    # return the time series resulting from (fractional) differencing
    length = len(series)
    weights=findWeights_FFD(d, length, threshold)
    weights = weights[::-1]
    res=0
    for k in range(len(weights)):
        res += weights[k]*series.shift(k).fillna(0)
    return res[len(weights):]


def corrvalues(series, dRange, step):
    difs = pd.DataFrame(series.Close)
    for i in np.arange(dRange[0], dRange[1]+step,step):
      difs['Diff %s'%i] = fracdiff_threshold(close,i,1e-4) # Where to set the threshold? More history allows smaller threshold (making the series bigger)
    corr_series = difs.corr().Close
    return corr_series


def plotMemoryVsCorr(result, seriesName):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()  
    color1='xkcd:deep red'; color2='xkcd:cornflower blue'
    ax.plot(result.order,result['adf'],color=color1)
    ax.plot(result.order, result['5%'], color='xkcd:slate')
    ax.plot(result.order, result['1%'], color='xkcd:slate')
    ax2.plot(result.order,result['corr'], color=color2)
    ax.set_xlabel('Order of differencing')
    ax.set_ylabel('ADF', color=color1);ax.tick_params(axis='y', labelcolor=color1)
    ax2.set_ylabel('Corr', color=color2); ax2.tick_params(axis='y', labelcolor=color2)
    #plt.title('ADF test statistics and correlation for %s' % (seriesName))
    plt.show()


def MemoryVsCorr(series, dRange, step, threshold):
    # return a data frame and plot comparing adf statistics and linear correlation
    # for numberPlots orders of differencing in the interval dRange up to a lag_cutoff coefficients
    corr_series = corrvalues(series,dRange,step)
    interval=np.arange(dRange[0], dRange[1]+step,step)
    result=pd.DataFrame(np.zeros((len(interval),4)))
    result.columns = ['order','adf','corr', '5%']
    result['order']=interval
    for counter,order in enumerate(interval):
        seq_traf=fracdiff_threshold(close,order,threshold)
        res=adfuller(seq_traf, maxlag=1, regression='c') #autolag='AIC'
        result.loc[counter,'adf']=res[0]
        result.loc[counter,'5%']=res[4]['5%']
        result.loc[counter,'1%']=res[4]['1%']
        result.loc[counter,'corr']= corr_series[counter+1]
    plotMemoryVsCorr(result, 'MSFT')
    return result


def least_diff(series, dRange: tuple, step, threshold, confidence: str):
    '''
    Function to fractionally differentiate a series using the minimum degree
    possible to make the series stationary.
    "series" is expected to be a DataFrame with the (log) close prices and with column name="Close"
    but can be any series in a DataFrame
    Stationarity is determined using ADF test
    Returns result as a DataFrame with the order of differentiation, the ADF test
    and the threshold value for the confidence interval used
    It also returns the differentiated series as a DataFrame
    '''
    interval=np.arange(dRange[0], dRange[1]+step,step)
    result=pd.DataFrame(columns = ['order','adf', confidence])
    deg = interval[0]
    seq_traf=fracdiff_threshold(series,deg,threshold)
    res=adfuller(seq_traf, maxlag=1, regression='c') #autolag='AIC'
    while res[0]>res[4][confidence]:
      deg += step
      res=adfuller(fracdiff_threshold(series,deg,threshold), maxlag=1, regression='c')
    interval=np.arange(deg-step, deg, step/10)
    deg = interval[0]
    seq_traf=fracdiff_threshold(series,deg,threshold)
    res=adfuller(seq_traf, maxlag=1, regression='c') #autolag='AIC'
    while res[0]>res[4][confidence]:
      deg += step/10
      seq_traf=fracdiff_threshold(series,deg,threshold)
      res=adfuller(seq_traf, maxlag=1, regression='c')
    result['order']=[deg]
    result['adf']=[res[0]]
    result[confidence]=[res[4][confidence]]
      
    return result,seq_traf