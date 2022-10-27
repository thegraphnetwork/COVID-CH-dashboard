#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 22:31:54 2021

@author: eduardoaraujo
"""

import pandas as pd
import streamlit as st
import geopandas as gpd
import numpy as np
from scipy.signal import correlate, correlation_lags
import scipy.cluster.hierarchy as hcluster
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import config
engine = create_engine(config.DB_URI)

def get_curve_all(curve):
    
    '''
    Function to return the cases or hosp for all the switzerland
    '''
    
    df2 = pd.read_sql(f"select datum, sum(entries) from switzerland.foph_{curve}_d group by datum;", engine)
    df2.index = pd.to_datetime(df2.datum)
    df2 = df2.rename(columns={'sum': 'entries'})
    df2.sort_index(inplace = True)
    df2
    
    return df2

def get_curve(curve, canton):
    '''
    Function to get the cases and hosp data in the database
    '''
    
    dict_columns = {'cases':'datum, entries, georegion',
                    'hosp':'datum, entries, georegion',
                    'hospcapacity': 'date, icu_covid19patients, georegion,total_covid19patients, totalpercent_covid19patients'}
    
    dict_dates = {'cases': 'datum', 'hosp': 'datum', 
                  'hospcapacity': 'date'}
    
    df = pd.read_sql(f"select {dict_columns[curve]} from switzerland.foph_{curve}_d where georegion='{canton}';", engine)
    
    df.index = pd.to_datetime(df[dict_dates[curve]])
    
    df = df.sort_index()
    
    return df
    
def get_lag(x, y, maxlags=5, smooth=True):
    if smooth:
        x = pd.Series(x).rolling(7).mean().dropna().values
        y = pd.Series(y).rolling(7).mean().dropna().values
    corr = correlate(x, y, mode='full')/np.sqrt(np.dot(x, x)*np.dot(y, y))
    slice = np.s_[(len(corr)-maxlags)//2:-(len(corr)-maxlags)//2]
    corr = corr[slice]
    lags = correlation_lags(x.size, y.size, mode='full')
    lags = lags[slice]
    lag = lags[np.argmax(corr)]

#     lag = np.argmax(corr)-(len(corr)//2)

    return lag, corr.max()

# @st.cache


def lag_ccf(a, maxlags=30, smooth=True):
    """
    Calculate the full correlation matrix based on the maximum correlation lag 
    """
    ncols = a.shape[1]
    lags = np.zeros((ncols, ncols))
    cmat = np.zeros((ncols, ncols))
    for i in range(ncols):
        for j in range(ncols):
            #             if j>i:
            #                 continue
            lag, corr = get_lag(a.T[i], a.T[j], maxlags, smooth)
            cmat[i, j] = corr
            lags[i, j] = lag
    return cmat, lags

# @st.cache


def compute_clusters(curve, t, plot=False):
    '''
    Function to compute the clusters 

    param curve: string. Represent the curve that will used to cluster the regions.

    param t: float. Represent the value used to compute the distance between the cluster and so 
    decide the number of clusters 

    return: array. 
    -> cluster: is the array with the computed clusters
    -> all_regions: is the array with all the regions
    '''

    df = pd.read_sql_table(f'foph_{curve}_d', engine, schema = 'switzerland', columns = ['datum','georegion',  'entries'])
    
    df.index = pd.to_datetime(df.datum)

    inc_canton = df.pivot(columns='georegion', values='entries')

    # changin the data
    # print(inc_canton)

    # Computing the correlation matrix based on the maximum correlation lag

    del inc_canton['CHFL']

    del inc_canton['CH']

    cm, lm = lag_ccf(inc_canton.values)

    # Plotting the dendrogram
    linkage = hcluster.linkage(cm, method='complete')

   
    if plot: 
        fig, ax = plt.subplots(1,1, figsize=(15,10), dpi = 300)
        hcluster.dendrogram(linkage, labels=inc_canton.columns, color_threshold=0.3, ax=ax)
        ax.set_title('Result of the hierarchical clustering of the series', fontdict= {'fontsize': 20})
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
    else: 
        fig = None
    

    # computing the cluster

    ind = hcluster.fcluster(linkage, t, 'distance')

    grouped = pd.DataFrame(list(zip(ind, inc_canton.columns))).groupby(
        0)  # preciso entender melhor essa linha do código
    clusters = [group[1][1].values for group in grouped]

    all_regions = df.georegion.unique()

    return clusters, all_regions, fig
    

#@st.cache
def get_canton_data(curve, canton, ini_date=None):
    '''
    This function provide a dataframe for the curve selected in the param curve and
    the canton selected in the param canton

    param curve: string. One of the following options are accepted: ['cases', 'casesvaccpersons', 'covidcertificates', 'death',
                                                             'deathvaccpersons', 'hosp', 'hospcapacity', 'hospvaccpersons',
                                                             'intcases', 're', 'test', 'testpcrantigen', 'virusvariantswgs']
    param canton: array with all the cantons of interest.

    return dataframe
    '''
    
    # dictionary with the columns that will be used for each curve. 
    dict_cols = {'cases': ['georegion', 'datum', 'entries'], 
                 'test': ['georegion', 'datum', 'entries', 'entries_pos'],
                 'hosp': ['georegion', 'datum', 'entries'], 
                 'hospcapacity': ['georegion', 'date', 'icu_covid19patients','total_covid19patients' ],
                 're': ['georegion', 'date', 'median_r_mean']
                 }

    # getting the data from the databank
    df = pd.read_sql_table(f'foph_{curve}_d', engine, schema = 'switzerland', columns = dict_cols[curve])

    # filtering by cantons
    df = df.loc[(df.georegion.isin(canton))]

    if (curve == 're') | (curve == 'hospcapacity'):
        df.index = pd.to_datetime(df.date)

    else:
        df.index = pd.to_datetime(df.datum)
        df = df.sort_index()

    if ini_date != None:
        df = df[ini_date:]

    return df

def get_ch_map():
    chmap = gpd.read_postgis("select * from switzerland.map_cantons;", 
                    con=engine, geom_col="geometry")
    chmap['geoRegion'] = chmap.HASC_1.transform(lambda x: x.split('.')[-1])
    return chmap
