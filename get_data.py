#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 22:31:54 2021

@author: eduardoaraujo
"""

import pandas as pd
import numpy as np
from scipy.signal import correlate, correlation_lags
import scipy.cluster.hierarchy as hcluster
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
#import streamlit as st

engine = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/epigraphhub")


def build_lagged_features(dt, maxlag=2, dropna=True):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    :param dt: Dataframe containing features
    :param maxlag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, maxlag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(maxlag + 1)
        res = pd.concat([dt.shift(i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res
# @st.cache


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

    df = pd.read_sql_table(f'foph_{curve}', engine, schema='switzerland', columns=[
                           'datum', 'geoRegion',  'entries'])
    df.index = pd.to_datetime(df.datum)

    inc_canton = df.pivot(columns='geoRegion', values='entries')

    # changin the data
    # print(inc_canton)

    # Computing the correlation matrix based on the maximum correlation lag

    del inc_canton['CHFL']

    del inc_canton['CH']

    cm, lm = lag_ccf(inc_canton.values)

    # Plotting the dendrogram
    linkage = hcluster.linkage(cm, method='complete')

    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
    hcluster.dendrogram(linkage, labels=inc_canton.columns,
                        color_threshold=0.3, ax=ax)
    ax.set_title('Result of the hierarchical clustering of the series',
                 fontdict={'fontsize': 20})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if plot:
        plt.show()

    # computing the cluster

    ind = hcluster.fcluster(linkage, t, 'distance')

    grouped = pd.DataFrame(list(zip(ind, inc_canton.columns))).groupby(
        0)  # preciso entender melhor essa linha do código
    clusters = [group[1][1].values for group in grouped]

    all_regions = df.geoRegion.unique()

    return clusters, all_regions, fig

# @st.cache


def get_cluster_data(curve, georegion):
    '''
    This function provide a dataframe with the curve selected in the param curve for each region selected in the 
    param georegion

    param curve: string. The following options are accepted: ['cases', 'death',
                                                              'hosp', 'hospCapacity', 
                                                              're', 'test', 'testPcrAntigen', 'virusVariantsWgs']
    param georegion: array with all the geoRegions of interest.

    return dataframe
    '''

    df = get_canton_data(curve, georegion)
    # print(df)
    # dataframe where will the curve for each region

    df_end = pd.DataFrame()

    for i in georegion:

        # print(i)

        if curve == 'hospcapacity':
            df_aux = df.loc[df.geoRegion == i].resample('D').mean()
            df_end['ICU_patients_'+i] = df_aux.ICU_Covid19Patients
        else:
            df_end[curve+'_'+i] = df.loc[df.geoRegion == i].entries

    df_end = df_end.resample('D').mean()

    return df_end


# @st.cache
def get_combined_data(data_types, georegion, vaccine=True, smooth=True):
    '''
    This function provide a dataframe with the all the curves selected in the param data_types for each region selected in the 
    param georegion

    param data_types: array. The following options are accepted: ['cases', 'casesVaccPersons', 'covidCertificates', 'death',
                                                             'deathVaccPersons', 'hosp', 'hospCapacity', 'hospVaccPersons',
                                                             'intCases', 're', 'test', 'testPcrAntigen', 'virusVariantsWgs']
    param georegion: array with all the geoRegions of interest.

    return dataframe
    '''

    for i in np.arange(0, len(data_types)):

        if i == 0:

            df = get_cluster_data(data_types[i], georegion)

        else:

            df = df.merge(get_cluster_data(
                data_types[i], georegion), left_index=True, right_index=True)

    if vaccine == True:
        # adicionando os dados de vacinação para a suiça disponibilizados pelo
        # Our World in Data
        vac = pd.read_csv(
            'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')

        # selecionando apenas os dados da suiça
        vac = vac.loc[vac.iso_code == 'CHE']
        vac.index = pd.to_datetime(vac.date)

        vac = vac[['total_vaccinations_per_hundred']]

        vac = vac.fillna(0)

        if vac.total_vaccinations_per_hundred[-1] == 0:
            vac.total_vaccinations_per_hundred[-1] = vac.total_vaccinations_per_hundred[-2]

        df['vac_all'] = vac.total_vaccinations_per_hundred

    # filling the NaN values by zero
    df = df.fillna(0)

    if smooth == True:
        df = df.rolling(window=7).mean()

        df = df.dropna()

    return df


# @st.cache
def get_canton_data(curve, canton, ini_date=None):

    dict_cols = {'cases': ['geoRegion', 'datum', 'entries'], 'test': ['geoRegion', 'datum', 'entries', 'entries_pos'],
                 'hosp': ['geoRegion', 'datum', 'entries'], 'hospcapacity': ['geoRegion', 'date', 'ICU_Covid19Patients'], 're': ['geoRegion', 'date', 'median_R_mean']
                 }

    df = pd.read_sql_table(
        f'foph_{curve}', engine, schema='switzerland', columns=dict_cols[curve])

    df = df.loc[(df.geoRegion.isin(canton))]

    if (curve == 're') | (curve == 'hospcapacity'):
        df.index = pd.to_datetime(df.date)

    else:
        df.index = pd.to_datetime(df.datum)
        df = df.sort_index()

    if ini_date != None:
        df = df[ini_date:]

    return df
