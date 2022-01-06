#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 09:20:09 2022

@author: eduardoaraujo
"""
import pandas as pd 
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from forecast import download_button
from get_data import get_canton_data  
from sqlalchemy import create_engine
from cachetools import cached
engine = create_engine("postgresql://epigraph:epigraph@localhost:5432/epigraphhub")



# @st.cache(allow_output_mutation=True)
@cached(cache={})
def get_curve(curve):
    '''
    Function to get the cases and hosp data in the database
    '''
    df = pd.read_sql_table(f'foph_{curve}', engine, schema = 'switzerland', index_col = 'datum', columns=[ 'geoRegion', 'entries'])
    df.index = pd.to_datetime(df.index)
    
    return df

def plot_cases_canton(canton):
    
    ''''
    Function to plot the new cases according to FOPH in any canton
    
    params canton: canton to plot the data
    
    return[0] plotly figure
    return[1] last data of new cases reported
    
    '''
    
    df = get_curve('cases')
    
    df = df[df.geoRegion == canton]
    df.sort_index(inplace = True)
    df = df['2021-08-01':]
    
    # computing the rolling average 
    m_movel = df.rolling(7).mean().dropna()
    
    fig = go.Figure()
        
    title = f"{canton}"
    
    fig.update_layout(width=900, height=500, title={
                'text': title,
                'y':0.87,
                'x':0.42,
                'xanchor': 'center',
                'yanchor': 'top'},
        xaxis_title='Report Date',
        yaxis_title='New cases',
      template = 'plotly_white')
    
    fig.add_trace(go.Bar(x = df.index, y = df.entries, name = 'New cases', marker_color='rgba(31, 119, 180, 0.7)'))
    
    
    fig.add_trace(go.Scatter(x = m_movel.index, y = m_movel.entries, name = 'Rolling average',line=dict(color = 'black', width = 2)))
        
    fig.update_xaxes( showgrid=True, gridwidth=1, gridcolor='lightgray',zeroline = False,
        showline=True, linewidth=1, linecolor='black', mirror = True)
    
    fig.update_yaxes( showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline = False,
        showline=True, linewidth=1, linecolor='black', mirror = True)
    
    return fig, df.index[-1], df.entries[-1]

def plot_hosp_canton(canton):
    '''
    Function to plot the number of new hospitalizations for any canton
    
    params canton: canton to plot the hospitalization data
    returns plotly figure
    
    '''
    
    df = get_curve('hosp')
    
    df = df.loc[df.geoRegion ==canton]
    df.sort_index(inplace = True)
    df = df['2021-08-01':]
    
    # computing the rolling average 
    m_movel = df.rolling(7).mean().dropna()
    
    fig = go.Figure()
        
    title = f"{canton}"
    
    fig.update_layout(width=900, height=500, title={
                'text': title,
                'y':0.87,
                'x':0.42,
                'xanchor': 'center',
                'yanchor': 'top'},
        xaxis_title='Report Date',
        yaxis_title='New hospitalizations',
      template = 'plotly_white')
    
    fig.add_trace(go.Bar(x = df.index, y = df.entries, name = 'New hospitalizations', marker_color='rgba(31, 119, 180, 0.7)'))
    
    
    fig.add_trace(go.Scatter(x = m_movel.index, y = m_movel.entries, name = 'Rolling average',line=dict(color = 'black', width = 2)))
        
    fig.update_xaxes( showgrid=True, gridwidth=1, gridcolor='lightgray',zeroline = False,
        showline=True, linewidth=1, linecolor='black', mirror = True)
    
    fig.update_yaxes( showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline = False,
        showline=True, linewidth=1, linecolor='black', mirror = True)
    
    return fig, df.entries[-1]

def plot_predictions_canton(table_name, curve, canton, title = None):
    ''''
    Function to plot the predictions
    
    params table_name: Name of the table with the predictions (name used to save
                                                               the table in the database)
    
    params curve: Curve related with the predictions that will be plotted
    
    return plotly figure
    '''
    target_curve_name = curve
    
    
    df_val = pd.read_sql_table(table_name, engine, schema = 'switzerland', index_col = 'date')
    
    df_val = df_val.loc[df_val.canton == canton]
    
    target = df_val['target']
    train_size = df_val['train_size'].values[0]
    x = df_val.index
    y5 = df_val['lower']
    y50 = df_val['median']
    y95 = df_val['upper']
    
    point = target.index[train_size]
    min_val = min([min(target), np.nanmin(y50)])
    max_val = max([max(target), np.nanmax(y50)])

    point_date = np.where(target.index == '2021-01-01')

    fig = go.Figure()
    
    # Dict with names for the curves 
    names = {'hosp': 'Hospitalizations', 'ICU_patients': 'ICU patients'}
    
    if title == None: 
        
        title = f"{canton}"

    fig.update_layout(width=900, height=500, title={
            'text': title,
            'y':0.87,
            'x':0.42,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title=f'New {names[target_curve_name]}',
    template = 'plotly_white')

    # adding the traces

    # Data
    fig.add_trace(go.Scatter(x = target.index, y = target.values, name = 'Data',line=dict(color = 'black')))
    
    
    # Line separing training data and test data
    fig.add_trace(go.Scatter(x=[point, point], y=[min_val,max_val], name="Out of sample predictions", mode = 'lines',line=dict(color = '#1CA71C', dash = 'dash')))

    # Separação entre os dados de teste e o forecast
    #fig.add_trace(go.Scatter(x=[target.index[-1], target.index[-1]], y=[min_val,max_val], name="Forecast", mode = 'lines',line=dict(color = '#FB0D0D', dash = 'dash')))

    
    # LightGBM predictions
    fig.add_trace(go.Scatter(x = x, y = y50, name = 'LightGBM',line=dict(color = '#FF7F0E')))
    
    fig.add_trace(go.Scatter(x = x, y = y5, line=dict(color = '#FF7F0E',width=0), showlegend=False))
    
    fig.add_trace(go.Scatter(x = x, y = y95,line=dict(color = '#FF7F0E', width=0),
        mode='lines',
        fillcolor='rgba(255, 127, 14, 0.3)', fill = 'tonexty', showlegend= False))

    
    fig.update_xaxes( showgrid=True, gridwidth=1, gridcolor='lightgray',zeroline = False,
    showline=True, linewidth=1, linecolor='black', mirror = True)

    fig.update_yaxes( showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline = False,
    showline=True, linewidth=1, linecolor='black', mirror = True)


    return fig 

def plot_forecast_canton(table_name,canton, curve, title= None):
    ''''
    Function to plot the forecast 
    
    params table_name: Name of the table with the predictions (name used to save
                                                      the table in the database)
    
    params curve: Curve related with the predictions that will be plotted
    
    return[0] plotly figure
    return[1] dataframe with the values forecasted 
    
    '''
    target_curve_name = curve

    
    df_for = pd.read_sql_table(table_name, engine, schema = 'switzerland', index_col = 'date')
    
    df_for = df_for.loc[df_for.canton == canton]
    
    curves = {'hosp': 'hosp', 'ICU_patients': 'hospcapacity'}
    ydata = get_canton_data(curves[curve], [canton])
    ydata = ydata.resample('D').mean() 
    ydata =  ydata.rolling(7).mean().dropna()
    
    
    dates_forecast = df_for.index
    forecast5 = df_for['lower']
    forecast50 = df_for['median']
    forecast95 = df_for['upper']
    
    fig = go.Figure()

    # Dict with names for the curves 
    names = {'hosp': 'Forecast Hospitalizations', 'ICU_patients': 'Forecast ICU patients'}
    
    if title == None: 
        
        title = f"{canton}"

    fig.update_layout(width=900, height=500, title={
            'text': title,
            'y':0.87,
            'x':0.42,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title=f'New {names[target_curve_name]}',
    template = 'plotly_white')

    # adding the traces
    # Data
    
        
    column_curves = {'hosp': 'entries', 'ICU_patients': 'ICU_Covid19Patients'}
        
    fig.add_trace(go.Scatter(x = ydata.index[-150:], y = ydata[column_curves[curve]][-150:], name = 'Data',line=dict(color = 'black')))
    
    # Separation between data and forecast 
    fig.add_trace(go.Scatter(x=[ydata.index[-1], ydata.index[-1]], y=[min( min(ydata[column_curves[curve]][-150:]), min(forecast95)),max(max(ydata[column_curves[curve]][-150:]), max(forecast95))], name="Data/Forecast", mode = 'lines',line=dict(color = '#FB0D0D', dash = 'dash')))

    # LightGBM
    fig.add_trace(go.Scatter(x = dates_forecast, y = forecast50, name = 'Forecast LightGBM',line=dict(color = '#FF7F0E')))
    
    fig.add_trace(go.Scatter(x = dates_forecast, y = forecast5, line=dict(color = '#FF7F0E',width=0),mode = 'lines',  showlegend=False))
    
    fig.add_trace(go.Scatter(x = dates_forecast, y =forecast95,line=dict(color = '#FF7F0E', width=0),
        mode='lines',
        fillcolor='rgba(255, 127, 14, 0.3)', fill = 'tonexty', showlegend= False))
    
    fig.update_xaxes( showgrid=True, gridwidth=1, gridcolor='lightgray',zeroline = False,
        showline=True, linewidth=1, linecolor='black', mirror = True)

    fig.update_yaxes( showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline = False,
        showline=True, linewidth=1, linecolor='black', mirror = True)

    #if path == None: 
        #path = f"images/forecast_{canton}.png"
        
    #fig.write_image(path)
    
    del df_for['index']
    df_for.index = pd.to_datetime(df_for.index)
    df_for.index = df_for.index.date
    df_for.reset_index(inplace = True)
    
    df_for.rename(columns={'index': 'date'}, inplace = True)
    return fig, df_for


def app():
    cantons = [ 'UR', 'VD', 'OW', 'AG', 'AI', 'VS', 'FL', 'SG', 'ZG',
           'AR', 'ZH', 'BE', 'GE', 'GL', 'GR', 'BS', 'JU', 'SH', 'FR', 'BL',
           'SZ', 'SO', 'TG', 'TI', 'LU', 'NE', 'NW']

    canton = st.selectbox("On which canton you want to forecast the hospitalizations?",
                cantons 
                )
    
    st.title('Number of cases and Hospitalizations')
    
<<<<<<< HEAD
    fig_c, last_date, last_cases = plot_cases(canton)
=======
    fig_c, last_date = plot_cases_canton(canton)
>>>>>>> e2a08feae37072314dcab00e29988f7ef0c06701
    
    st.write(f'''
             The graphs below show the number of cases and hospitalizations in {canton}
             according to FOPH. The data was updated in: {str(last_date)[:10]}
             ''')
             
<<<<<<< HEAD
    fig_h, last_hosp = plot_hosp(canton)
=======
    fig_h = plot_hosp_canton(canton)
>>>>>>> e2a08feae37072314dcab00e29988f7ef0c06701
    
    st.plotly_chart(fig_c, use_container_width = True)
    st.plotly_chart(fig_h, use_container_width = True)

    st.write('''
            ## Model Validation
             In the Figure below, the model's predictions are plotted against data, both in sample (for 
             the data range used for training) and out of  sample (part of the series not used during model training).  

             ''')
    fig_val  = plot_predictions_canton('ml_val_hosp_all_cantons', curve = 'hosp', canton = canton)
    
    st.plotly_chart(fig_val, use_container_width = True)
    
    st.write('''
             Below, we have the same as above, but for the ICU occupancy.  

             ''')
             
    fig_val_icu  = plot_predictions_canton('ml_val_icu_all_cantons', curve = 'ICU_patients', canton = canton)
    
    st.plotly_chart(fig_val_icu, use_container_width = True)
    
    
    st.write('''
    ## 14-day Forecasts
             Below, we have the forecast for the next 14 days, for both Hospitalizations,
             and ICU occupancy
             
             The 95% confidence bounds are also shown
             The table with the forecasts can be downloaded by clicking on the button. 

             ''')
             
    fig_for, df_hosp = plot_forecast_canton('ml_for_hosp_all_cantons', canton= canton, curve = 'hosp')
    st.plotly_chart(fig_for, use_container_width = True)
    filename = 'forecast_hosp.csv'
    download_button_str = download_button(df_hosp, filename, 'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)
    
    fig_for_icu, df_icu = plot_forecast_canton('ml_for_icu_all_cantons', canton= canton, curve = 'ICU_patients')
    st.plotly_chart(fig_for_icu, use_container_width = True)
    filename = 'forecast_icu.csv'
    download_button_str = download_button(df_icu, filename, 'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)
    
    
    
    