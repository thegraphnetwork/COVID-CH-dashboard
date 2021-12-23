#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:21:46 2021

@author: eduardoaraujo
"""

# importando os pacotes necessários
import pandas as pd 
import numpy as np 
import io
import base64
import json
import pickle
import uuid
import re
import plotly.graph_objects as go 
import streamlit as st 
from get_data import *
from sqlalchemy import create_engine
engine = create_engine("postgresql://epigraph:epigraph@localhost:5432/epigraphhub")


def plot_predictions(title = None):
    ''''
    Function to plot the predictions
    '''
    
    target_curve_name = 'hosp'
    canton = 'GE'
    
    df_val = pd.read_sql_table(f'ml_validation', engine, schema = 'switzerland', index_col = 'date')
    
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
        
        title = f"{names[target_curve_name]} - {canton}"

    fig.update_layout(width=900, height=500, title={
            'text': title,
            'y':0.87,
            'x':0.42,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title=f'New {names[target_curve_name]}',
    template = 'plotly_white')

    # adicionando os traços 

    # Data
    fig.add_trace(go.Scatter(x = target.index, y = target.values, name = 'Data',line=dict(color = 'black')))
    
    #fig.add_trace(go.Scatter(x = get_real_data('2021-12-09').index, y = get_real_data('2021-12-09').ID.values, name = 'Updated Data',line=dict(color = 'green')))

    #print(get_real_data(target.index[0]).index)
    # separação entre os dados de treino e de teste
    fig.add_trace(go.Scatter(x=[point, point], y=[min_val,max_val], name="Out of sample predictions", mode = 'lines',line=dict(color = '#1CA71C', dash = 'dash')))

    # Separação entre os dados de teste e o forecast
    fig.add_trace(go.Scatter(x=[target.index[-1], target.index[-1]], y=[min_val,max_val], name="Forecast", mode = 'lines',line=dict(color = '#FB0D0D', dash = 'dash')))

    # KNN
    # LightGBM
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




def plot_forecast(title= None):
    ''''
    Function to plot the forecast 
    '''
    target_curve_name = 'hosp'
    canton = 'GE'
    
    
    df_for = pd.read_sql_table('ml_forecast', engine, schema = 'switzerland', index_col = 'date')
    
    ydata = get_canton_data('hosp', ['GE'])
    ydata =  ydata.rolling(7).mean().dropna()
    
    
    dates_forecast = df_for.index
    forecast5 = df_for['lower']
    forecast50 = df_for['median']
    forecast95 = df_for['upper']
    
    fig = go.Figure()

    # Dict with names for the curves 
    names = {'hosp': 'Forecast Hospitalizations', 'ICU_patients': 'Forecast ICU patients'}
    
    if title == None: 
        
        title = f"{names[target_curve_name]} - {canton}"

    fig.update_layout(width=900, height=500, title={
            'text': title,
            'y':0.87,
            'x':0.42,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title=f'New {names[target_curve_name]}',
    template = 'plotly_white')

    # adicionando os traços 

    # Data
    fig.add_trace(go.Scatter(x = ydata.index[-150:], y = ydata.entries[-150:], name = 'Data',line=dict(color = 'black')))

    # Separation between data and forecast 
    fig.add_trace(go.Scatter(x=[ydata.index[-1], ydata.index[-1]], y=[min( min(ydata.entries[-150:]), min(forecast95)),max(max(ydata.entries[-150:]), max(forecast95))], name="Data/Forecast", mode = 'lines',line=dict(color = '#FB0D0D', dash = 'dash')))

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

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            #object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
            towrite.seek(0)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    return dl_link



def app(): 
    
    st.title('Forecast of Daily Hospitalizations')
    
    st.write('''
             To forecast the daily hospitalizations in canton Geneva, we have applied 
             Gradient Boosting Machine quantile regression model (LightGBM).

             This model is non-parametric, quite robust to non-Gaussian data, and
             includes automatically variable (predictor) selection. 
             
             In the model, we use as predictors the series of cases, hospitalizations,
             tests and ICU occupations from all the cantons belonging to the same cluster
             of the one we are forecasting for, as well as total vaccinations per hundred thousand in 
             Switzerland. The regression, in somewhat compact notation, is defined as  ''')
             
    st.latex(r'''
             H_{k,t} \sim C_{k,t-\tau_i} + H_{k,t-\tau_i} +V_{k,t-\tau_i} + ICU_{k,t-\tau_i},
             ''')
    
    st.write(r'''
             where C stands for cases, H stands for hospitalizations, V for vaccination,
             and ICU for the number of ICU patients. The model bases its predictions
             on the last 14 days: $\tau_1, \tau_2, \ldots, \tau_{14}$ and predicts the
             next 14 days. For each of these 14 days, one model is trained. 
             ''')
             
    st.write('''
             In the Figure below, it is plotted the predictions of the model 
             in the sample and out the sample.  

             ''')
    fig  = plot_predictions() 
    st.plotly_chart(fig, use_container_width = True)
    
    fig_for, df = plot_forecast()
    
    st.write('''
             In the Figure below, it is plotted the forecast for the next 14 days.
             The number of hospitalizations forecasted is also shown in the table below,
             the lower and upper columns represent the 95% confidence interval.
             The table can be downloaded by clicking on the button. 

             ''')
    
    st.plotly_chart(fig_for, use_container_width = True)
    
    st.write('## Forecasted Number of Daily Hospitalizations')
    "The table below contains the median and 90% bands for the forecasts."
    
    # df['lower'] = df['lower'].astype(float)
    # st.table(df)
    
    filename = 'forecast_hosp.csv'
    download_button_str = download_button(df, filename, f'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)
    
    