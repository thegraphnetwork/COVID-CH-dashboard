#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:21:46 2021

@author: eduardoaraujo
"""

# importando os pacotes necessários
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import io
from io import BytesIO
import base64
import json
import pickle
import uuid
import re
import plotly.graph_objects as go 
import streamlit as st 
from get_data import get_canton_data, get_updated_data, get_curve
from sqlalchemy import create_engine
engine = create_engine("postgresql://epigraph:epigraph@localhost:5432/epigraphhub")





def scatter_plot_cases_hosp(region):
    
    if region == 'All':
        
        cases = get_curve('cases')
        
        cases = cases.resample('D').sum()
        
        hosp = get_curve('hosp')
        
        hosp = hosp.resample('D').sum()
        
        fig, ax = plt.subplots()
        
        sns.regplot(x = cases[:'2020-12-31'].entries,
                    y=hosp[:'2020-12-31'].entries,
                    label = '2020', color ='tab:gray', ax =ax )
        
        sns.regplot(x = cases['2021-01-01':'2021-03-31'].entries,
                    y=hosp['2021-01-01':'2021-03-31'].entries,
                    label = '2021-Q1', color ='tab:cyan', ax =ax )
        
        sns.regplot(x = cases['2021-04-01':'2021-06-30'].entries,
                    y=hosp['2021-04-01':'2021-06-30'].entries,
                    label = '2021-Q2', color ='tab:orange', ax =ax )
        
        sns.regplot(x = cases['2021-07-01':'2021-09-30'].entries,
                    y=hosp['2021-07-01':'2021-09-30'].entries,
                    label = '2021-Q3', color ='tab:red', ax =ax )
        
        
        sns.regplot(x = cases['2021-10-01':'2021-12-31'].entries,
                    y=hosp['2021-10-01':'2021-12-31'].entries,
                    label = '2021-Q4', color ='tab:blue', ax =ax )
        
        sns.regplot(x = cases['2022-01-01':'2022-03-31'].entries,
                    y=hosp['2022-01-01':'2022-03-31'].entries,
                    label = '2022-Q1', color ='tab:green', ax =ax )
        
        ax.set_title('Cases vs hospitalizations in Switzerland')
        
        ax.set_xlabel('Cases')
        ax.set_ylabel('Hospitalizations')
        ax.legend()
        ax.grid()
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        
    
    if region == 'GE':
        fig, ax = plt.subplots()
        
        cases = get_canton_data('cases', ['GE'])
        
        hosp = get_canton_data('hosp', ['GE'])
        
        sns.regplot(x = cases['2021-10-01':'2021-12-31'].entries,
                    y=hosp['2021-10-01':'2021-12-31'].entries,
                    label = '2021-Q4', color ='tab:blue', ax =ax )
        
        sns.regplot(x = cases['2022-01-01':'2022-03-31'].entries,
                    y=hosp['2022-01-01':'2022-03-31'].entries,
                    label = '2022-Q1', color ='tab:green', ax =ax )
        
        ax.set_title('Cases vs hospitalizations in Geneva')
        
        ax.set_xlabel('Cases')
        ax.set_ylabel('Hospitalizations')
        
        ax.grid()
    
        buf = BytesIO()
        fig.savefig(buf, format="png")
    
    return buf

def plot_predictions(table_name, curve, title = None):
    ''''
    Function to plot the predictions
    
    params table_name: Name of the table with the predictions (name used to save
                                                               the table in the database)
    
    params curve: Curve related with the predictions that will be plotted
    
    return plotly figure
    '''
    target_curve_name = curve
    
    canton = 'GE'
    
    df_val = pd.read_sql_table(table_name, engine, schema = 'switzerland', index_col = 'date')
    
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


def plot_cases():
    ''''
    Function to plot the new cases according to FOPH in GE
    
    return[0] plotly figure
    return[1] last data of new cases reported
    
    '''
    
    df = pd.read_sql_table('foph_cases', engine, schema = 'switzerland', index_col = 'datum', columns=[ 'geoRegion', 'entries'])
    df.index = pd.to_datetime(df.index)
    
    df = df.loc[df.geoRegion =='GE']
    df.sort_index(inplace = True)
    df = df['2021-08-01':]
    
    # computing the rolling average 
    m_movel = df.rolling(7).mean().dropna()
    
    fig = go.Figure()
        
    title = "GE"
    
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

def plot_hosp():
    ''''
    Function to plot the number of new hospitalizations for GE
    
    
    returns plotly figure
    
    '''
    
    df = pd.read_sql_table('foph_hosp', engine, schema = 'switzerland', index_col = 'datum', columns=[ 'geoRegion', 'entries'])
    df.index = pd.to_datetime(df.index)
    
    df = df.loc[df.geoRegion =='GE']
    df.sort_index(inplace = True)
    df = df['2021-08-01':]
    
    # computing the rolling average 
    m_movel = df.rolling(7).mean().dropna()
    
    fig = go.Figure()
        
    title = "GE"
    
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
    
    




def plot_forecast(table_name, curve, title= None):
    ''''
    Function to plot the forecast 
    
    params table_name: Name of the table with the predictions (name used to save
                                                      the table in the database)
    
    params curve: Curve related with the predictions that will be plotted
    
    return[0] plotly figure
    return[1] dataframe with the values forecasted 
    
    '''
    target_curve_name = curve
    canton = 'GE'
    
    
    df_for = pd.read_sql_table(table_name, engine, schema = 'switzerland', index_col = 'date')
    
    if table_name == 'ml_forecast_hosp_up':
        ydata = get_updated_data(smooth = True)
        
    else:
        curves = {'hosp': 'hosp', 'ICU_patients': 'hospcapacity'}
        ydata = get_canton_data(curves[curve], ['GE'])
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
    
    if table_name == 'ml_forecast_hosp_up':
        fig.add_trace(go.Scatter(x = ydata.index[-150:], y = ydata.hosp_GE[-150:], name = 'Data',line=dict(color = 'black')))
    
        # Separation between data and forecast 
        fig.add_trace(go.Scatter(x=[ydata.index[-1], ydata.index[-1]], y=[min( min(ydata.hosp_GE[-150:]), min(forecast95)),max(max(ydata.hosp_GE[-150:]), max(forecast95))], name="Data/Forecast", mode = 'lines',line=dict(color = '#FB0D0D', dash = 'dash')))

    
    else: 
        
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
    
    st.title('Switzerland COVID-19 Hospitalization forecasts')
    
    fig_c, last_date, last_cases = plot_cases()
    fig_h, last_hosp = plot_hosp()
    
    st.write(f'''
            ## Current Status in Geneva
             On  **{last_date.date()}**, the FOPH (Federal Office of Public Health) reported {last_cases} new cases and {last_hosp} new hospitalizations.

            For forecasts of other cantons, see sidebar menu.
             ''')
             
    
    
    
    st.plotly_chart(fig_c, use_container_width = True)
    st.plotly_chart(fig_h, use_container_width = True)
    
    
    st.write('''
             ##### Relation between cases and hospitalizations in Geneva:
        ''')
        
    scatter_cases_hosp_GE = scatter_plot_cases_hosp('GE')
    
    st.image(scatter_cases_hosp_GE)
    
    st.write('''
             ##### Relation between cases and hospitalizations in Switzerland:
        ''')
        
    scatter_cases_hosp_all = scatter_plot_cases_hosp('All')
    
    st.image(scatter_cases_hosp_all)
    
    st.title('Forecast of Daily Hospitalizations')
    
    st.write('''
             Daily hospitalizations in canton Geneva, were forecasted using a 
             Gradient Boosting Machine quantile regression model (LightGBM).
             
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
            ## Model Validation
             In the Figure below, the model's predictions are plotted against data, both in sample (for 
             the data range used for training) and out of  sample (part of the series not used during model training).  

             ''')
    fig  = plot_predictions('ml_validation_hosp_up', curve = 'hosp') 
    st.plotly_chart(fig, use_container_width = True)
    
    st.write('''
             Below, we have the same as above, but for the ICU occupancy.  

             ''')
    fig  = plot_predictions('ml_validation_icu', curve = 'ICU_patients') 
    st.plotly_chart(fig, use_container_width = True)
    
    
    st.write('''
    ## 14-day Forecasts
             Below, we have the forecast for the next 14 days, for both Hospitalizations,
             and ICU occupancy
             
             The 95% confidence bounds are also shown
             The table with the forecasts can be downloaded by clicking on the button. 

            The checkbox selects the usage of HUG updated hospitalization data, in the generation of the forecasts.

             ''')
             
    select_data = st.checkbox('Updated data', value=True)

    if select_data:
         fig_for, df_hosp = plot_forecast('ml_forecast_hosp_up', curve = 'hosp')
         st.plotly_chart(fig_for, use_container_width = True)
         filename = 'forecast_hosp.csv'
         download_button_str = download_button(df_hosp, filename, 'Download data', pickle_it=False)

         st.markdown(download_button_str, unsafe_allow_html=True)
         
    else:
         fig_for, df_hosp = plot_forecast('ml_forecast_hosp', curve = 'hosp')
         st.plotly_chart(fig_for, use_container_width = True)
         filename = 'forecast_hosp.csv'
         download_button_str = download_button(df_hosp, filename, 'Download data', pickle_it=False)

         st.markdown(download_button_str, unsafe_allow_html=True)
        
    
             
    fig_for, df_icu = plot_forecast('ml_forecast_icu', curve = 'ICU_patients')
    st.plotly_chart(fig_for, use_container_width = True)
    filename = 'forecast_ICU.csv'
    download_button_str = download_button(df_icu, filename, 'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)

    
    
    
    