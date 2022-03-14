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
from datetime import timedelta 
from io import BytesIO
import base64
import json
import pickle
import uuid
import re
import plotly.graph_objects as go
import streamlit as st
from get_data import get_canton_data, get_curve
from plots import scatter_plot_cases_hosp
from sqlalchemy import create_engine
engine = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/epigraphhub")





def plot_predictions(table_name, curve, title=None):
    ''''
    Function to plot the predictions

    params table_name: Name of the table with the predictions (name used to save
                                                               the table in the database)

    params curve: Curve related with the predictions that will be plotted

    return plotly figure
    '''
    target_curve_name = curve

    canton = 'GE'

    df_val = pd.read_sql_table(
        table_name, engine, schema='switzerland', index_col='date')

    df_val  = df_val.loc[df_val.canton == canton]

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
    names = {'hosp': 'New Hospitalizations', 'ICU_patients': 'Total ICU patients', 
             'total_hosp': 'Total hospitalizations'}

    if title == None:

        title = f"{canton}"

    fig.update_layout(width=900, height=500, title={
        'text': title,
        'y': 0.87,
        'x': 0.42,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title='Date',
        yaxis_title=f'{names[target_curve_name]}',
        template='plotly_white')

    # adding the traces

    # Data
    fig.add_trace(go.Scatter(x=target.index, y=target.values,
                  name='Data', line=dict(color='black')))

    # Line separing training data and test data
    fig.add_trace(go.Scatter(x=[point, point], y=[
                  min_val, max_val], name="Out of sample predictions", mode='lines', line=dict(color='#1CA71C', dash='dash')))

    # Separação entre os dados de teste e o forecast
    # fig.add_trace(go.Scatter(x=[target.index[-1], target.index[-1]], y=[min_val,max_val], name="Forecast", mode = 'lines',line=dict(color = '#FB0D0D', dash = 'dash')))

    # NGBoost predictions
    fig.add_trace(go.Scatter(x=x, y=y50, name='NGBoost',
                  line=dict(color='#FF7F0E')))

    fig.add_trace(go.Scatter(x=x, y=y5, line=dict(
        color='#FF7F0E', width=0), showlegend=False))

    fig.add_trace(go.Scatter(x=x, y=y95, line=dict(color='#FF7F0E', width=0),
                             mode='lines',
                             fillcolor='rgba(255, 127, 14, 0.3)', fill='tonexty', showlegend=False))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig


def plot_cases():
    ''''
    Function to plot the new cases according to FOPH in GE

    return[0] plotly figure
    return[1] last data of new cases reported

    '''

    df = get_curve('cases', 'GE')
    
    last_date = df.index[-1]
    
    df = df['2021-11-01':]
    
    df = df.iloc[:-3]

    # computing the rolling average
    m_movel = df.rolling(7).mean().dropna()

    fig = go.Figure()

    title = "Geneva"

    fig.update_layout(width=900, height=500, title={
        'text': title,
        'y': 0.87,
        'x': 0.42,
        'xanchor': 'center',
        'yanchor': 'top'},
        legend={'orientation': 'h', 'valign':'top', 'y': -0.25},
        xaxis_title='Report Date',
        yaxis_title='New cases',
        template='plotly_white')

    fig.add_trace(go.Bar(x=df.index, y=df.entries, name='New cases',
                  marker_color='rgba(31, 119, 180, 0.7)'))

    fig.add_trace(go.Scatter(x=m_movel.index, y=m_movel.entries,
                  name='Rolling average', line=dict(color='black', width=2)))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig, last_date, df.entries[-2:]

def get_hospCapacity():
    df = get_curve('hospcapacity', 'GE')
    df = df.resample('D').mean()
    df = df.sort_index()
    df = df.iloc[:-3]
    return df.Total_Covid19Patients[-2:].astype('int'),df.TotalPercent_Covid19Patients[-2:]

def plot_hosp():
    ''''
    Function to plot the number of new hospitalizations for GE


    returns plotly figure

    '''

    df = get_curve('hosp', 'GE')
    
    df = df['2021-11-01':]
    
    df = df.iloc[:-3]

    # computing the rolling average
    m_movel = df.rolling(7).mean().dropna()

    fig = go.Figure()

    title = "Geneva"

    fig.update_layout(width=900, height=500, title={
        'text': title,
        'y': 0.87,
        'x': 0.42,
        'xanchor': 'center',
        'yanchor': 'top'},
        legend={'orientation': 'h', 'valign':'top', 'y': -0.25},
        xaxis_title='Report Date',
        yaxis_title='New hospitalizations',
        template='plotly_white')

    fig.add_trace(go.Bar(x=df.index, y=df.entries,
                  name='New hospitalizations', marker_color='rgba(31, 119, 180, 0.7)'))

    fig.add_trace(go.Scatter(x=m_movel.index, y=m_movel.entries,
                  name='Rolling average', line=dict(color='black', width=2)))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig, df.entries[-2:]


def plot_forecast(table_name, curve, SEIR_preds, title=None):
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

    df_for = pd.read_sql_table(
        table_name, engine, schema='switzerland', index_col='date')
    
    df_for  = df_for.loc[df_for.canton == canton]

    #if table_name == 'ml_forecast_hosp_up':
        #ydata = get_updated_data(smooth=True)

    
    curves = {'hosp': 'hosp', 'ICU_patients': 'hospcapacity', 'total_hosp': 'hospcapacity'}
    ydata = get_canton_data(curves[curve], ['GE'])
    ydata = ydata.resample('D').mean()
    ydata = ydata.iloc[:-3]
    ydata = ydata.rolling(7).mean().dropna()

    dates_forecast = df_for.index
    forecast5 = df_for['lower']
    forecast50 = df_for['median']
    forecast95 = df_for['upper']

    fig = go.Figure()

    # Dict with names for the curves
    names = {'hosp': 'Forecast New Hospitalizations',
             'ICU_patients': 'Forecast Total ICU patients', 
             'total_hosp': 'Total hospitalizations'}

    if title == None:

        title = f"{canton}"

    fig.update_layout(width=900, height=500, title={
        'text': title,
        'y': 0.87,
        'x': 0.42,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title='Date',
        yaxis_title=f'{names[target_curve_name]}',
        template='plotly_white')

    # adding the traces
    # Data

    #if table_name == 'ml_forecast_hosp_up':
    #    fig.add_trace(go.Scatter(
    #        x=ydata.index[-150:], y=ydata.hosp_GE[-150:], name='Data', line=dict(color='black')))

        # Separation between data and forecast
    #    fig.add_trace(go.Scatter(x=[ydata.index[-1], ydata.index[-1]], y=[min(min(ydata.hosp_GE[-150:]), min(forecast95)), max(
    #        max(ydata.hosp_GE[-150:]), max(forecast95))], name="Data/Forecast", mode='lines', line=dict(color='#FB0D0D', dash='dash')))



    column_curves = {'hosp': 'entries',
                         'ICU_patients': 'ICU_Covid19Patients', 
                         'total_hosp': 'Total_Covid19Patients'}

    min_data = min(ydata.index[-1], df_for.index[0] - timedelta(days=1))

    fig.add_trace(go.Scatter(
        x=ydata.loc[:min_data].index[-150:], y=ydata.loc[:min_data][column_curves[curve]][-150:], name='Data', line=dict(color='black')))

    if SEIR_preds == False:
        # Separation between data and forecast
        fig.add_trace(go.Scatter(x=[df_for.index[0], df_for.index[0]], y=[min(min(ydata[column_curves[curve]][-150:]), min(forecast95)), max(
        max(ydata[column_curves[curve]][-150:]), max(forecast95))], name="Data/Forecast", mode='lines', line=dict(color='#FB0D0D', dash='dash')))

        # NGBoost
        fig.add_trace(go.Scatter(x=dates_forecast, y=forecast50,
                    name='Forecast NGBoost', line=dict(color='#FF7F0E')))

        fig.add_trace(go.Scatter(x=dates_forecast, y=forecast5, line=dict(
            color='#FF7F0E', width=0), mode='lines',  showlegend=False))

        fig.add_trace(go.Scatter(x=dates_forecast, y=forecast95, line=dict(color='#FF7F0E', width=0),
                                mode='lines',
                                fillcolor='rgba(255, 127, 14, 0.3)', fill='tonexty', showlegend=False))
        
    
        
    if SEIR_preds:
            
        df_scen2 = pd.read_sql_table('janne_scenario_2', engine, 
                                         schema = 'switzerland', 
                                         columns = ['Date', 'New_hospitalisations', 
                                                    'Total_hospitalisations',
                                                    'Total_ICU'])
            
        df_scen2.set_index('Date', inplace = True)
            
        df_scen2.index = pd.to_datetime(df_scen2.index)
            
        df_scen2 = df_scen2.loc[ydata.index[-150]:]
            
        df_scen3 = pd.read_sql_table('janne_scenario_3', engine, 
                                         schema = 'switzerland', 
                                         columns = ['Date', 'New_hospitalisations', 
                                                    'Total_hospitalisations',
                                                    'Total_ICU'])
            
        df_scen3.set_index('Date', inplace = True)
            
        df_scen3.index = pd.to_datetime(df_scen3.index)
            
        df_scen3 = df_scen3.loc[ydata.index[-150]:]
            
        df_scen4 = pd.read_sql_table('janne_scenario_4', engine, 
                                         schema = 'switzerland', 
                                         columns = ['Date', 'New_hospitalisations', 
                                         'Total_hospitalisations',
                                         'Total_ICU'])
            
        df_scen4.set_index('Date', inplace = True)
            
        df_scen4.index = pd.to_datetime(df_scen4.index)
            
        df_scen4 = df_scen4.loc[ydata.index[-150]:]
            
        dict_seir = {'hosp': 'New_hospitalisations',
                         'total_hosp': 'Total_hospitalisations',
                         'ICU_patients': 'Total_ICU'}


        fig.add_trace(go.Scatter(x=df_scen2.index, y= df_scen2[dict_seir[curve]],
                          name='SEIR - model: Scenario 2', line=dict(color='#D62728', dash = 'dash')))
            
        fig.add_trace(go.Scatter(x=df_scen3.index, y= df_scen3[dict_seir[curve]],
                          name='SEIR - model: Scenario 3', line=dict(color='#2CA02C', dash = 'dash')))
            
        fig.add_trace(go.Scatter(x=df_scen4.index, y= df_scen4[dict_seir[curve]],
                          name='SEIR - model: Scenario 4', line=dict(color='#1F77B4', dash = 'dash')))
                                                                                

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    # if path == None:
    #path = f"images/forecast_{canton}.png"

    # fig.write_image(path)

    del df_for['index']
    df_for.index = pd.to_datetime(df_for.index)
    df_for.index = df_for.index.date
    df_for.reset_index(inplace=True)

    df_for.rename(columns={'index': 'date'}, inplace=True)
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
            object_to_download = object_to_download.to_excel(
                towrite, encoding='utf-8', index=False, header=True)
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

    dl_link = custom_css + \
        f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def app():

    st.title('Switzerland COVID-19 Hospitalization forecasts')

    fig_c, last_date, last_cases = plot_cases()
    fig_h, last_hosp = plot_hosp()
    total_hc, perc_hc = get_hospCapacity()

    st.markdown(f'''
            ## Current Status in Geneva
            On  **{last_date.date()}**, the FOPH (Federal Office of Public Health) reported:
            ''')

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Daily new cases", value=last_cases[-1],
                  delta=f"{last_cases.diff()[-1]} cases", delta_color="inverse")
        st.metric("Total COVID-19 Hospitalizations", value=total_hc[-1],
        delta=f"{total_hc.diff()[-1]} cases", delta_color='inverse')
        st.plotly_chart(fig_c, use_container_width=True)
    # print(last_cases,last_cases.diff())
    with c2:
        st.metric("Daily new Hospitalizations", value=last_hosp[-1],
                  delta=f"{last_hosp.diff()[-1]:.1f} Hospitalizations", delta_color="inverse")
        st.metric("Percent of total Hospitalizations", value=f"{perc_hc[-1]:.2f}", 
        delta=f"{perc_hc.diff()[-1]:.2f} %",
        delta_color="inverse")
        st.plotly_chart(fig_h, use_container_width=True)

    st.write('''
            For forecasts of other cantons, see sidebar menu.
             ''')

    st.write('''
             #### Relation between cases and hospitalizations in Geneva:
             If we look at hospitalization *vs* cases, we can hint at the change in case severity over time.
        ''')

    scatter_cases_hosp_GE = scatter_plot_cases_hosp('GE')

    st.image(scatter_cases_hosp_GE)

    st.title('Forecasts')

    st.write('''
             To forecast the daily hospitalizations, total hospitalizations, and total ICU hospitalizations
             in canton Geneva, it was used a Gradient Boosting Machine that returns
             probabilistic predictions implemented in the python package `NGBoost`.
             
             In the model, we use as predictors the series of cases, hospitalizations,
             tests and ICU occupations from all the cantons belonging to the same cluster
             of the one we are forecasting for, as well as total vaccinations per hundred thousand in 
             Switzerland. The regression, in somewhat compact notation, is defined as  ''')

    st.latex(r'''
             H_{k,t} \sim C_{k,t-\tau_i} + H_{k,t-\tau_i} +V_{k,t-\tau_i} + ICU_{k,t-\tau_i},
             ''')

    st.write(r'''
             for the daily hospitalizations, where C stands for cases, H stands for hospitalizations, V for vaccination,
             and ICU for the number of ICU patients. The model bases its predictions
             on the last 14 days: $\tau_1, \tau_2, \ldots, \tau_{14}$ and predicts the
             next 14 days. For each of these 14 days, one model is trained. The regression for the 
             total hospitalizations and total ICU hospitalizations follows the same format defined above.
             ''')

    st.write('''
    ## 14-day Forecasts
    Below, we have the forecast for the next 14 days, for daily hospitalizations, total hospitalizations
    and total ICU hospitalizations. The 95% confidence bounds are also shown. The table with the forecasts can be downloaded by clicking on the button. 

             ''')
             
    st.write('''
              ## SEIR model
              
              The checkbox `SEIR - model` plots in the graph the predictions
              of an SEIR model stratified by age (In the graph are the sum of the observations in all ages).
              This model considers the scenarios below:
              
              ##### Scenario 2: Omicron included with the following (overall pessimistic) assumptions:
              - Omicron will completely take over within one month
              - Omicron is 2x more infectious and has 50% shorter incubation time than Delta
              - One or two vaccine doses have no protection against Omicron infection; three doses prevent 50% less cases than for Delta
              - The risk of severe disease is the same as with Delta

             ##### Scenario 3: Omicron included with the following assumptions (optimistic about infectiousness):
             - Omicron will completely take over within 1.5 months
             - Omicron is 1.5x more infectious and has 50% shorter incubation time than Delta
             - Vaccination prevents 50% less cases than for Delta (regardless of number of doses)
             - The risk of severe disease is the same as with Delta
                 
                 
             ##### Scenario 4: Omicron included with the following assumptions (optimistic about severe disease):
             - Omicron will completely take over within one month
             - Omicron is 2x more infectious and has 50% shorter incubation time than Delta
             - One or two vaccine doses have no protection against Omicron infection; three doses prevent 50% less cases than for Delta
             - The risk of severe disease is the same as with Delta

             ''')
             
    st.write('## Forecast results')
    st.write('### New Hospitalizations')

    #select_data = st.checkbox('Updated data', value=False)
    
    #if select_data == False:
    SEIR_preds = st.checkbox('SEIR - model', key = 'check_1', value = False )

    #if select_data:
    #    fig_for, df_hosp = plot_forecast('ml_forecast_hosp_up',curve = 'hosp',SEIR_preds = False)
    #    st.plotly_chart(fig_for, use_container_width=True)
    #    filename = 'forecast_hosp.csv'
    #    download_button_str = download_button(
    #        df_hosp, filename, 'Download data', pickle_it=False)

    #    st.markdown(download_button_str, unsafe_allow_html=True)

    #else:
        
    if SEIR_preds:
        fig_for, df_hosp = plot_forecast('ml_for_hosp_all_cantons', curve='hosp',SEIR_preds = True)
        st.plotly_chart(fig_for, use_container_width=True)
        filename = 'forecast_hosp.csv'
        download_button_str = download_button(
                df_hosp, filename, 'Download data', pickle_it=False)
    
        st.markdown(download_button_str, unsafe_allow_html=True)
        
    else:
        fig_for, df_hosp = plot_forecast('ml_for_hosp_all_cantons', curve='hosp',SEIR_preds = False)
        st.plotly_chart(fig_for, use_container_width=True)
        filename = 'forecast_hosp.csv'
        download_button_str = download_button(
            df_hosp, filename, 'Download data', pickle_it=False)
    
        st.markdown(download_button_str, unsafe_allow_html=True)
    
    st.write('### Total Hospitalizations')

    SEIR_preds_tot = st.checkbox('SEIR - model', key = 'check_2', value = False )
    
    if SEIR_preds_tot:
        fig_for, df_icu = plot_forecast('ml_for_total_all_cantons', curve='total_hosp',SEIR_preds = True)
        st.plotly_chart(fig_for, use_container_width=True)
        filename = 'forecast_total_hosp.csv'
        download_button_str = download_button(
            df_icu, filename, 'Download data', pickle_it=False)
    
        st.markdown(download_button_str, unsafe_allow_html=True)
        
    else:
        fig_for, df_icu = plot_forecast('ml_for_total_all_cantons', curve='total_hosp',SEIR_preds = False)
        st.plotly_chart(fig_for, use_container_width=True)
        filename = 'forecast_total_hosp.csv'
        download_button_str = download_button(
            df_icu, filename, 'Download data', pickle_it=False)
    
        st.markdown(download_button_str, unsafe_allow_html=True)
        
    st.write('### Total ICU Hospitalizations')

    SEIR_preds_icu = st.checkbox('SEIR - model', key = 'check_3', value = False )
    
    if SEIR_preds_icu:
        fig_for, df_icu = plot_forecast('ml_for_icu_all_cantons', curve='ICU_patients',SEIR_preds = True)
        st.plotly_chart(fig_for, use_container_width=True)
        filename = 'forecast_ICU.csv'
        download_button_str = download_button(
            df_icu, filename, 'Download data', pickle_it=False)
    
        st.markdown(download_button_str, unsafe_allow_html=True)

    else:
        fig_for, df_icu = plot_forecast('ml_for_icu_all_cantons', curve='ICU_patients',SEIR_preds = False)
        st.plotly_chart(fig_for, use_container_width=True)
        filename = 'forecast_ICU.csv'
        download_button_str = download_button(
            df_icu, filename, 'Download data', pickle_it=False)
    
        st.markdown(download_button_str, unsafe_allow_html=True)
        
        

    st.write('''
            ## Model Validation
             In the Figure below, the model's predictions are plotted against data, both *in sample* (for 
             the data range used for training) and *out of sample* (part of the series not used during model training)
             for the machine learning model.  

             ''')
    fig = plot_predictions('ml_val_hosp_all_cantons', curve='hosp')
    st.plotly_chart(fig, use_container_width=True)
    
    st.write('''
             Below, we have the same as above, but for the total hospitalizations.  

             ''')
    fig = plot_predictions('ml_val_total_all_cantons', curve='total_hosp')
    st.plotly_chart(fig, use_container_width=True)


    st.write('''
             Below, we have the same as above, but for the ICU occupancy.  

             ''')
    fig = plot_predictions('ml_val_icu_all_cantons', curve='ICU_patients')
    st.plotly_chart(fig, use_container_width=True)
    