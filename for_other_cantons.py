#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 09:20:09 2022

@author: eduardoaraujo
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta 
import plotly.graph_objects as go
from forecast import download_button
from get_data import get_curve
from sqlalchemy import create_engine
from plots import scatter_plot_cases_hosp_all
engine = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/epigraphhub")

dict_cantons_names = {
    'Uri (UR)': 'UR',
    'Vaud (VD)': 'VD',
    'Obwalden (OW)': 'OW',
    'Aargau (AG)': 'AG',
    'Appenzell Innerrhoden (AI)': 'AI',
    'Valais (VS)': 'VS',
    # 'FL':'' ,
    'Sankt Gallen (SG)': 'SG',
    'Zug (ZG)': 'ZG',
    'Appenzell Ausserrhoden (AR)': 'AR',
    'Zürich (ZH)': 'ZH',
    'Bern (BE)': 'BE',
    'Geneva (GE)': 'GE',
    'Glarus (GL)': 'GL',
    'Graubünden (GR)': 'GR',
    'Basel Stadt (BS)': 'BS',
    'Jura (JU)': 'JU',
    'Schaffhausen (SH)': 'SH',
    'Freiburg (FR)': 'FR',
    'Basel Land (BL)': 'BL',
    'Schwyz (SZ)': 'SZ',
    'Solothurn (SO)': 'SO',
    'Thurgau (TG)': 'TG',
    'Ticino (TI)': 'TI',
    'Luzern (LU)': 'LU',
    'Neuchâtel (NE)': 'NE',
    'Nidwalden (NW)': 'NW',
}


def plot_cases_canton(full_name_canton, canton):
    ''''
    Function to plot the new cases according to FOPH in any canton

    params canton: canton to plot the data

    return[0] plotly figure
    return[1] last data of new cases reported

    '''

    df = get_curve('cases', canton)

    df.sort_index(inplace=True)
    
    last_date = df.index[-1]
    df = df['2021-11-01':]
    df = df.iloc[:-3]

    # computing the rolling average
    m_movel = df.rolling(7).mean().dropna()

    fig = go.Figure()

    title = f"{full_name_canton}"

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


def get_hospCapacity(canton):
    '''
    Returns Hospital capacity data for the specified `canton`
    :param canton: Two letter symbol for the canton 
    '''
    df = get_curve('hospcapacity', canton)
    df = df.resample('D').mean()
    df = df.sort_index() 
    df = df.iloc[:-3]
    df = df.fillna(0)
    return df.Total_Covid19Patients[-2:].astype('int'), df.TotalPercent_Covid19Patients[-2:]


def plot_hosp_canton(full_name_canton, canton):
    '''
    Function to plot the number of new hospitalizations for any canton

    params canton: canton to plot the hospitalization data
    returns plotly figure

    '''

    df = get_curve('hosp', canton)

    df.sort_index(inplace=True)
    df = df['2021-11-01':]
    
    df = df.iloc[:-3]
    
    # computing the rolling average
    m_movel = df.rolling(7).mean().dropna()

    fig = go.Figure()

    title = f"{full_name_canton}"

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


def plot_predictions_canton(table_name, curve, canton, full_name_canton, title=None):
    ''''
    Function to plot the predictions

    params table_name: Name of the table with the predictions (name used to save
                                                               the table in the database)

    params curve: Curve related with the predictions that will be plotted

    return plotly figure
    '''
    target_curve_name = curve

    df_val = pd.read_sql(
        f"select * from switzerland.{table_name} where canton='{canton}';", engine)

    df_val.index = pd.to_datetime(df_val.date)

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

        title = f"{full_name_canton}"

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


def plot_forecast_canton(table_name, canton, curve, full_name_canton, title=None):
    ''''
    Function to plot the forecast 

    params table_name: Name of the table with the predictions (name used to save
                                                      the table in the database)

    params curve: Curve related with the predictions that will be plotted

    return[0] plotly figure
    return[1] dataframe with the values forecasted 

    '''
    target_curve_name = curve

    df_for = pd.read_sql(
        f"select * from switzerland.{table_name} where canton='{canton}';", engine)
    

    df_for.index = pd.to_datetime(df_for.date)

    curves = {'hosp': 'hosp', 'ICU_patients': 'hospcapacity', 
              'total_hosp': 'hospcapacity'}
    ydata = get_curve(curves[curve], canton)
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

        title = f"{full_name_canton}"

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

    column_curves = {'hosp': 'entries', 'ICU_patients': 'ICU_Covid19Patients',
                     'total_hosp' : 'Total_Covid19Patients'}

    min_data = min(ydata.index[-1], df_for.index[0] - timedelta(days=1))


    fig.add_trace(go.Scatter(
        x=ydata.loc[:min_data].index[-150:], y=ydata.loc[:min_data][column_curves[curve]][-150:], name='Data', line=dict(color='black')))

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


def app():
    dict_cantons_names = {
        'Uri (UR)': 'UR',
        'Vaud (VD)': 'VD',
        'Obwalden (OW)': 'OW',
        'Aargau (AG)': 'AG',
        'Appenzell Innerrhoden (AI)': 'AI',
        'Valais (VS)': 'VS',
        # 'FL':'' ,
        'Sankt Gallen (SG)': 'SG',
        'Zug (ZG)': 'ZG',
        'Appenzell Ausserrhoden (AR)': 'AR',
        'Zürich (ZH)': 'ZH',
        'Bern (BE)': 'BE',
        'Geneva (GE)': 'GE',
        'Glarus (GL)': 'GL',
        'Graubünden (GR)': 'GR',
        'Basel Stadt (BS)': 'BS',
        'Jura (JU)': 'JU',
        'Schaffhausen (SH)': 'SH',
        'Freiburg (FR)': 'FR',
        'Basel Land (BL)': 'BL',
        'Schwyz (SZ)': 'SZ',
        'Solothurn (SO)': 'SO',
        'Thurgau (TG)': 'TG',
        'Ticino (TI)': 'TI',
        'Luzern (LU)': 'LU',
        'Neuchâtel (NE)': 'NE',
        'Nidwalden (NW)': 'NW',
    }

    list_cantons = list(dict_cantons_names.keys())
    list_cantons.sort()

    full_name_canton = st.sidebar.selectbox("For which canton you want to forecast?",
                                            list_cantons
                                            )

    canton = dict_cantons_names[full_name_canton]

    st.title('Number of cases and Hospitalizations')

    fig_c, last_date, last_cases = plot_cases_canton(full_name_canton, canton)
    total_hc, perc_hc = get_hospCapacity(canton)

    st.write(f'''
             The graphs below show the number of cases and hospitalizations in {full_name_canton}
             according to FOPH. The data was updated in: {str(last_date)[:10]}
             ''')

    fig_h, last_hosp = plot_hosp_canton(full_name_canton, canton)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Daily new cases", value=last_cases[-1],
                  delta=f"{last_cases.diff()[-1]} cases", delta_color="inverse")
        st.metric("Total COVID-19 Hospitalizations", value=total_hc[-1],
                  delta=f"{total_hc.diff()[-1]} cases", delta_color='inverse')
        st.plotly_chart(fig_c, use_container_width=True)
    with c2:
        st.metric("Daily new Hospitalizations", value=last_hosp[-1],
                  delta=f"{last_hosp.diff()[-1]:.1f} Hospitalizations",
                  delta_color="inverse")
        st.metric("Percent of total Hospitalizations", value=f"{perc_hc[-1]:.2f}",
                  delta=f"{perc_hc.diff()[-1]:.2f} %", delta_color="inverse")
        st.plotly_chart(fig_h, use_container_width=True)

    st.write('''
             #### Relation between cases and hospitalizations in Switzerland:
             If we look at hospitalization *vs* cases, we can hint at the change in case severity over time.
        ''')

    scatter_cases_hosp_all = scatter_plot_cases_hosp_all()

    st.image(scatter_cases_hosp_all)

    st.write('''
     ## 14-day Forecasts
    Below, we have the forecast for the next 14 days, for daily hospitalizations, total hospitalizations
    and total ICU hospitalizations. The 95% confidence bounds are also shown. The table with the forecasts can be downloaded by clicking on the button. 

             ''')

    fig_for, df_hosp = plot_forecast_canton(
        'ml_for_hosp_all_cantons', canton=canton, curve='hosp', full_name_canton=full_name_canton)
    st.plotly_chart(fig_for, use_container_width=True)
    filename = 'forecast_hosp.csv'
    download_button_str = download_button(
        df_hosp, filename, 'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)
    
    fig_total, df_total = plot_forecast_canton(
        'ml_for_total_all_cantons', canton=canton, curve='total_hosp', full_name_canton=full_name_canton)
    st.plotly_chart(fig_total, use_container_width=True)
    filename = 'forecast_total.csv'
    download_button_str = download_button(
        df_total, filename, 'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)

    fig_for_icu, df_icu = plot_forecast_canton(
        'ml_for_icu_all_cantons', canton=canton, curve='ICU_patients', full_name_canton=full_name_canton)
    st.plotly_chart(fig_for_icu, use_container_width=True)
    filename = 'forecast_icu.csv'
    download_button_str = download_button(
        df_icu, filename, 'Download data', pickle_it=False)

    st.markdown(download_button_str, unsafe_allow_html=True)

    st.write('''
            ## Model Validation
             In the Figure below, the model's predictions are plotted against data, both in sample (for 
             the data range used for training) and out of  sample (part of the series not used during model training).  

             ''')
    fig_val = plot_predictions_canton(
        'ml_val_hosp_all_cantons', curve='hosp', canton=canton, full_name_canton=full_name_canton)

    st.plotly_chart(fig_val, use_container_width=True)
    
    st.write('''
             Below, we have the same as above, but for the total hospitalizations.  

             ''')

    fig_val_total = plot_predictions_canton(
        'ml_val_total_all_cantons', curve='total_hosp', canton=canton, full_name_canton=full_name_canton)

    st.plotly_chart(fig_val_total, use_container_width=True)

    st.write('''
             Below, we have the same as above, but for the ICU occupancy.  

             ''')

    fig_val_icu = plot_predictions_canton(
        'ml_val_icu_all_cantons', curve='ICU_patients', canton=canton, full_name_canton=full_name_canton)

    st.plotly_chart(fig_val_icu, use_container_width=True)
