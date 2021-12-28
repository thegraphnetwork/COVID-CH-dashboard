#!/usr/bin/env python3
"""
Created on Sat Dec 18 20:40:33 2021

@author: eduardoaraujo
"""

import streamlit as st
import bayesian_inference
import SIR_forecast
import forecast 
import home
import clustering

st.set_page_config(page_title='COVID19 Hospitalization Forecasts', 
                    page_icon=":chart:", 
                    layout="wide", 
                    initial_sidebar_state="auto", 
                    menu_items={
                                'Get Help': 'https://www.thegraphnetwork.org',
                                'Report a bug': "https://github.com/thegraphnetwork/COVID-CH-dashboard/issues",
     })

st.sidebar.image('tgn.png')


PAGES = {"Home": forecast, "Cluster analysis":clustering, "Bayesian Inference": bayesian_inference, 
         "SIR-based Forecasting": SIR_forecast, "credits": home}

select_page = st.sidebar.selectbox(
    "Select the analysis:",
    (list(PAGES.keys()))
)

page = PAGES[select_page]

page.app()
