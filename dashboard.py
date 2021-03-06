#!/usr/bin/env python3
"""
Created on Sat Dec 18 20:40:33 2021

@author: eduardoaraujo
"""

import streamlit as st
import bayesian_inference
import SIR_forecast
import forecast
import for_other_cantons 
import credits
import clustering

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='Switzerland COVID19 Hospitalization Forecasts', 
                    page_icon=":chart:", 
                    layout="wide", 
                    initial_sidebar_state="auto", 
                    menu_items={
                                'Get Help': 'https://www.thegraphnetwork.org',
                                'Report a bug': "https://github.com/thegraphnetwork/COVID-CH-dashboard/issues",
     })

st.sidebar.image('tgn.png')


PAGES = {"Geneva": forecast, "Cluster analysis":clustering, 
         'Other cantons': for_other_cantons,
        # "Bayesian Inference": bayesian_inference, 
        #  "Rt estimation": SIR_forecast, 
         "credits": credits}

select_page = st.sidebar.selectbox(
    "Select the analysis:",
    (list(PAGES.keys()))
)

page = PAGES[select_page]

page.app()
