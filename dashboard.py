#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


PAGES = {"Home": forecast, "Cluster analysis":clustering, "Bayesian Inference": bayesian_inference, 
         "SIR-based Forecasting": SIR_forecast, "credits": home}

select_page = st.sidebar.selectbox(
    "Select the analysis:",
    (list(PAGES.keys()))
)

page = PAGES[select_page]

page.app()


# Hospitalizations vs cases devo fazer por trimestre? 



# Bayesian Inference 