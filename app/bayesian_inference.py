#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 22:31:57 2021

@author: eduardoaraujo
"""
#import arviz as az

import plotly.graph_objects as go
import pandas as pd 
import streamlit as st 
from PIL import Image
from get_data import get_canton_data 
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv('../.env')

engine = create_engine(f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')




def make_plots():
    
    Prev_post = pd.read_sql_table('prev_post', engine, schema = 'switzerland',index_col = 'datum')
    
    Phosp_post = pd.read_sql_table('phosp_post', engine, schema = 'switzerland',index_col = 'datum')
    
    Tests21GE = get_canton_data('test', ['GE'], '2021-01-01')
    Positivity21GE = Tests21GE.entries_pos/Tests21GE.entries 
    
    
    
    fig = go.Figure()


    fig.add_trace(go.Scatter(x=Positivity21GE.index, y=Positivity21GE.values, name='Test positivity', mode='markers'))
    
    fig.add_trace(go.Scatter(x = Prev_post.index, y = Prev_post['median'], name = 'Median',line=dict(color = '#FF7F0E')))
    
    fig.add_trace(go.Scatter(x = Prev_post.index, y = Prev_post['lower'], line=dict(color = '#FF7F0E',width=0), showlegend=False))
        
    fig.add_trace(go.Scatter(x = Prev_post.index, y = Prev_post['upper'],line=dict(color = '#FF7F0E', width=0),
            mode='lines',
            fillcolor='rgba(255, 127, 14, 0.4)', fill = 'tonexty', showlegend= False))
        
    fig.update_layout(title="Estimated prevalence of COVID in the population of Geneva",
                      yaxis_title = "Prevalence of infected",
                      xaxis_title = "Time (days)"
                     )
    
    
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(x = Phosp_post.index, y = Phosp_post['median'], name = 'Median',line=dict(color = '#FF7F0E')))
        
    fig2.add_trace(go.Scatter(x = Phosp_post.index, y = Phosp_post['lower'], line=dict(color = '#FF7F0E',width=0), showlegend=False))
            
    fig2.add_trace(go.Scatter(x = Phosp_post.index, y = Phosp_post['upper'],line=dict(color = '#FF7F0E', width=0),
                mode='lines',
                fillcolor='rgba(255, 127, 14, 0.3)', fill = 'tonexty', showlegend= False))
            
            
    # fig.add_scatter(x=Positivity21GE.index, y=Cases21GE.values, name='Test positivity', mode='markers')
    fig2.update_layout(title="Estimated Probability of Hospitalization in Geneva",
                      yaxis_title = "probability",
                     )

    
    return fig, fig2 


def app():

    st.title('Bayesian Inference')
    
    st.write('''
             If we think about Epidemics as stochastic processes, it may be worth to 
             try to infer some basic rates of these processes from the data. \n
             We start by restricting our observations to 2021, to avoid potentially 
             different dynamics in the beginning of the pandemic.

''')

    st.header('Cases and Hospitalizations as Binomial processes with variable rates')
    
    st.write(r'''
             If we treat cases and hospitalizations as binomial process, we can estimate 
             their variable rates. From the Test series, $T_t$ we can model the prevalence
             series in the population, 
             $p_{v_t} \thicksim Beta(\alpha, \beta)$ as 
             ''')
    st.latex(r'Cases_{t} \thicksim Bin(n = T_t, p = P_{v_t}).')
    
    st.write(r'''
         
             In similar fashion, we can model the probability of Hospitalization, 
             $P_{h_t} \thicksim Beta(\alpha, \beta)$ as ''')
             
    st.latex(r'{Hospitalizations}_{t} \thicksim Bin(n = Cases, p = P_{h_t}).')
    
    st.write('''
             
             The diagram of our inference looks like the figure below: 
                 
             ''')
    image = Image.open('diag_bin_model.png')
            
    st.image(image, caption= 'Binomial Model' )
    
    st.write('''
             Wich after our estimation yields the following posterior probability 
             distribution for the prevalence over time: 
        
             ''')
             
    fig,fig2 = make_plots()
    
    st.plotly_chart(fig, use_container_width = True)
             
             
    st.write('''
             The posterior distribution of the probability of Hospitalization for 
             positive cases is shown in the figure below:
                 
            ''')
    st.plotly_chart(fig2, use_container_width = True)
    