#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 23:36:59 2021

@author: eduardoaraujo
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd 
import pymc3 as pm
#import os
import streamlit as st 
import arviz as az
from PIL import Image
import matplotlib.pyplot as plt 
from get_data import get_canton_data 
from sqlalchemy import create_engine
import config as conf
engine = create_engine(f'postgresql://{conf.POSTGRES_USER}:{conf.POSTGRES_PASSWORD}@{conf.POSTGRES_HOST}:{conf.POSTGRES_PORT}/{conf.POSTGRES_DB}')


def make_inference_sir():
    Prev_post = pd.read_sql_table('prev_post', engine, schema = 'switzerland', index_col = 'datum')
    
    
    prev = Prev_post.loc['2021-10-13':]['median']
    
    RE21GE = get_canton_data('re', ['GE'] , '2021-01-01')
    
    rtge = RE21GE.median_R_mean.loc['2021-10-14':]
    rtge_short = rtge.loc[:'2021-11-22'] # Last days are NaNs
    prev_short = prev.loc[:'2021-11-22']
    rtge_filled = rtge.fillna(method='ffill')[:-1]
    # nnan = len(rtge)-len(rtge.dropna())
    # rtge_filled.iloc[-nnan:] = np.linspace(1.3,2,nnan)
    with pm.Model() as model_gam:
        sig = pm.HalfCauchy('σ',0.05)
        gam = pm.Uniform("γ", 0.01, 0.3)
        i0 = pm.Beta("I(0)", mu=0.001, sigma=.01)
    #     rtplus = pm.Uniform("Rt_plus",0.0,2.0, shape=len(rtge_filled))
        rtplus = pm.HalfCauchy("Rt_plus",0.3, shape=len(rtge_filled))
        t=np.arange(len(rtge_filled))
        rhs = pm.Deterministic("Prev", i0*pm.math.exp((rtge_filled.values+rtplus-1)*gam*t))
        I = pm.Beta("I(t)", mu = rhs, sigma=sig, observed=prev.values)
        
    with model_gam:
        tracegam = pm.sample(5000,tune=3500,start={'I(0)':0.01,'σ':.05,"γ":0.02},init='auto', return_inferencedata=True)
        
    fig_mat, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,3))
    az.plot_posterior(tracegam, var_names=['γ','I(0)', 'σ'], ax = [ax1,ax2,ax3])
    
    inc_post = pd.DataFrame(index=rtge_filled.index, 
                         data={'median': tracegam.posterior.Prev.median(axis=(0,1)),
           'lower': np.percentile(tracegam.posterior.Prev, 2.5, axis=(0,1)),
           'upper': np.percentile(tracegam.posterior.Prev,97.5, axis=(0,1)),
          })

    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(x=rtge_filled.index, y=inc_post['median'], name = 'Median',line=dict(color = '#FF7F0E')))
    
    fig1.add_trace(go.Scatter(x = rtge_filled.index, y = inc_post['lower'], line=dict(color = '#FF7F0E',width=0), showlegend=False))
            
    fig1.add_trace(go.Scatter(x = rtge_filled.index, y = inc_post['upper'],line=dict(color = '#FF7F0E', width=0),
                mode='lines',
                fillcolor='rgba(255, 127, 14, 0.3)', fill = 'tonexty', name = '95% CI'))
            
    fig1.add_trace(go.Scatter(x=rtge_filled.index, y=prev.values, name='Test positivity', mode='markers', line = dict(color = '#636EFA') ))   
    
    fig1.update_layout(title="Prevalence estimates from the SIR model", yaxis_title="Prevalence")
        
    rtplus_post = pd.DataFrame(index=rtge_filled.index, 
                         data={'median': tracegam.posterior.Rt_plus.median(axis=(0,1)),
           'lower': np.percentile(tracegam.posterior.Rt_plus, 2.5, axis=(0,1)),
           'upper': np.percentile(tracegam.posterior.Rt_plus,97.5, axis=(0,1)),
          })
    
    fig2 = go.Figure() 

    fig2.add_trace(go.Scatter(x=rtge_filled.index, y=rtplus_post['median'], name = 'Median',line=dict(color = '#FF7F0E')))
        
    fig2.add_trace(go.Scatter(x = rtge_filled.index, y = rtplus_post['lower'], line=dict(color = '#FF7F0E',width=0), showlegend = False))
                
    fig2.add_trace(go.Scatter(x = rtge_filled.index, y = rtplus_post['upper'],line=dict(color = '#FF7F0E', width=0),
                    mode='lines',
                    fillcolor='rgba(255, 127, 14, 0.3)', fill = 'tonexty',name = '95% CI'))
                
    
    fig2.update_layout(title=r"$Additional\, R_e$", yaxis_title=r"$Additional\, R_e$")

    fig3 = go.Figure() 
    fig3.add_scatter(x=rtge_filled.index, y=rtge_filled.values,name=r'$FOPH\, R_e$', stackgroup='one')
    fig3.add_scatter(x=rtge_filled.index, y=rtplus_post['median'].values, name=r'$Adjusted\, R_e$', 
                    stackgroup='one', hoverinfo='x+y')
    fig3.update_layout(yaxis_title= r'$R_e$', xaxis_title="Date")
        
    return fig_mat, fig1, fig2, fig3 



def app():
    
    fig_mat, fig1, fig2, fig3 = make_inference_sir()
    
    st.title('SIR-based Forecast')
    
    st.write(r'''
             The FOPH 
             makes available the daily estimates of the effective reproductive number, 
             $R_t$. With a good estimate of $R_t$ it is possible to simulate growth based 
             on transmission models. Since for the SIR model,
             ''')
             
    st.latex(r'R_t = R_0 S(t) = \cfrac{\beta}{\gamma} S(t),')
    
    st.write('''
             we can use it to parameterize a simple transmission model with which to 
             forecast cases and thus hospitalizations. We would normally write the SIR 
             model as:
             
             ''')
             
    st.latex(r' \dot{S} = -\beta S(t) I(t) \tag{1}')
    
    st.latex(r'\dot{I} = \beta S(t) I(t) - \gamma I(t) \tag{2}')
    
    st.write(r'''
             From the $R_t$ time series, we can derive a time-dependent transmission 
             parameter, 
             
             ''')
             
    st.latex(r'\beta(t) = \cfrac{R_t \gamma}{S(t)}')
    
    st.write('Then we can re-write the SIR model as')
    
    st.latex(r' \dot{S} = -R_t \gamma I(t) \tag{3}')
    
    st.latex(r'\dot{I} = R_t \gamma I(t) - \gamma I(t) \tag{4}')
    
    st.write('''
             We can reduce the system above to just equation (4), which has the 
             following solution:
             ''')
             
    st.latex(r'I(t) = I(0)e^{(R_t - 1)\gamma t} \tag{5}')
    
    st.write(r'''
             The prevalence in the population that we estimated before, in the SIR 
             model, is repre- sented by $I(t)$. We can see that its evolution is 
             dependent of the effective reproductive number($R_t$). O
             ne limitation of using the $R_t$ estimated series made available by FOPH, 
             is that it stops about two weeks before the present day. Therefore we 
             have modified the model from equation (5) to include a correction 
             term we can estimate from data. Then it becomes
             
             ''')
             
    st.latex(r'I(t) = I(0) e^{[(R_t + R_t^+) - 1]\gamma t} \tag{6}')
    
    st.write(r'''
             We the attribute prior probability distributions to the parameters of 
             this model, so we can estimate the posterior distribution for $I(t)$. 
             Since $I(t)$ varies between 0 and 1, we can treat it as a 
             Beta random variable.
             ''')
             
             
    st.latex(r'I(t) \thicksim Beta(\alpha_t, \beta_t).')
    
    st.write(r'''
             The correction term $R_t^+$ takes on a HalfCauchy(0.3) prior distribution. 
             Additionaly, we give $\gamma$ an uniform prior, $U(0.01, 0.3)$, and model 
             the initial fraction of infected as $I(0) \thicksim Beta().$
             
             We run the Bayesian inference on the last wave starting on October 15th.
             To facilitate the fit of the data to the model let’s also use the 7-day 
             moving average of it.
             The resulting structure of the probabilistic model is depicted below.
             ''')
             
             
    image = Image.open('diag_sir_model_inference.png')
                     
    st.image(image, caption= 'Diagram of the SIR model-based inference' )
    
    st.write('''
             After we run the inference, we obtain posterior distributions for a
             all parameters in the diagram above which are shown below.
             
             ''')
             
    st.pyplot(fig_mat )
    
    
    st.write(r'''
             We should point out that our posterior estimate for the $\gamma$ parameter, 
             which is the recovery rate of infected individuals, corresponds to a 
             recovery period of about a month ($1/ \gamma$). 
             
             We also get a posterior distribution for the prevalence curve $(I(t))$ 
             shown on the graph below. With the correction term for the reproductive 
             number being estimated from data, the posterior $I(t)$ curve matches 
             quite well the prevalence curve, as expected.
             

             ''')
             
    st.plotly_chart(fig1, use_container_width = True)
    st.plotly_chart(fig2, use_container_width = True)
    st.plotly_chart(fig3, use_container_width = True)
    
    # lower, median, upper 
    

    
    