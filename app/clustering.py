#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:58:03 2021

@author: eduardoaraujo
"""

import streamlit as st 
from get_data import compute_clusters
from geoanalysis import correlation_map, correlation_map_bokeh


def app():
    
    st.header('Clustering the Cantons of Switzerland')
    
    st.write('We use the function below to compute the cross-correlation between the cantons cases time-series:')
    
    st.latex(r''' 
             \rho_{XY}(\tau)={\frac {\operatorname{E} \left[\left(X_{t}-\mu _{X}\right){\overline {\left(Y_{t+\tau }-\mu _{Y}\right)}}\right]}{\sigma _{X}\sigma _{Y}}}.
            ''') 
    
    st.write(r'''
    The sign of $\tau$ that maximizes the cross-correlation function, is a
    proxy of the direction of predictability, i.e., if $\rho_{XY}(\tau>0)$
    it means that canton $X$ anticipates $Y$ in incidence trends, and
    can thus be a good predictor for the $Y$.
    ''')
    
    st.write('''
             Using this equation, we created a cross-correlation matrix between the 
             cantons and used it as the basis of a hierarchical clustering algorithm. 
             The result of the clustering is shown in the dendrogram below:
             ''')

    clusters, all_reg, fig = compute_clusters('cases', t=0.8, plot=True)
    
    st.pyplot(fig) 
    st.write("""
    ## Correlation Map
    In the map below each canton is colored according to its correlation to the Canton of Geneva.
    """)
    curve = st.selectbox("On which series you want to base the correlation on?",
                ('cases', 'hospitalizations') 
    )
    #figmap = correlation_map(curve=curve)
    #st.pyplot(figmap)
    
    figmap_bokeh = correlation_map_bokeh(curve=curve)
    st.bokeh_chart(figmap_bokeh)