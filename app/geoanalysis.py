import pandas as pd
import matplotlib.pyplot as plt
from bokeh.models import (ColorBar,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper)
from bokeh.palettes import brewer
from bokeh.plotting import figure

from get_data import get_ch_map, lag_ccf, engine

def correlation_map(curve='cases'):
    if curve == 'hospitalizations':
        curve = 'hosp'
    chmap = get_ch_map()
    
    df = pd.read_sql_table(f'foph_{curve}', engine, schema = 'switzerland', columns = ['datum','geoRegion',  'entries'])
    df.index = pd.to_datetime(df.datum)
    inc_canton = df.pivot(columns='geoRegion', values='entries')
    del inc_canton['CHFL']
    del inc_canton['CH']
    cm, lm = lag_ccf(inc_canton.values)
    corrsGE = pd.DataFrame(index= inc_canton.columns, data={'corr':cm[8,:]})
    chmap_corr = pd.merge(left=chmap, right=corrsGE, on='geoRegion')
    fig, ax = plt.subplots(1, 1)

    chmap_corr.plot(column='corr', ax=ax,
        legend=True,
        legend_kwds={'label': "Correlation",
                        'orientation': "horizontal"}
    )
    ax.set_axis_off();
    return fig


def correlation_map_bokeh(curve='cases'):
    if curve == 'hospitalizations':
        curve = 'hosp'
    chmap = get_ch_map()
    
    df = pd.read_sql_table(f'foph_{curve}', engine, schema = 'switzerland', columns = ['datum','geoRegion',  'entries'])
    df.index = pd.to_datetime(df.datum)
    inc_canton = df.pivot(columns='geoRegion', values='entries')
    del inc_canton['CHFL']
    del inc_canton['CH']
    cm, lm = lag_ccf(inc_canton.values)
    corrsGE = pd.DataFrame(index= inc_canton.columns, data={'corr':cm[8,:]})
    chmap_corr = pd.merge(left=chmap, right=corrsGE, on='geoRegion')
    
    # Input GeoJSON source that contains features for plotting
    geosource = GeoJSONDataSource(geojson = chmap_corr.to_json())

    # Define custom tick labels for color bar.
    palette = brewer['BuGn'][8]
    palette = palette[::-1] # reverse order of colors so higher values have darker colors
    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    color_mapper = LinearColorMapper(palette = palette, low = 0.6, high = 1.0)
    # Define custom tick labels for color bar.
    tick_labels = { '0.7': '0.7', '0.75': '0.75', '0.8': '0.8', '0.85': '0.85', '0.9': '0.9', '0.95': '0.95', '1.00': '1.00'}
    # Create color bar.
    color_bar = ColorBar(color_mapper = color_mapper, 
                        label_standoff = 8,
                        width = 500, height = 20,
                        border_line_color = None,
                        location = (0,0), 
                        orientation = 'horizontal',
                        major_label_overrides = tick_labels,
                        title = 'Correlation')
    # Create figure object.
    p = figure(title = '', 
            plot_height = 600, plot_width = 950, 
            toolbar_location = 'below',
            tools = "pan, wheel_zoom, box_zoom, reset")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    # Add patch renderer to figure.
    states = p.patches('xs','ys', source = geosource,
                    fill_color = { 'field' :'corr',
                                    'transform' : color_mapper},
                    line_color = 'black', 
                    line_width = 0.8, 
                    fill_alpha = 1)
    # Create hover tool
    p.add_tools(HoverTool(renderers = [states],
                        tooltips = [('Cantons','@geoRegion'),
                                ('Correlation', '@corr')]))
    # Specify layout
    p.add_layout(color_bar, 'below')

    p.xaxis.visible = False

    p.yaxis.visible = False

    return p




