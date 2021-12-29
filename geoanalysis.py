from matplotlib.pyplot import get
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from get_data import get_ch_map, lag_ccf, engine_public

def correlation_map(curve='cases'):
    if curve == 'hospitalizations':
        curve = 'hosp'
    chmap = get_ch_map()
    
    df = pd.read_sql_table(f'foph_{curve}', engine_public, schema = 'switzerland', columns = ['datum','geoRegion',  'entries'])
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






