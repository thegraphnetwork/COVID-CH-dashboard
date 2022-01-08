import seaborn as sns
from get_data import get_canton_data, get_updated_data, get_curve
from io import BytesIO
import matplotlib.pyplot as plt

def scatter_plot_cases_hosp(region):

    if region == 'All':

        cases = get_curve('cases')

        cases = cases.resample('D').sum()

        hosp = get_curve('hosp')

        hosp = hosp.resample('D').sum()

        fig, ax = plt.subplots()

        sns.regplot(x=cases[:'2020-12-31'].entries,
                    y=hosp[:'2020-12-31'].entries,
                    label='2020', color='tab:gray', robust=True, ax=ax)

        sns.regplot(x=cases['2021-01-01':'2021-03-31'].entries,
                    y=hosp['2021-01-01':'2021-03-31'].entries,
                    label='2021-Q1', color='tab:cyan',robust=True, ax=ax)

        sns.regplot(x=cases['2021-04-01':'2021-06-30'].entries,
                    y=hosp['2021-04-01':'2021-06-30'].entries,
                    label='2021-Q2', color='tab:orange',robust=True, ax=ax)

        sns.regplot(x=cases['2021-07-01':'2021-09-30'].entries,
                    y=hosp['2021-07-01':'2021-09-30'].entries,
                    label='2021-Q3', color='tab:red',robust=True, ax=ax)

        sns.regplot(x=cases['2021-10-01':'2021-12-31'].entries,
                    y=hosp['2021-10-01':'2021-12-31'].entries,
                    label='2021-Q4', color='tab:blue',robust=True, ax=ax)

        sns.regplot(x=cases['2022-01-01':'2022-03-31'].entries,
                    y=hosp['2022-01-01':'2022-03-31'].entries,
                    label='2022-Q1', color='tab:green', ax=ax)

        ax.set_title('Cases vs hospitalizations in Switzerland')

        ax.set_xlabel('Cases')
        ax.set_ylabel('Hospitalizations')
        ax.legend()
        ax.grid()

        buf = BytesIO()
        fig.savefig(buf, format="png")

    if region == 'GE':
        fig, ax = plt.subplots()

        cases = get_canton_data('cases', ['GE'])

        hosp = get_canton_data('hosp', ['GE'])

        sns.regplot(x=cases['2021-10-01':'2021-12-31'].entries,
                    y=hosp['2021-10-01':'2021-12-31'].entries,
                    label='2021-Q4', color='tab:blue',robust=True, ax=ax)

        sns.regplot(x=cases['2022-01-01':'2022-03-31'].entries,
                    y=hosp['2022-01-01':'2022-03-31'].entries,
                    label='2022-Q1', color='tab:green',robust=True, ax=ax)

        ax.set_title('Cases vs hospitalizations in Geneva')

        ax.set_xlabel('Cases')
        ax.set_ylabel('Hospitalizations')

        ax.grid()
        ax.legend()

        buf = BytesIO()
        fig.savefig(buf, format="png")

    return buf