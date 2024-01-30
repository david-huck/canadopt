import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from functools import partial

hoep = pd.read_csv("ontario_hoep_historic.csv").set_index("Year").values.reshape((-1,))
valid = hoep < 4  # exclude three values that suggest 1/x shape
hoep = hoep[valid]
ga = pd.read_csv("ontario_ga_historic.csv").set_index("Year").values.reshape((-1,))
ga = ga[valid]


def lin_func(x, m=1, b=0):
    return m * x + b


p, v = curve_fit(lin_func, hoep, ga)


def global_adjustment(hoep):
    """returns the ga as a function of the hoep

    Args:
        hoep (float): hourly ontario electricity price

    Returns:
        ga (float): _description_
    """
    return lin_func(hoep, m=p[0], b=p[1])


def plot_lin_fit():
    fig = px.scatter(x=hoep, y=ga)
    x = np.linspace(min(hoep), max(hoep), 30)
    fig.add_trace(go.Scatter(x=x, y=global_adjustment(x), showlegend=False))
    fig.update_layout(
        yaxis_title="Global adjustment ct/kWh",
        xaxis_title="HOEP ct/kWh",
        width=500,
        xaxis_range=(0, 4),
        yaxis_range=(0, 14),
    )
    return fig
