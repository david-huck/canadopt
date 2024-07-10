import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from typing import Iterable

hoep = pd.read_csv("ontario_hoep_historic.csv").set_index("Year").values.reshape((-1,))
ga = pd.read_csv("ontario_ga_historic.csv").set_index("Year").values.reshape((-1,))

exclude_outliers = True
if exclude_outliers:
    valid = hoep < 4  # exclude three values that suggest 1/x shape
    hoep = hoep[valid]
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
    ga = lin_func(hoep, m=p[0], b=p[1])
    if isinstance(ga, Iterable):
        ga[ga < 0] = 0
    elif ga < 0:
        ga = 0
    return ga


def plot_lin_fit():
    fig = px.scatter(x=hoep, y=ga)
    x = np.linspace(min(hoep), max(hoep), 30)
    fig.add_trace(go.Scatter(x=x, y=global_adjustment(x), showlegend=False))
    fig.update_layout(
        yaxis_title="Global adjustment (ct/kWh)",
        xaxis_title="HOEP (ct/kWh)",
        width=500,
        xaxis_range=(0, 4),
        yaxis_range=(0, 14),
    )
    residuals = ga - lin_func(hoep, *p)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ga - np.mean(ga)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    fig.add_annotation(
        x=1.372,
        y=10.2747,
        xref="x",
        yref="y",
        text=f"R²={r_squared:.2f}",
        showarrow=True,
        font=dict(family="Courier New, monospace", size=16, color="#ffffff"),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=-40,
        ay=40,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8,
    )
    return fig
