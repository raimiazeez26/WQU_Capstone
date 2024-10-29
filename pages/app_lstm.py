import datetime, time
import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, ctx, callback
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import ThemeSwitchAIO, ThemeChangerAIO, template_from_url
from dash.exceptions import PreventUpdate

from .side_bar import sidebar
from utils import fetch_data, add_indicators, plot_subplots, apply_directional_filter

dash.register_page(__name__, name='PREDICTION MODEL')


def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                [
                    sidebar()
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2,
            ),

            dbc.Col([
                html.P(
                    'Still in development...  Be patient!'
                    , style={'textAlign': 'left'}),
            ])

            ])
    ], fluid=True, class_name='g-0', )