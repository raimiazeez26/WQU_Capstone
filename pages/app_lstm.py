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
import dash_ag_grid as dag
from pages.side_bar import sidebar
from utils import fetch_data, add_indicators, plot_subplots, apply_directional_filter
# from pages.app_backtest import symbol_dropdown, timeframe_dropdown, filters_dropdown, run_button
from pages.lstm import run_lstm

dash.register_page(__name__, name='PREDICTION MODEL')

##=========================================================================================================
# Inputs parameters
##=========================================================================================================
TIMEFRAMES = ['1d']
SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDJPY', 'USDCHF', 'NZDUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY']
FILTERS = ['HYBRID', 'FILTERS2']

symbol_dropdown = html.Div([
    html.P('Symbol:'),
    dcc.Dropdown(
        id='symbol-dropdown-lstm',
        options=[{'label': symbol, 'value': symbol} for symbol in SYMBOLS],
        value='EURUSD',
    )
])

timeframe_dropdown = html.Div([
    html.P('Timeframe:'),
    dcc.Dropdown(
        id='timeframe-dropdown-lstm',
        options=[{'label': timeframe, 'value': timeframe} for timeframe in TIMEFRAMES],
        value='1d',
    )
])

filters_dropdown = html.Div([
    html.P('Filters:'),
    dcc.Dropdown(
        id='filters-dropdown-lstm',
        options=[{'label': filter, 'value': filter} for filter in FILTERS],
        value='HYBRID',
    )
])

run_button = html.Div(
    [
        dbc.Button(
            "Train Model", id="run-button-lstm", className="me-2", n_clicks=0
        )
    ]
)


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

                dbc.Container([

                    # inputs
                    dbc.Row([
                        dbc.Col([
                            symbol_dropdown,
                        ], xs=2, sm=2, md=2, lg=2, xl=2, className='p-3'),
                        dbc.Col([
                            timeframe_dropdown,
                        ], xs=2, sm=2, md=2, lg=2, xl=2, className='p-3'),

                        dbc.Col([
                            filters_dropdown,
                        ], xs=2, sm=2, md=2, lg=2, xl=2, className='p-3'),

                        dbc.Col([
                            run_button,
                        ], xs=2, sm=2, md=2, lg=2, xl=2, className='p-3'),

                    ], className='align-items-end', justify='center'),

                    html.Br(),
                    html.Hr(),

                    dbc.Row([
                        html.Br(),
                        html.Hr(),
                        dls.Dot(
                            html.Div(id='training-result'),
                            color="#0f62fe",  # Customize color of the spinner
                        ),

                        html.Br(),
                        html.Hr(),
                        html.Div(id='transactions-result'),

                        html.Br(),
                        html.Hr(),
                        html.Div(id='training-result2'),

                        html.Br(),
                        html.Hr(),

                        html.Div(id='transactions-result2'),

                    ]),
                ])
            ])

            ])
    ], fluid=True, class_name='g-0', )


@callback(
    Output("training-result", "children"),
    Output("transactions-result", "children"),
    Output("training-result2", "children"),
    Output("transactions-result2", "children"),
    Input("run-button-lstm", "n_clicks"),
    Input('symbol-dropdown-lstm', 'value'),
    Input('timeframe-dropdown-lstm', 'value'),

    prevent_initial_call=True,
)
def train_predict(n_click, ticker, timeframe):
    if "run-button-lstm" == ctx.triggered_id:
        print('Button Clicked')
        figs, transactions = run_lstm(ticker, timeframe)

        transaction_table1 = [
            html.H4(f"Transactions for {ticker}: Hybrid: True", className="text-center mb-4"),
            dag.AgGrid(
                className="ag-theme-alpine-dark header-style-on-filter",
                id="data-grid",
                columnSize="autoSize",
                columnDefs=[{"field": i, "resizable": True} for i in transactions[0].columns],
                rowData=transactions[0].to_dict("records"),
                defaultColDef={
                    "filter": "agTextColumnFilter",
                    "cellStyle": {"textAlign": "center"}
                },
                dashGridOptions={"animateRows": True},
            )
        ]

        transaction_table2 = [
            html.H4(f"Transactions for {ticker}: Hybrid: True", className="text-center mb-4"),
            dag.AgGrid(
                className="ag-theme-alpine-dark header-style-on-filter",
                id="data-grid2",
                columnSize="autoSize",
                columnDefs=[{"field": i, "resizable": True} for i in transactions[1].columns],
                rowData=transactions[1].to_dict("records"),
                defaultColDef={
                    "filter": "agTextColumnFilter",
                    "cellStyle": {"textAlign": "center"}
                },
                dashGridOptions={"animateRows": True},

            )
        ]

        return [dcc.Graph(figure=figs[0]), transaction_table1, dcc.Graph(figure=figs[1]), transaction_table2]

    return [None, None, None]