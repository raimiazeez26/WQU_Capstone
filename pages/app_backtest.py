import datetime, time
import os
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
from utils import fetch_data, add_indicators, plot_subplots, apply_directional_filter, Strategy
import quantstats_lumi as qs

# Initialize QuantStats
qs.extend_pandas()

dash.register_page(__name__, name='FILTERS BACKTEST')

# creates the Dash App
# app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

##=========================================================================================================
# Inputs parameters
##=========================================================================================================
TIMEFRAMES = ['1d']
SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDJPY', 'USDCHF', 'NZDUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY']
FILTERS = ['HYBRID', 'FILTERS2']

symbol_dropdown = html.Div([
    html.P('Symbol:'),
    dcc.Dropdown(
        id='symbol-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in SYMBOLS],
        value='EURUSD',
    )
])

timeframe_dropdown = html.Div([
    html.P('Timeframe:'),
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[{'label': timeframe, 'value': timeframe} for timeframe in TIMEFRAMES],
        value='1d',
    )
])

filters_dropdown = html.Div([
    html.P('Filters:'),
    dcc.Dropdown(
        id='filters-dropdown',
        options=[{'label': filter, 'value': filter} for filter in FILTERS],
        value='HYBRID',
    )
])

run_button = html.Div(
    [
        dbc.Button(
            "Run Backtest", id="run-button", className="me-2", n_clicks=0
        )
    ]
)

# File path
file_path = "/assets/StrategyBacktest.html"
##=========================================================================================================


def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                [
                    sidebar()
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2,
            ),

            dbc.Col(
                [
                    html.H3('SIGNAL FILTERS BACKTEST', style={'textAlign': 'center'},
                            className='p4'),
                    html.Br(),
                    html.Hr(),

                    html.P(
                        'Add description here.. '
                        , style={'textAlign': 'left'}),

                    dcc.Markdown('''
                            * More information on Filters...
                            * More information on Filters...
                            * More information on Filters...
                            ''', style={'textAlign': 'left'}),

                    html.P('Instruction to run the backtest'
                           '** Click the Run Backtest button to start.**'
                           , style={'textAlign': 'left'}),
                    html.Hr(),

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

                        # tabs
                        dbc.Row([
                            dls.Dot(
                                # div for signal plot
                                html.Div(id='signal-plot'),
                                color="#0f62fe",  # Customize color of the spinner
                            ),

                            html.Br(),
                            html.Hr(),
                            html.Div(id='download-report'),
                            html.Div(id='returns-plot'),

                            html.Div([
                                html.Div(id="download-link-container"),
                            ])
                        ]),

                    ])

                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10,
                className='p-3'
            )
        ]
        )], fluid=True, class_name='g-0', )  # p-4
    # style={"height": "99vh", 'background-size': '100%'})


@callback(
    Output("signal-plot", "children"),
    Output("download-report", "children"),
    Output("returns-plot", "children"),
    Input('symbol-dropdown', 'value'),
    Input('timeframe-dropdown', 'value'),
    Input('filters-dropdown', 'value'),
    Input('run-button', 'n_clicks'),
    prevent_initial_call=True)
def plot_signals(ticker, timeframe, filter, n_clicks):
    if "run-button" == ctx.triggered_id:

        price_daily, vix_daily, usdx_daily, macro_data, sp_daily = fetch_data(ticker)
        print(f'price_daily shape; {price_daily.shape}')

        data = add_indicators(price_daily)
        print(f'price_daily shape; {price_daily.shape}')

        data_filters = apply_directional_filter(price_daily, vix_daily, usdx_daily, macro_data, sp_daily)

        filters_dict = {
            'HYBRID': 'Hybrid_Signal',
            'FILTERS1': 'Signal_1',
            'FILTERS2': 'Signal_2',
            'FILTERS3': 'Signal_3',
        }

        # Run the strategy using EUR/USD data
        strategy = Strategy(data_filters, filters_dict[filter])
        result = strategy.run()

        # Convert index to datetime
        result = result.set_index('open_datetime')
        result.index = pd.to_datetime(result.index)

        # generate report tearsheet with SPY BENCHMARK
        qs.reports.html(result['cumulative_returns'], title=f'{ticker} Strategy backtest',
                        download_filename=f'{ticker}_StrategyBacktest.html', output=f'assets/StrategyBacktest.html')

        chart = plot_subplots(data_filters, filters_dict[filter], ticker)
        # button = html.Button("Download Backtest Report", id="btn-dwnld")
        button = html.Button("Generate Report", id="btn-generate")

        return [dcc.Graph(figure=chart), button, None]

    else:
        raise PreventUpdate


@callback(
    Output("download-link-container", "children"),
    Input("btn-generate", "n_clicks"),
    Input('timeframe-dropdown', 'value'),
    prevent_initial_call=True,
)
def generate_report(n_clicks, ticker):
    if n_clicks:
        print("Report generated!")  # Replace with actual report generation code if needed

        # Create a link to download the file
        download_link = html.A(
            "Click here to download the report",
            href=file_path,
            download=f"{ticker}_StrategyBacktest.html",  # This attribute prompts download on click
            target="_blank"  # Opens in a new tab if needed
        )
        return download_link

    return "Click the button to generate the report."

# @callback(
#     Output("download-report", "data"),
#     Input("btn-dwnld", "n_clicks"),
#     prevent_initial_call=True,
# )
# def func(n_clicks):
#     # Set an absolute path
#     file_path = os.path.abspath("./dash-community-components.png")
#     print(file_path)
#
#     if n_clicks:
#         print('Download button clicked!')
#
#         # Verify file exists
#         if os.path.exists(file_path):
#             print('File path exists')
#             return dcc.send_file('./dash-community-components.png')
#         else:
#             raise Exception(f"File not found: {file_path}")
#
#     raise PreventUpdate
