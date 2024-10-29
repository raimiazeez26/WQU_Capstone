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

dash.register_page(__name__, name='FILTERS BACKTEST')

# creates the Dash App
# app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

##=========================================================================================================
# Inputs parameters
##=========================================================================================================
TIMEFRAMES = ['1d']
SYMBOLS = ['EURUSD', 'GBPUSD']
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
                           '** Click the Run Backtest button to start. **'
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
                            )
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
    Input('symbol-dropdown', 'value'),
    Input('timeframe-dropdown', 'value'),
    Input('filters-dropdown', 'value'),
    Input('run-button', 'n_clicks'),
    prevent_initial_call=True)

def plot_signals(ticker, timeframe, filter, n_clicks):
    if "run-button" == ctx.triggered_id:

        price_daily, vix_daily, usdx_daily, macro_data, sp_daily = fetch_data(ticker)
        data = add_indicators(price_daily)

        data_filters = apply_directional_filter(price_daily, vix_daily, usdx_daily, macro_data, sp_daily)

        if filter == 'HYBRID':
            chart = plot_subplots(data_filters, 'Hybrid_Signal', ticker)

        elif filter == 'FILTERS2':
            chart = plot_subplots(data_filters, 'Signal_2', ticker)

        return dcc.Graph(figure=chart)

    else:
        raise PreventUpdate
#
#
# @callback(
#     Output("chart-content", "children"),
#     # Output('price-plot-div', 'src'),
#     Input("tabs", "active_tab"),
#     Input('train_store', 'data'),
#     Input('test_store', 'data'),
#     Input('symbol-dropdown', 'value'),
#     Input('timeframe-dropdown', 'value'),
#     Input('split', 'value'),
#     Input('diff-checkbox', 'value'),
#     Input('log-checkbox', 'value'),
#     Input('run-button', 'n_clicks'),
#     prevent_initial_call=True
#
# )
# def Run_model(tab, train_data, test_data, symbol, timeframe, split, diff, log, n_clicks):  # ,
#
#     # if "run-button" == ctx.triggered_id:
#     train_data = pd.DataFrame(train_data)
#     test_data = pd.DataFrame(test_data)
#     best, model, summary = model_selection(train_data)
#
#     split = int(split)
#
#     # if "run-button" == ctx.triggered_id:
#
#     if tab == "chart":
#
#         tr_data, te_data, price_chart = chart_data(symbol, timeframe, split=int(split), diff=diff, log=log)
#
#         return html.Div([html.Img(id='price_plot', src=price_chart)],
#                         id='price-plot-div')  # , price_chart
#
#     elif tab == "acf_pacf":
#
#         acf_pacf = acf_pcf(train_data)
#         return html.Div([html.Img(id='acf_pacf', src=acf_pacf)],
#                         id='acf_pacf-div')  # , price_chart
#
#     elif tab == "model-selection":
#
#         # model, summary = model_selection(train_data)
#
#         return html.Div(
#             [
#                 html.P(f'Best Model : ARIMA {best}'),
#                 html.P(children=str(summary), style={'whiteSpace': 'pre-wrap'})
#
#             ], id='model_selection-div')  # , price_chart
#
#     elif tab == "model-diagnostics":
#
#         dig = diagnostics(model)
#         return html.Div([html.Img(id='giagnostics', src=dig)],
#                         id='dig-div')  # , price_chart
#
#     elif tab == "model-forecast":
#         if log:
#
#             fore = forecast_log(model, train_data, test_data, symbol, timeframe, split=split, alpha1=0.20, alpha2=0.05)
#             return html.Div([html.Img(id='forecast', src=fore)],
#                             id='forecast-div')  # , price_chart
#         else:
#             fore = forecast(model, train_data, test_data, symbol, timeframe, split=split, alpha1=0.20, alpha2=0.05)
#             return html.Div([html.Img(id='forecast', src=fore)],
#                             id='forecast-div')  # , price_chart
#
#     # else:
#     # raise PreventUpdate