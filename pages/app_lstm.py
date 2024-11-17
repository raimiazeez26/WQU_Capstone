import warnings
import os
import dash
from dash import html, dcc, ctx, callback
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
from pages.side_bar import sidebar
from pages.lstm import run_lstm
import redis
from rq import Queue

# Suppress warnings
warnings.filterwarnings("ignore")

dash.register_page(__name__, name='PREDICTION MODEL')

redis_url = os.getenv('REDISCLOUD_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
q = Queue(connection=conn)

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
                html.H3('LSTM MODEL PREDICTION', style={'textAlign': 'center'},
                        className='p4'),
                html.Br(),
                html.Hr(),
                html.P(
                    'LSTM Predictive Model. The development of a LSTM model trained on the various FOREX currency '
                    'pair data and the enriched dataset to predict price movement of the FOREX data was explored. '
                    'The data for each pair was split in training, validation and test in the ratio of 0.7, 0.1 and 0.2'
                    'respectively, with MinMax scaling. A total of 50 epochs was instituted for the training. '
                    'Back testing and mean error squared (MSE), mean absolute error (MAE) and root mean squared error '
                    '(RMSE) were performed to evaluate the predictive performance of the model with the actual data.'
                    , style={'textAlign': 'left'}),
                html.Br(),
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
                            html.P('No of Epochs:'),
                            dbc.Input(id='no-epochs', type='number', min=1, max=100, value=10),
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

                        html.Div(id='transactions-result'),
                        html.Div(id='training-result2'),
                        html.Div(id='transactions-result2'),
                        dcc.Loading(
                            id="result-loading",
                            children=[
                                html.Div(id='job-id', style={'display': 'none'}),
                            ],
                            type="circle",
                        ),

                        dcc.Interval(id='interval-component', interval=2 * 1000, n_intervals=0, disabled=True),

                    ]),
                ])
            ])

            ])
    ], fluid=True, class_name='g-0', )


@callback(
    Output('job-id', 'children'),
    Output('interval-component', 'disabled', allow_duplicate=True),
    Input("run-button-lstm", "n_clicks"),
    Input('symbol-dropdown-lstm', 'value'),
    Input('timeframe-dropdown-lstm', 'value'),
    Input('no-epochs', 'value'),
    prevent_initial_call=True
)
def run_model(n_click, ticker, timeframe, no_epochs):
    if "run-button-lstm" == ctx.triggered_id:
        print('Button Clicked')
        # figs, transactions = run_lstm(ticker, timeframe)

        job = q.enqueue(run_lstm, ticker, timeframe, no_epochs)
        # job = q.enqueue(test_job)
        print(f"Job ID: {job.id}")

        # print(f'Job Created with ID: {job.id}')
        return job.id, False
    else:
        # return None, True
        raise PreventUpdate


@callback(
    Output("training-result", "children"),
    Output("transactions-result", "children"),
    Output("training-result2", "children"),
    Output("transactions-result2", "children"),
    Output('interval-component', 'disabled'),
    Input('interval-component', 'n_intervals'),
    Input('symbol-dropdown-lstm', 'value'),
    State('job-id', 'children'),
    prevent_initial_call=True,
)
def train_predict(n_intervals, ticker, job_id):
    if job_id:
        job = q.fetch_job(job_id)
        if job.is_finished:
            figs, transactions = job.return_value()

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
            return [[html.Br(), html.Hr(), dcc.Graph(figure=figs[0])],
                    [html.Br(), html.Hr(), transaction_table1],
                    [html.Br(), html.Hr(), dcc.Graph(figure=figs[1])],
                    [html.Br(), html.Hr(), transaction_table2], True]

        elif job.is_failed:
            return 'Model Training failed', None, None, None, True  # None, None,
        else:
            return 'Training Model... This might take a few minutes, please wait...', None, None, None, False

    raise PreventUpdate
    # return [None, None, None, None, False]