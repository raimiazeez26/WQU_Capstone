import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from .side_bar import sidebar

dash.register_page(__name__, path='/', name='Project Description', order=0)

def layout():
    return html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        sidebar()
                    ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

                dbc.Col(
                    [
                        html.H2('A Novel Hybrid Directional Filter Approach for enhanced FOREX Trade Signal Prediction',
                                style={'textAlign': 'center'}, className='p4'),
                        # html.H5('This Page is Actively Updated', style={'textAlign': 'center'},
                        #         className='p4 text-warning'),
                        html.Hr(),

                        html.H5('ABSTRACT', style={'textAlign': 'center'},
                                className='p4'),
                        html.P(
                            'FOREX trading is a complex task due to the high risk and volatility involved, as'
                            'well as the need to get a better sense of the wider market conditions, in order to make a'
                            'decisive and profitable trading decision. Traditional methods of fundamental and'
                            'technical analysis may not be adequate in itself to predict directional movement of '
                            'prices.'

                            # 'Please switch Theme (top right corner) for varying experience for each app!'
                            , style={'textAlign': 'left'}),

                        html.P(
                            'This study proposes a novel hybrid filter approach to boost the precision of FOREX'
                            'signals via integration of macroeconomic data, broad market filters and technical'
                            'indicators. Results indicate that the hybrid approach performed better in terms of'
                            'improved trade signals, compared to individual filters.'

                            # 'Please switch Theme (top right corner) for varying experience for each app!'
                            , style={'textAlign': 'left'}),

                        html.P(
                            'A LSTM model was trained on the data and predictive performance presented. '
                            'This paper highlights the importance of capturing more broad market indicators and other '
                            'economic data in the prediction of trade signals.'

                            # 'Please switch Theme (top right corner) for varying experience for each app!'
                            , style={'textAlign': 'left'}),

                        html.P(
                            'Keywords: predictive modeling, FOREX, precision trading, directional filters, '
                            'financial engineering \n \n'

                            'Please switch Theme (top right corner) for varying experience for each app!'
                            , style={'textAlign': 'center'}),

                        # dcc.Markdown('''
                        #     Other Project Sources
                        #     * [Github](https://github.com/raimiazeez26)
                        #     * [Tableau](https://public.tableau.com/app/profile/raimi.azeez.babatunde)
                        #     * [Kaggle](https://www.kaggle.com/raimiazeezbabatunde)
                        #     ''', style={'textAlign': 'left'}),

                        html.Hr(),
                        # dcc.Graph(figure=fig, id='line_chart'),

                    ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10,
                    className='p-3'
                )
            ]
        )
    ])

