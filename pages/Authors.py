import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/authors', order=0)

green_text = {'color':'green'}
#image card
picture_card = dbc.Card(
    [
        dbc.CardImg(src="assets/portrait.jpeg", top=True),
        #dbc.CardBody(
            #html.P("This card has an image at the top", className="card-text")
        #),
    ],
    #style={"width": "21rem"},
    className='border rounded-lg',
)

def layout():
    return dbc.Row([
        # put image here
        # dbc.Col([
        #     picture_card
        # ], xs=1, sm=2, md=2, lg=3, xl=3, xxl=3, #style={'align': 'center'},
        #     className='pt-5'), #width={"size": 3} #float-right

        dbc.Col([
            html.H1('A Novel Hybrid Directional Filter Approach for enhanced FOREX Trade Signal Prediction',
                    style={'textAlign': 'center'}, className='p4'),
            html.Hr(),
            #picture_card,
            html.P("Matthew Chua Chin Heng, Azeez Babatunde Raimi, Dejen Zelalem Nugusse",
                 style={'textAlign': 'center'}),
            html.P("WorldQuant University, USA",
                   style={'textAlign': 'center'}),

            html.P("Games_snake@yahoo.com.sg, raimiazeez26@gmail.com, dejen4chess@gmail.com",
                   style={'textAlign': 'center'}),

            # html.P("When I'm not geeking out on data, you can find me exploring the fascinating realms of "
            #      "Finance, or in the field flexing my photography skills.\n"
            #      "Ready to embark on a data-driven adventure? Let's innovate together!",
            #        style={'textAlign': 'left'}),
            #
            # html.P("With my diverse skill set, I am suitable and well-equipped to excel in a variety of roles:\n",
            #        style={'textAlign': 'left'}),

        ], xs=6, sm=6, md=8, lg=8, xl=8, xxl=8, className='align-content-around flex-wrap') #width={'size':6}

], justify='center', className='p-3')