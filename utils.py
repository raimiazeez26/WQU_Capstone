import yfinance as yf
import pandas as pd
import numpy as np

from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

import quantstats_lumi as qs

# Initialize QuantStats
qs.extend_pandas()


def fetch_data(ticker, tf):
    # Data Collection from yfinance
    # EUR/USD Data
    eurusd_daily = yf.download(f'{ticker}=X', period='10y', interval=tf)
    eurusd_daily = eurusd_daily.droplevel(level=1, axis=1)

    # VIX (Volatility Index)
    vix_daily = yf.download('^VIX', period='10y', interval=tf)
    # USDX (US Dollar Index)
    usdx_daily = yf.download('DX-Y.NYB', period='10y', interval=tf)
    # SP500 data
    sp_daily = yf.download('^GSPC', period='10y', interval=tf)

    # Macroeconomic Proxies from yfinance
    # Non-Farm Payroll proxy (XLI - Industrials ETF)
    xli = yf.download('XLI', period='10y', interval=tf)
    # CPI proxy (TIP - iShares TIPS Bond ETF)
    tip = yf.download('TIP', period='10y', interval=tf)
    # Treasury Yield proxy (TLT - iShares 20+ Year Treasury Bond ETF)
    tlt = yf.download('TLT', period='10y', interval=tf)
    # Interest Rate proxy (SHY - iShares 1-3 Year Treasury Bond ETF)
    shy = yf.download('SHY', period='10y', interval=tf)

    # Merge macroeconomic data into one DataFrame
    macro_data = pd.DataFrame()
    macro_data['XLI_Close'] = xli['Close']
    macro_data['TIP_Close'] = tip['Close']
    macro_data['TLT_Close'] = tlt['Close']
    macro_data['SHY_Close'] = shy['Close']

    macro_data.index = pd.to_datetime(macro_data.index)

    return eurusd_daily, vix_daily, usdx_daily, macro_data, sp_daily


# Technical Indicator Calculation
def add_indicators(data):
    # Add EMA, ADX, MACD, RSI, and Bollinger Bands
    ema1 = EMAIndicator(close=data['Close'], window=7)
    ema2 = EMAIndicator(close=data['Close'], window=21)
    data['EMA7'] = ema1.ema_indicator()
    data['EMA21'] = ema2.ema_indicator()

    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd_signal()

    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    bb = BollingerBands(close=data['Close'], window=20)
    data['bb_mavg'] = bb.bollinger_mavg()
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()

    return data.dropna()


# Signal Generation
def generate_signals(data):
    # data['Buy_Signal'] = (data['RSI'] < 30) & (data['Close'] < data['bb_lower'])
    # data['Sell_Signal'] = (data['RSI'] > 70) & (data['Close'] > data['bb_upper'])

    data['Signal'] = np.where((data['RSI'] < 30) & (data['Close'] < data['bb_lower']), 'Buy', 'Hold')
    data['Signal'] = np.where((data['RSI'] > 70) & (data['Close'] > data['bb_upper']), 'Sell', data['Signal'])
    return data


# Plot signals with Bollinger Bands and RSI
def plot_subplots(data, signal_col, ticker):
    """
    Create subplots for EUR/USD price data, including Bollinger Bands and RSI signals.

    Parameters:
        :param data: (DataFrame) DataFrame containing price data and signals.
        :param signal_col:  (str) Column name for the signal ('Buy', 'Sell', 'Hold').
        :param ticker: ticker of asset
    """
    # Filter out signals that are 'Hold'
    df_signals = data[data[signal_col] != 'Hold']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[800, 200],
        shared_xaxes=True,
        vertical_spacing=0.02,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Add Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        ),
        secondary_y=True,
        row=1, col=1
    )

    # Add Bollinger Bands traces
    fig.add_trace(
        go.Scatter(x=data.index, y=data["bb_lower"], name='Lower Bollinger Band',
                   line=dict(color='royalblue', width=1)),
        secondary_y=True, row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data['bb_upper'], name='Upper Bollinger Band',
                   line=dict(color='royalblue', width=1)),
        secondary_y=True, row=1, col=1
    )

    # Add Buy/Sell annotations
    for i, dtt in df_signals.iterrows():
        if dtt[signal_col] == 'Buy':
            fig.add_annotation(
                x=i,
                y=dtt['Low'] - 0.05,
                xref='x',
                yref='y',
                text="▲",
                showarrow=False,
                font=dict(size=20, color='Green'),
                secondary_y=True,
                row=1, col=1
            )
        elif dtt[signal_col] == 'Sell':
            fig.add_annotation(
                x=i,
                y=dtt['High'] + 0.05,
                xref='x',
                yref='y',
                text="▼",
                showarrow=False,
                font=dict(size=20, color='Red'),
                secondary_y=True,
                row=1, col=1
            )

    # Add RSI trace
    fig.add_trace(
        go.Scatter(x=data.index, y=data.RSI,
                   line=dict(color='red', width=1),
                   mode='lines',
                   name='RSI'),
        row=2, col=1
    )

    # Add horizontal lines for RSI thresholds
    fig.add_hline(y=30.0, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=70.0, line_dash="dash", line_color="green", row=2, col=1)

    # Update layout settings
    fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=1, t=50, b=10),
        title_text=f"{signal_col} - {ticker} with Bollinger Bands and RSI",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500
    )

    # Customize axes appearance
    fig.update_xaxes(showgrid=False, showspikes=True, rangebreaks=[
        dict(bounds=["sat", "mon"])  # Hide weekends
    ])

    fig.update_yaxes(showgrid=False)

    # Show the figure
    # fig.show()

    return fig


# Apply Broad Market Filter with Macroeconomic Proxies
def apply_directional_filter(eurusd, vix, usdx, macro, sp):
    """
    Apply directional filters to EUR/USD data based on VIX, USDX, macroeconomic proxies, and SP500 trends.

    Parameters:
        eurusd (DataFrame): DataFrame containing EUR/USD price data and signals.
        vix (DataFrame): DataFrame containing VIX price data.
        usdx (DataFrame): DataFrame containing USDX price data.
        macro (DataFrame): DataFrame containing macroeconomic proxies.
        sp (DataFrame): DataFrame containing SP500 price data.

    Returns:
        DataFrame: Updated EUR/USD DataFrame with directional signals applied.
    """

    # VIX Signal: Trade when current close > VIX 20-day mean
    vix['vix_Signal'] = np.where(vix['Close'].rolling(window=20).mean() > vix['Close'], 'Trade', 'No Trade')

    # USDX Signal: Bullish if current close > USDX 20-day mean
    usdx['usdx_Signal'] = np.where(usdx['Close'].rolling(window=20).mean() < usdx['Close'], 'Bullish', 'Bearish')

    # Macroeconomic Conditions
    macro['xli_Signal'] = np.where(macro['XLI_Close'].rolling(window=20).mean() > macro['XLI_Close'], 'Bullish',
                                   'Bearish')
    macro['tip_Signal'] = np.where(macro['TIP_Close'].rolling(window=20).mean() > macro['TIP_Close'], 'Bullish',
                                   'Bearish')
    macro['tlt_Signal'] = np.where(macro['TLT_Close'].rolling(window=20).mean() < macro['TLT_Close'], 'Bullish',
                                   'Bearish')
    macro['shy_Signal'] = np.where(macro['SHY_Close'].rolling(window=20).mean() < macro['SHY_Close'], 'Bullish',
                                   'Bearish')

    # Market Trend from SP500
    sp['pct_change'] = sp['Close'].pct_change()
    sp['sp_signal'] = np.where(sp['Close'].rolling(window=20).mean() < sp['Close'], 'Bullish', 'Bearish')

    # Combine signals with EUR/USD
    eurusd = eurusd.copy()  # Avoid SettingWithCopyWarning
    # Generate signals
    eurusd = generate_signals(eurusd)

    eurusd['VIX_Signal'] = vix['vix_Signal']
    eurusd['USDX_Signal'] = usdx['usdx_Signal']
    eurusd['XLI_Signal'] = macro['xli_Signal']
    eurusd['TIP_Signal'] = macro['tip_Signal']
    eurusd['TLT_Signal'] = macro['tlt_Signal']
    eurusd['SHY_Signal'] = macro['shy_Signal']
    eurusd['Mkt_Trend'] = sp['sp_signal']

    # Signal 1: Combination of mean reversion + market trend only
    eurusd['Signal_1'] = np.where((eurusd['Signal'] == 'Buy') & (eurusd['Mkt_Trend'] == 'Bullish'), 'Buy', 'Hold')
    eurusd['Signal_1'] = np.where((eurusd['Signal'] == 'Sell') & (eurusd['Mkt_Trend'] == 'Bearish'), 'Sell',
                                  eurusd['Signal_1'])

    # Signal 2: Combination of mean reversion + USDX only + VIX
    eurusd['Signal_2'] = np.where(
        (eurusd['Signal'] == 'Buy') & (eurusd['USDX_Signal'] == 'Bearish') & (eurusd['VIX_Signal'] == 'Trade'), 'Buy',
        'Hold')
    eurusd['Signal_2'] = np.where(
        (eurusd['Signal'] == 'Sell') & (eurusd['USDX_Signal'] == 'Bullish') & (eurusd['VIX_Signal'] == 'Trade'), 'Sell',
        eurusd['Signal_2'])

    # Signal 3: Combination of mean reversion + macro data
    eurusd['Signal_3'] = np.where(
        (eurusd['Signal'] == 'Buy') &
        (eurusd[['XLI_Signal', 'TIP_Signal', 'TLT_Signal', 'SHY_Signal']] == ['Bullish', 'Bearish', 'Bullish',
                                                                              'Bullish']).all(axis=1),
        'Buy',
        'Hold'
    )

    eurusd['Signal_3'] = np.where(
        (eurusd['Signal'] == 'Sell') &
        (eurusd[['XLI_Signal', 'TIP_Signal', 'TLT_Signal', 'SHY_Signal']] == ['Bearish', 'Bullish', 'Bearish',
                                                                              'Bearish']).all(axis=1),
        'Sell',
        eurusd['Signal_3']
    )

    # Hybrid Signal: Combination of all filters
    eurusd['Hybrid_Signal'] = np.where(
        (eurusd[['USDX_Signal']] == ['Bearish']).all(axis=1) &
        (eurusd[['XLI_Signal']] == ['Bullish']).all(axis=1) &
        (eurusd[['TIP_Signal']] == ['Bullish']).all(axis=1) &
        (eurusd[['TLT_Signal']] == ['Bullish']).all(axis=1) &
        (eurusd[['SHY_Signal']] == ['Bullish']).all(axis=1),
        'Buy',
        'Hold'
    )

    eurusd['Hybrid_Signal'] = np.where(
        (eurusd[['USDX_Signal']] == ['Bearish']).all(axis=1) &
        (eurusd[['XLI_Signal']] == ['Bearish']).all(axis=1) &
        (eurusd[['TIP_Signal']] == ['Bearish']).all(axis=1) &
        (eurusd[['TLT_Signal']] == ['Bearish']).all(axis=1) &
        (eurusd[['SHY_Signal']] == ['Bearish']).all(axis=1),
        'Sell',
        eurusd['Hybrid_Signal']
    )

    return eurusd


# class Position contain data about trades opened/closed during the backtest
class Position:
    def __init__(self, open_datetime, open_price, order_type):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.close_datetime = None
        self.close_price = None
        self.profit = None
        self.status = 'open'
        self.returns = 0.0

    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price

        if self.order_type == 'Buy':
            self.returns = (self.close_price - self.open_price) / self.open_price

        elif self.order_type == 'Sell':
            self.returns = (self.open_price - self.close_price) / self.open_price

        self.status = 'closed'

    def _asdict(self):
        return {
            'open_datetime': self.open_datetime,
            'open_price': self.open_price,
            'order_type': self.order_type,
            'close_datetime': self.close_datetime,
            'close_price': self.close_price,
            'returns': round(self.returns, 5),
            'status': self.status,
        }


# class Strategy defines trading logic and evaluates the backtest based on opened/closed positions
class Strategy:
    def __init__(self, df, signal_col):
        self.positions = []
        self.data = df.copy()
        self.data.reset_index(inplace=True)  #
        self.data = self.data.rename(columns={'Date': 'date'})
        self.signal_col = signal_col
        self.data[self.signal_col] = self.data[self.signal_col].shift(1)  # shift signal one step for next day open
        self.last_date = str(self.data['date'].iloc[-1])

    # return backtest result
    def get_positions_df(self):
        df = pd.DataFrame([position._asdict() for position in self.positions])
        # print(df['returns'])
        df["cumulative_returns"] = (1 + (df['returns']).cumprod())
        return df

    # add Position class to list
    def add_position(self, position):
        self.positions.append(position)
        return True

    # close positions when stop loss or take profit is reached
    def close_tp_sl(self, data):

        for pos in self.positions:
            if pos.status == 'open':

                # close Buy position if signal changes to Sell
                if pos.order_type == 'Buy' and data[self.signal_col] == 'Sell':
                    pos.close_position(data['date'], data.Close)

                # close sell position if signal changes to Buy
                elif pos.order_type == 'Sell' and data[self.signal_col] == 'Buy':
                    pos.close_position(data['date'], data.Close)

                # close any position on last backtest date
                elif str(data['date']) == self.last_date:
                    pos.close_position(data['date'], data.Close)

    # check for open positions
    def has_open_positions(self):
        for pos in self.positions:
            if pos.status == 'open':
                return True
        return False

    # strategy logic how positions should be opened/closed
    def logic(self, data):

        # if no position is open
        if not self.has_open_positions():

            # if signal on last backtest date, do not open
            if str(data['date']) == self.last_date:
                pass

            else:

                # BUY
                if data[self.signal_col] == "Buy":

                    # Position variables
                    open_datetime = data['date']
                    open_price = data['Open']
                    order_type = 'Buy'

                    self.add_position(Position(open_datetime, open_price, order_type))

                # SELL
                elif data[self.signal_col] == "Sell":

                    # Position variables
                    open_datetime = data['date']
                    open_price = data['Open']
                    order_type = 'Sell'

                    self.add_position(Position(open_datetime, open_price, order_type))

                # Neutral
                else:
                    pass

    # logic
    def run(self):
        # data represents a moment in time while iterating through the backtest
        for i, data in self.data.iterrows():
            # close positions when stop loss or take profit is reached
            self.close_tp_sl(data)

            # strategy logic
            self.logic(data)

        return self.get_positions_df()

#
# price_daily, vix_daily, usdx_daily, macro_data, sp_daily = fetch_data('EURUSD')
# data = add_indicators(price_daily)
#
# data_filters = apply_directional_filter(price_daily, vix_daily, usdx_daily, macro_data, sp_daily)
#
# # Run the strategy using EUR/USD data
# strategy = Strategy(data_filters, 'Signal_1')
# result = strategy.run()
#
# # Convert index to datetime
# result = result.set_index('open_datetime')
#
# print(type(result.index[0]))
# result.index = pd.to_datetime(result.index)
#
#
# # generate report tearsheet with SPY BENCHMARK
# qs.reports.html(result['cumulative_returns'], title=f'Strategy backtest',
#                 download_filename=f'Strategy Backtest.html', output=f'Strategy Backtest.html')
