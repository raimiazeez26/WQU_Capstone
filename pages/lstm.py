import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# LSTM Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Multiply, Permute
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

#RFE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Set a random seed for reproducibility
np.random.seed(42)


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
    usdx['usdx_Signal'] = np.where(usdx['Close'].rolling(window=20).mean() < usdx['Close'], 'Buy', 'Sell')

    # Macroeconomic Conditions
    macro['xli_Signal'] = np.where(macro['XLI_Close'].rolling(window=20).mean() > macro['XLI_Close'], 'Buy',
                                   'Sell')
    macro['tip_Signal'] = np.where(macro['TIP_Close'].rolling(window=20).mean() > macro['TIP_Close'], 'Buy',
                                   'Sell')
    macro['tlt_Signal'] = np.where(macro['TLT_Close'].rolling(window=20).mean() < macro['TLT_Close'], 'Buy',
                                   'Sell')
    macro['shy_Signal'] = np.where(macro['SHY_Close'].rolling(window=20).mean() < macro['SHY_Close'], 'Buy',
                                   'Sell')

    # Market Trend from SP500
    sp['pct_change'] = sp['Close'].pct_change()
    sp['sp_signal'] = np.where(sp['Close'].rolling(window=20).mean() < sp['Close'], 'Buy', 'Sell')

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
    eurusd['Signal_1'] = np.where((eurusd['Signal'] == 'Sell') & (eurusd['Mkt_Trend'] == 'Sell'), 'Sell',
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
        (eurusd[['XLI_Signal', 'TIP_Signal', 'TLT_Signal', 'SHY_Signal']] == ['Buy', 'Sell', 'Buy',
                                                                              'Buy']).all(axis=1),
        'Buy',
        'Hold'
    )

    eurusd['Signal_3'] = np.where(
        (eurusd['Signal'] == 'Sell') &
        (eurusd[['XLI_Signal', 'TIP_Signal', 'TLT_Signal', 'SHY_Signal']] == ['Sell', 'Buy', 'Sell',
                                                                              'Sell']).all(axis=1),
        'Sell',
        eurusd['Signal_3']
    )

    # Hybrid Signal: Combination of all filters
    eurusd['Hybrid_Signal'] = np.where(
        (eurusd[['USDX_Signal']] == ['Sell']).all(axis=1) &
        (eurusd[['XLI_Signal']] == ['Buy']).all(axis=1) &
        (eurusd[['TIP_Signal']] == ['Buy']).all(axis=1) &
        (eurusd[['TLT_Signal']] == ['Buy']).all(axis=1) &
        (eurusd[['SHY_Signal']] == ['Buy']).all(axis=1),
        'Buy',
        'Hold'
    )

    eurusd['Hybrid_Signal'] = np.where(
        (eurusd[['USDX_Signal']] == ['Buy']).all(axis=1) &
        (eurusd[['XLI_Signal']] == ['Sell']).all(axis=1) &
        (eurusd[['TIP_Signal']] == ['Sell']).all(axis=1) &
        (eurusd[['TLT_Signal']] == ['Sell']).all(axis=1) &
        (eurusd[['SHY_Signal']] == ['Sell']).all(axis=1),
        'Sell',
        eurusd['Hybrid_Signal']
    )

    return eurusd


# feature selection function
def feature_selection(X, y, no_features):

    # Initialize the model
    model = RandomForestRegressor()

    # Initialize the RFE selector with the model and number of features to select
    selector = RFE(model, n_features_to_select=no_features)  # Select the top 5 features

    # Fit the selector on the data
    selector = selector.fit(X, y)
    # selector = selector.fit(X_train_normalized, y_train)

    # Get the selected features
    selected_features = X.columns[selector.support_]
    print("Selected Features:", list(selected_features))

    return selected_features, selector.ranking_



# Function to create sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :-1])  # Input features (without 'close')
        y.append(data[i + n_steps, -1])  # Output (close price)
    return np.array(X), np.array(y)

# Prepare data for LSTM
def prepare_data(data, no_features=6, time_step=20, test_size=0.2, val_size=0.1, use_hybrid=True):

    # Define a mapping dictionary
    replacement_dict = {
        'Hold': 0,
        'Buy': 1,
        'Sell': -1
    }

    cols = ['Signal', 'VIX_Signal', 'USDX_Signal', 'XLI_Signal', 'TIP_Signal', 'TLT_Signal',
            'SHY_Signal', 'Mkt_Trend', 'Signal_1', 'Signal_2', 'Signal_3', 'Hybrid_Signal']

    # Replace the strings using the map function in pandas
    for col in cols:
        data[col] = data[col].map(replacement_dict)

    # data = data.dropna()
    data = data.iloc[55:-1]
    data = data.fillna(0)
    # print(data)

    # USE FEATURE SELECTION
    # X = data.drop(['Close'], axis=1)
    # y = data['Close']

    # feature_col, rankings = feature_selection(X, y, no_features)

    # if feature_col not in data.columns:
    #     raise ValueError(f"Column '{feature_col}' not found in the dataset")
    # feature_col = list(feature_col) + ['Close']

    # data = data[feature_col].values


    # move close column to last column
    df = data.copy()

    if use_hybrid:
      cols_to_drop = ['Signal', 'VIX_Signal', 'USDX_Signal', 'XLI_Signal', 'TIP_Signal', 'TLT_Signal',
              'SHY_Signal', 'Mkt_Trend', 'Signal_1', 'Signal_2', 'Signal_3', 'Hybrid_Signal', 'Close']

    else:
      cols_to_drop = ['Signal', 'VIX_Signal', 'USDX_Signal', 'XLI_Signal', 'TIP_Signal', 'TLT_Signal',
              'SHY_Signal', 'Mkt_Trend', 'Signal_1', 'Signal_2', 'Signal_3', 'Close']

    data = data.drop(cols_to_drop, axis = 1)
    data['Close'] = df['Close']


    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Split data into train, validation, and test sets
    total_size = len(data)
    train_size = int(total_size * (1 - test_size - val_size))
    val_size = int(total_size * val_size)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Fit scaler only on training data
    train_data_scaled = scaler.fit_transform(train_data)
    print(f'Shape of train_data_scaled: {train_data_scaled.shape}')
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Create sequences for LSTM
    X_train, y_train = create_sequences(train_data_scaled, time_step)
    X_test, y_test = create_sequences(test_data_scaled, time_step)
    X_val, y_val = create_sequences(val_data_scaled, time_step)

    # Assuming y_train is your target, reshape it for fitting:
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler


# Custom Attention Layer
def attention_layer(inputs):
    # Permute dimensions for attention mechanism
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a = Permute((2, 1), name='attention_weights')(a)
    output_attention_mul = Multiply()([inputs, a])
    return output_attention_mul


# Build LSTM Model
def build_lstm_model(X_train, y_train, X_val, y_val, epochs):

    # Build the LSTM model with Attention
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(units=64, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LSTM(units=64, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LSTM(units=32, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Apply attention mechanism
    x = attention_layer(x)

    # Flatten the output of the attention layer
    x = tf.keras.layers.Flatten()(x)

    # Dense layer with L2 regularization
    x = Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')(x)

    # Final output layer (close price prediction)
    outputs = Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a specific learning rate and Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Use EarlyStopping to monitor validation loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with validation data
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=1)

    return model, history

# Train LSTM Model
def train_lstm(ticker, data, time_step=10, test_size=0.2, val_size=0.1, epochs=50, batch_size=32, use_hybrid=True):
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = prepare_data(
        data, time_step=time_step, test_size=test_size, val_size=val_size, use_hybrid=use_hybrid
    )
    model, history = build_lstm_model(x_train, y_train, x_val, y_val, epochs)

    plot_training_history(history, ticker, use_hybrid)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)

    return model, scaler, x_train, y_train, x_val, y_val, x_test, y_test

# Predict using the trained LSTM model
def predict_lstm(model, x_data_scaled, scaler):
    predictions_scaled = model.predict(x_data_scaled)
    print('Prediction Scaled Shape:')
    print(predictions_scaled.shape)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    return predictions


# Function to plot training history (to track loss and accuracy)
def plot_training_history(history, ticker, use_hybrid):
    # Plot loss for both training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title(f'{ticker} Training vs Validation Loss - Hybrid {use_hybrid}')
    plt.savefig(f'{ticker}_Validation Loss - Hybrid {use_hybrid}.png')
    plt.show()

    # If you want to plot Mean Absolute Error (MAE)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.title(f'{ticker} Training vs Validation MAE - Hybrid {use_hybrid}')
    plt.savefig(f'{ticker}_Validation MAE - Hybrid {use_hybrid}.png')
    plt.show()


# Plot actual vs predicted values for train, validation, and test sets
def plot_predictions(ticker, actual_train, predicted_train, actual_val, predicted_val, actual_test, predicted_test,
                     use_hybrid):
    # Flatten the input arrays
    actual_train = actual_train.flatten()
    predicted_train = predicted_train.flatten()
    actual_val = actual_val.flatten()
    predicted_val = predicted_val.flatten()
    actual_test = actual_test.flatten()
    predicted_test = predicted_test.flatten()

    # Create x values for each dataset
    x_train = list(range(len(actual_train)))
    x_val = list(range(len(actual_val)))
    x_test = list(range(len(actual_test)))

    # Create a figure with three rows, one for each dataset
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                        subplot_titles=(
                            f'{ticker} Train: Actual vs Predicted Values - Hybrid {use_hybrid}',
                            f'{ticker} Validation: Actual vs Predicted Values - Hybrid {use_hybrid}',
                            f'{ticker} Test: Actual vs Predicted Values - Hybrid {use_hybrid}'
                        ))

    # Add traces for the training data
    fig.add_trace(go.Scatter(x=x_train, y=actual_train,
                             mode='lines', name='Actual Train Values',
                             line=dict(color='blue', width=2), opacity=0.6),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x_train, y=predicted_train,
                             mode='lines', name='Predicted Train Values',
                             line=dict(color='red', width=2), opacity=0.6),
                  row=1, col=1)

    # Add traces for the validation data
    fig.add_trace(go.Scatter(x=x_val, y=actual_val,
                             mode='lines', name='Actual Validation Values',
                             line=dict(color='blue', width=2), opacity=0.6, showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=x_val, y=predicted_val,
                             mode='lines', name='Predicted Validation Values',
                             line=dict(color='red', width=2), opacity=0.6, showlegend=False),
                  row=2, col=1)

    # Add traces for the testing data
    fig.add_trace(go.Scatter(x=x_test, y=actual_test,
                             mode='lines', name='Actual Test Values',
                             line=dict(color='blue', width=2), opacity=0.6, showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=predicted_test,
                             mode='lines', name='Predicted Test Values',
                             line=dict(color='red', width=2), opacity=0.6, showlegend=False),
                  row=3, col=1)

    # Update layout
    fig.update_layout(height=900, width=1000, title_text=f'{ticker} Predictions - Hybrid {use_hybrid}')

    # Update axis labels
    fig.update_xaxes(title_text='Time Steps', row=3, col=1)
    fig.update_yaxes(title_text='Values', row=1, col=1)
    fig.update_yaxes(title_text='Values', row=2, col=1)
    fig.update_yaxes(title_text='Values', row=3, col=1)

    # Show the figure
    # fig.show()
    return fig



# Calculate performance metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


# Create transaction table
def create_transaction_table(actual, predicted, threshold=0.01):
    transaction_data = {
        'Actual': actual.flatten(),
        'Predicted': predicted.flatten(),
        'Direction': np.where(predicted > actual, 'Buy', 'Sell')
    }

    # Create a DataFrame
    transactions_df = pd.DataFrame(transaction_data)

    # Identify significant transactions
    transactions_df['Transaction'] = np.where(
        abs(transactions_df['Predicted'] - transactions_df['Actual']) > threshold,
        transactions_df['Direction'],
        'Hold'
    )

    return transactions_df


def run_lstm(ticker, timeframe):
    use_hybrid = [True, False]
    threshold = 0.01
    figures = []
    transactions = []

    train_metrics_data = []
    val_metrics_data = []
    test_metrics_data = []

    for hybrid in use_hybrid:
        print(f'Ticker: {ticker}, Hybrid: {hybrid}')

        price_daily, vix_daily, usdx_daily, macro_data, sp_daily = fetch_data(ticker, timeframe)
        price_daily = price_daily.drop('Adj Close', axis = 1)

        data = add_indicators(price_daily)
        data_filters = apply_directional_filter(price_daily, vix_daily, usdx_daily, macro_data, sp_daily)

        model, scaler, x_train, y_train, x_val, y_val, x_test, y_test = train_lstm(
        ticker, data_filters, time_step=10, test_size=0.2, val_size=0.1, epochs=50, batch_size=32,  use_hybrid=hybrid
        )

        # Predictions for training, validation, and test sets
        predicted_train = predict_lstm(model, x_train, scaler)
        predicted_val = predict_lstm(model, x_val, scaler)
        predicted_test = predict_lstm(model, x_test, scaler)

        actual_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        actual_val = scaler.inverse_transform(y_val.reshape(-1, 1))
        actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate performance metrics for all sets
        print("Training Set Performance Metrics:")
        train_metrics = calculate_metrics(actual_train.flatten(), predicted_train.flatten())
        print(f"MAE: {train_metrics[0]:.5f}, MSE: {train_metrics[1]:.5f}, RMSE: {train_metrics[2]:.5f}")

        train_met = {
            'Ticker': ticker,
            'Hybrid': hybrid,
            'MAE': train_metrics[0],
            'MSE': train_metrics[1],
            'RMSE': train_metrics[2]
        }

        train_metrics_data.append(train_met)

        print("\nValidation Set Performance Metrics:")
        val_metrics = calculate_metrics(actual_val.flatten(), predicted_val.flatten())
        print(f"MAE: {val_metrics[0]:.5f}, MSE: {val_metrics[1]:.5f}, RMSE: {val_metrics[2]:.5f}")

        v_met = {
            'Ticker': ticker,
            'Hybrid': hybrid,
            'MAE': val_metrics[0],
            'MSE': val_metrics[1],
            'RMSE': val_metrics[2]
        }

        val_metrics_data.append(v_met)

        print("\nTest Set Performance Metrics:")
        test_metrics = calculate_metrics(actual_test.flatten(), predicted_test.flatten())
        print(f"MAE: {test_metrics[0]:.5f}, MSE: {test_metrics[1]:.5f}, RMSE: {test_metrics[2]:.5f}")

        test_met = {
            'Ticker': ticker,
            'Hybrid': hybrid,
            'MAE': test_metrics[0],
            'MSE': test_metrics[1],
            'RMSE': test_metrics[2]
        }

        test_metrics_data.append(test_met)

        # Create transaction table for test set
        transactions_df = create_transaction_table(actual_test.flatten(), predicted_test.flatten(), threshold)
        # print("\nTransaction DF:")
        # print(transactions_df)
        # transactions_df.to_csv(f'Transaction_df_{ticker}_hybrid_{hybrid}.csv')

        # Plot the predictions
        fig = plot_predictions(ticker, actual_train, predicted_train, actual_val, predicted_val, actual_test, predicted_test, hybrid)
        figures.append(fig)
        transactions.append(transactions_df)

    return figures, transactions