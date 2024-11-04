

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ta.trend import EMAIndicator, ADXIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input, Concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings

# Set a random seed for reproducibility
np.random.seed(42)

# Suppress warnings
warnings.filterwarnings("ignore")
# Prepare data for LSTM
def prepare_data(data, feature_col='Close', time_step=1, test_size=0.2, val_size=0.1):
    if feature_col not in data.columns:
        raise ValueError(f"Column '{feature_col}' not found in the dataset")

    data = data[[feature_col]].values
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
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Prepare training data
    x_train, y_train = [], []
    for i in range(len(train_data_scaled) - time_step):
        x_train.append(train_data_scaled[i:(i + time_step), 0])
        y_train.append(train_data_scaled[i + time_step, 0])

    # Prepare validation data
    x_val, y_val = [], []
    for i in range(len(val_data_scaled) - time_step):
        x_val.append(val_data_scaled[i:(i + time_step), 0])
        y_val.append(val_data_scaled[i + time_step, 0])

    # Prepare testing data
    x_test, y_test = [], []
    for i in range(len(test_data_scaled) - time_step):
        x_test.append(test_data_scaled[i:(i + time_step), 0])
        y_test.append(test_data_scaled[i + time_step, 0])

    x_train = np.array(x_train).reshape(-1, time_step, 1)
    x_val = np.array(x_val).reshape(-1, time_step, 1)
    x_test = np.array(x_test).reshape(-1, time_step, 1)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler

# Build LSTM Model
def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train LSTM Model
def train_lstm(data, feature_col='Close', time_step=10, test_size=0.2, val_size=0.1, epochs=50, batch_size=32):
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = prepare_data(
        data, feature_col=feature_col, time_step=time_step, test_size=test_size, val_size=val_size
    )

    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

    return model, scaler, x_train, y_train, x_val, y_val, x_test, y_test

# Predict using the trained LSTM model
def predict_lstm(model, x_data_scaled, scaler):
    predictions_scaled = model.predict(x_data_scaled)
    predictions = scaler.inverse_transform(predictions_scaled)
    return predictions

# Plot actual vs predicted values for train, validation, and test sets
def plot_predictions(actual_train, predicted_train, actual_val, predicted_val, actual_test, predicted_test):
    plt.figure(figsize=(14, 7))

    # Plot training data
    plt.subplot(3, 1, 1)
    plt.plot(actual_train, color='blue', label='Actual Train Values', alpha=0.6)
    plt.plot(predicted_train, color='red', label='Predicted Train Values', alpha=0.6)
    plt.title('Train: Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    # Plot validation data
    plt.subplot(3, 1, 2)
    plt.plot(actual_val, color='blue', label='Actual Validation Values', alpha=0.6)
    plt.plot(predicted_val, color='red', label='Predicted Validation Values', alpha=0.6)
    plt.title('Validation: Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    # Plot testing data
    plt.subplot(3, 1, 3)
    plt.plot(actual_test, color='blue', label='Actual Test Values', alpha=0.6)
    plt.plot(predicted_test, color='red', label='Predicted Test Values', alpha=0.6)
    plt.title('Test: Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

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

# Main function for handling multiple currency pairs
# def main(data_dict, feature_col='Close', time_step=10, test_size=0.2, val_size=0.1, epochs=50, batch_size=32, threshold=0.01):
#     for pair, data in data_dict.items():
#         print(f"\nTraining LSTM model for {pair}...")
#
#         model, scaler, x_train, y_train, x_val, y_val, x_test, y_test = train_lstm(
#             data, feature_col=feature_col, time_step=time_step, test_size=test_size, val_size=val_size, epochs=epochs, batch_size=batch_size
#         )
#
#         # Predictions for training, validation, and test sets
#         predicted_train = predict_lstm(model, x_train, scaler)
#         predicted_val = predict_lstm(model, x_val, scaler)
#         predicted_test = predict_lstm(model, x_test, scaler)
#
#         actual_train = scaler.inverse_transform(y_train.reshape(-1, 1))
#         actual_val = scaler.inverse_transform(y_val.reshape(-1, 1))
#         actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))
#
#         # Calculate performance metrics for all sets
#         print("Training Set Performance Metrics:")
#         train_metrics = calculate_metrics(actual_train.flatten(), predicted_train.flatten())
#         print(f"MAE: {train_metrics[0]:.5f}, MSE: {train_metrics[1]:.5f}, RMSE: {train_metrics[2]:.5f}")
#
#         print("\nValidation Set Performance Metrics:")
#         val_metrics = calculate_metrics(actual_val.flatten(), predicted_val.flatten())
#         print(f"MAE: {val_metrics[0]:.5f}, MSE: {val_metrics[1]:.5f}, RMSE: {val_metrics[2]:.5f}")
#
#         print("\nTest Set Performance Metrics:")
#         test_metrics = calculate_metrics(actual_test.flatten(), predicted_test.flatten())
#         print(f"MAE: {test_metrics[0]:.5f}, MSE: {test_metrics[1]:.5f}, RMSE: {test_metrics[2]:.5f}")
#
#         # Create transaction table for test set
#         transactions_df = create_transaction_table(actual_test.flatten(), predicted_test.flatten(), threshold)
#
#         # Print transaction table
#         print("Transaction Table:")
#         print(transactions_df)
#
#         # Plot the predictions
#         plot_predictions(actual_train, predicted_train, actual_val, predicted_val, actual_test, predicted_test)
#
#
# # Example of running the main function with multiple currency pairs
# # Assuming you have a dictionary of DataFrames for each currency pair
# data_dict = {
#     'EUR/USD': eurusd_daily,
#     'GBP/USD': gbpusd_daily,
#     'JPY/USD': jpyusd_daily,
#     'AUD/USD': audusd_daily,
#     'USD/CAD': usdcad_daily,
#     'NZD/USD': nzdusd_daily,
#     'CHF/USD': chfusd_daily,
#     'EUR/GBP': eurgbp_daily,
#     'EUR/JPY': eurjpy_daily,
#     'GBP/JPY': gbpjpy_daily,
#     # Add more currency pairs as needed
# }
#
# main(data_dict, feature_col='Close', time_step=10, test_size=0.2, val_size=0.1, epochs=50, batch_size=32, threshold=0.01)