import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

import yfinance as yf


class TransformerTradingAgent:
    def __init__(self, look_back=60, train_test_split=0.8):
        """
        Transformer-based Trading Agent for regression (predicts next day's price)
        """
        self.look_back = look_back
        self.train_test_split = train_test_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.close_index = 3  # column index of "Close" in data
        
    def fetch_data(self, ticker, start_date, end_date, interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")
            
        self.data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.dates = data.index
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        return data
    
    def create_dataset(self, data):
        """Create dataset for regression (next-day closing price prediction)."""
        X, y = [], []
        for i in range(self.look_back, len(data) - 1):
            X.append(data[i-self.look_back:i, :])  # features
            y.append(data[i + 1][self.close_index])  # next day's close (scaled)
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Split into train/test sets."""
        X, y = self.create_dataset(self.scaled_data)
        
        train_size = int(len(X) * self.train_test_split)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
    
    def build_model(self, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.2):
        """
        Build Transformer model for regression.
        """
        inputs = Input(shape=(self.look_back, self.X_train.shape[2]))
        
        # Multi-Head Attention block
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feed Forward Network
        ffn = Dense(ff_dim, activation="relu")(out1)
        ffn = Dense(d_model)(ffn)
        ffn_output = Dropout(dropout_rate)(ffn)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        # Global pooling
        x = GlobalAveragePooling1D()(out2)
        
        # Final regression output
        outputs = Dense(1, activation="linear")(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the Transformer model."""
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=1
        )
    
    def plot_training_history(self):
        """Plot training/validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label="Training Loss")
        plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.title("Training History (MSE)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def predict_prices(self):
        """Predict prices for test set and inverse transform back to real values."""
        preds = self.model.predict(self.X_test)
        
        # Rebuild full feature vectors for inverse scaling
        y_pred_full = np.zeros((len(preds), self.data.shape[1]))
        y_true_full = np.zeros((len(self.y_test), self.data.shape[1]))
        
        y_pred_full[:, self.close_index] = preds.flatten()
        y_true_full[:, self.close_index] = self.y_test.flatten()
        
        y_pred = self.scaler.inverse_transform(y_pred_full)[:, self.close_index]
        y_true = self.scaler.inverse_transform(y_true_full)[:, self.close_index]
        
        return y_true, y_pred
    
    def evaluate_model(self):
        """Evaluate regression performance."""
        y_true, y_pred = self.predict_prices()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("\n=== Model Evaluation ===")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²:  {r2:.4f}")
        
        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual Price", color="blue")
        plt.plot(y_pred, label="Predicted Price", color="red", alpha=0.7)
        plt.title("Actual vs Predicted Stock Prices")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        
        return mse, mae, r2
    
    def backtest(self, initial_balance=10000):
        """
        Backtest a simple strategy:
        - Buy if predicted next-day price > today's price
        - Sell if predicted price < today's price
        """
        y_true, y_pred = self.predict_prices()
        
        balance = initial_balance
        position = 0
        portfolio_value = []
        trades = []
        
        for i in range(len(y_pred) - 1):
            current_price = y_true[i]
            predicted_price = y_pred[i]
            
            if predicted_price > current_price and position == 0:
                position = balance / current_price
                balance = 0
                trades.append(("buy", current_price, str(self.dates[self.look_back + i])))
            
            elif predicted_price < current_price and position > 0:
                balance = position * current_price
                position = 0
                trades.append(("sell", current_price, str(self.dates[self.look_back + i])))
            
            if position > 0:
                portfolio_value.append(position * current_price)
            else:
                portfolio_value.append(balance)
        
        # Plot backtest results
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_value, label="Portfolio Value")
        plt.title("Backtesting Results (Transformer Strategy)")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()
        
        return portfolio_value, trades
