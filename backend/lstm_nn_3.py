'''
this one is a variant of the second lstm model which predicts three classes - 
0 = down, 
1 = neutral (no price change within a threshold), 
2 = up
'''

import joblib
import json

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import yfinance as yf


class LSTMTradingAgent:
    def __init__(self, look_back=60, train_test_split=0.8):
        """
        Initialize the LSTM Trading Agent
        
        Parameters:
        - look_back: Number of previous time steps to use for prediction
        - train_test_split: Ratio of training to testing data
        """
        self.look_back = look_back
        self.train_test_split = train_test_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_data(self, ticker, start_date, end_date, interval='1d'):
        """
        Fetch data from Yahoo Finance
        
        Parameters:
        - ticker: Stock ticker symbol (e.g., 'AAPL')
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format
        - interval: Data interval ('1d', '1h', etc.)
        
        Returns:
        - DataFrame with stock data
        """
        
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}. Check the ticker symbol and date range.")
            
        self.data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.dates = data.index
        
        # keep track of the index the closing prices sit in for later use 
        self.close_index = 3
        
        # Normalize the data
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        return data
    
    def create_dataset(self, data):
        """
        Create the dataset for LSTM training
        
        Parameters:
        - data: The input time series data
        
        Returns:
        - X, y: Features and targets for the LSTM
        """
        X, y = [], []
        
        for i in range(self.look_back, len(data) - 1):
            X.append(data[i-self.look_back:i, :])
            
            # target to predict is one of three classes -- down (0), neutral (1), up (2)
            current_close = data[i][self.close_index]
            next_close = data[i + 1][self.close_index]
            
            # 0 is down, 1 is neutral, 2 is up
            threshold = 0.01
            
            change = (next_close - current_close) / current_close
            
            if change > threshold:
                label = 2
            elif change < -threshold:
                label = 0
            else:
                label = 1
            
            y.append(label)
            
        return np.array(X), np.array(y)
    
    def prepare_data(self, ):
        """
        Prepare the training and testing datasets
        """
        # Create the dataset
        X, y = self.create_dataset(self.scaled_data)
        
        # Split into train and test sets
        train_size = int(len(X) * self.train_test_split)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM [samples, time steps, features]
        # self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 5))
        # self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 5))
        
    def build_model(self):
        """
        Build the LSTM model architecture
        """
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=3, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    def train_model(self, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Parameters:
        - epochs: Number of training epochs
        - batch_size: Size of training batches
        """
        # early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        
        # fit the model onto the training data
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            # callbacks=[early_stopping],
            verbose=1
        )
    
    def plot_training_history(self, history=None):
        history_data = history if history else self.history.history
        
        """Plot the training and validation loss"""
        plt.figure(figsize=(10, 6))
        # plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(history_data['loss'], label='Training Loss')
        # plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.plot(history_data['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict_direction(self, input_data):
        # input_data shape: (look_back, 5)
        input_data = np.array(input_data).reshape(-1, 5)
        
        scaled_input = self.scaler.transform(input_data)
        
        X = scaled_input.reshape(1, self.look_back, 5)
        
        prob = self.model.predict(X)[0]
        predicted_class = np.argmax(prob)
        mapping = {0 : 'down', 1 : 'neutral', 2 : 'up'}
        
        return mapping[predicted_class], prob
    
    
    def backtest(self, initial_balance=10000):
        """
        Backtest the trading strategy
        
        Parameters:
        - initial_balance: Starting capital for backtesting
        
        Returns:
        - portfolio_value: Array of portfolio values over time
        - trades: Array of trades made
        """
        # Prepare test data
        X_test, _ = self.create_dataset(self.scaled_data)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
        
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Plot a histogram showing the frequency distributions of the predictions (from 0 to 1) of the model
        plt.hist(predicted_classes, bins=[0,1,2,3], align='left', rwidth=0.8)
        plt.xticks([0,1,2], ['Down','Neutral','Up'])
        plt.show()

        # Initialize trading variables
        balance = initial_balance
        position = 0
        portfolio_value = []
        trades = []
        
        # buy all if price is predicted to go up -- sell all if price is predicted to go down
        for i in range(len(predicted_classes)):
            current_close = self.data[self.look_back + i][self.close_index]
            
            predicted_class = predicted_classes[i]
            # confidence = np.max(predictions[i])

            # buy signal -- if price is going to go up -- buy as many shares as possible
            if predicted_class == 2 and position == 0:
                position = balance / current_close
                balance = 0
                trades.append(('buy', current_close, str(self.dates[self.look_back + i])))

            # sell signal -- if price is going to go down -- sell all the shares and convert to cash 
            elif predicted_class == 0 and position > 0:
                balance = position * current_close
                position = 0
                trades.append(('sell', current_close, str(self.dates[self.look_back + i])))

            if position > 0:
                portfolio_value.append(position * current_close)
            else:
                portfolio_value.append(balance)

        return portfolio_value, trades
    
    def plot_backtest_results(self, portfolio_value):
        """Plot the backtesting results"""
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_value, label='Portfolio Value')
        plt.title('Backtesting Results')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.show()
    
    
    def plot_trade_signals(self, trades):
        """
        Plot the stock price with buy/sell markers based on executed trades.
        
        Parameters:
        - trades: List of tuples (action, price, date)
        """
        close_prices = [row[self.close_index] for row in self.data]  # Extract close prices
        dates = self.dates[:len(close_prices)]       # Match date range

        plt.figure(figsize=(14, 7))
        plt.plot(dates, close_prices, label='Close Price', color='blue', alpha=0.6)

        buy_dates = [pd.to_datetime(t[2]) for t in trades if t[0] == 'buy']
        buy_prices = [t[1] for t in trades if t[0] == 'buy']
        sell_dates = [pd.to_datetime(t[2]) for t in trades if t[0] == 'sell']
        sell_prices = [t[1] for t in trades if t[0] == 'sell']

        # Plot buy signals
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy', s=100)

        # Plot sell signals
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell', s=100)

        plt.title('Stock Price with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    def print_trade_summary(self, trades, portfolio_value, initial_balance, ticker, start_date, end_date):
        """Print a summary of the trading performance"""
        if not trades:
            print("No trades were executed during the backtest period.")
            return
            
        winning_trades = 0
        losing_trades = 0
        buy_prices = []
        sell_prices = []
        
        for i in range(len(trades)-1):
            if trades[i][0] == 'buy' and trades[i+1][0] == 'sell':
                buy_price = trades[i][1]
                sell_price = trades[i+1][1]
                buy_prices.append(buy_price)
                sell_prices.append(sell_price)
                if sell_price > buy_price:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        print("\n=== Trading Summary ===")
        print(f"Ticker: {ticker}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Initial balance: ${initial_balance:,.2f}")
        print(f"Final balance: ${portfolio_value[-1]:,.2f}")
        print(f"Profit/Loss: {((portfolio_value[-1]/initial_balance)-1)*100:.2f}%")
        print(f"\nTotal trades executed: {len(trades)//2} round-trip trades")
        print(f"Winning trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing trades: {losing_trades}")
        
        if winning_trades > 0 and losing_trades > 0:
            avg_win = (sum(sell_prices) - sum(buy_prices)) / winning_trades
            avg_loss = (sum(buy_prices) - sum(sell_prices)) / losing_trades
            print(f"\nAverage win: ${avg_win:.2f} per share")
            print(f"Average loss: ${avg_loss:.2f} per share")
            print(f"Risk/Reward ratio: {abs(avg_win/avg_loss):.2f}:1")
    
    def evaluate_model(self):
        preds = self.model.predict(self.X_test)
        y_pred = np.argmax(preds, axis=1)
        
        print(classification_report(self.y_test, y_pred, target_names=['Down', 'Neutral', 'Up']))
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.show()
    
    def save_model(self, model_path, scaler_path, current_time, 
                   ticker, start_date, end_date, initial_balance, epochs, batch_size, interval):
        """Save model and scaler to disk"""        
        # save the model
        self.model.save(model_path)
        
        # save the scaler
        joblib.dump(self.scaler, scaler_path)
        
        # output confirmation message
        print(f"Model saved to {model_path}, scaler to {scaler_path}")
        
        # save the important hyperparameters 
        with open('./backend/lstm_nn/model_metadata.json', 'r+') as file:
            models = json.load(file)
            
            models.append({
                'name' : current_time,
                'lookback' : self.look_back,
                'train_test_split' : self.train_test_split,
                'ticker' : ticker,
                'start_date' : start_date,
                'end_date' : end_date,
                'initial_balance' : initial_balance,
                'epochs' : epochs,
                'batch_size' : batch_size,
                'interval' : interval,
                'history' : self.history.history
            })
            
            file.seek(0)
            
            json.dump(models, file, indent=4)
        

    def load_model(self, model_path='./lstm_nn/trading_model.h5', scaler_path='./scalers/scaler.save'):
        """Load model and scaler from disk"""
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}, scaler from {scaler_path}")