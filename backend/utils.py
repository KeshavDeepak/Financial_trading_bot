import io
import base64

import json
import joblib

import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

from backend.lstm_nn_3 import LSTMTradingAgent

#* ------------------------------------------------ helper functions --------------------------------------------------

#* load up a LSTMTradingAgent object with a saved agent's metadata and weights
def load_saved_agent(model_name):
    #* import the agent model's metadata in
    with open("./backend/lstm_nn/model_metadata.json", "r") as file:
        agent_info = next((model for model in json.load(file) if model["name"] == model_name), None)

    if not agent_info:
        raise Exception("Model not found") 


    #* initialize a trading agent with the saved agent's parameters
    agent = LSTMTradingAgent(
        look_back=agent_info['lookback'], 
        train_test_split=agent_info['train_test_split']
    )

    #* load up the saved agent's weights into the trading agent
    agent.load_model(
        f"./backend/lstm_nn/{model_name}.h5", 
        f"./backend/scalers/{model_name}.save"
    )
    
    #* return the agent and its metadata back
    return agent, agent_info

#* visualize the stock line of a particular ticker and timeline
def visualize(ticker, start, end):
    """
    Fetches and visualizes the closing price of a stock over a given time period.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    """
    # Download historical data
    data = yf.download(ticker, start=start, end=end)

    # Check if data was returned
    if data.empty:
        print(f"No data found for {ticker} between {start} and {end}.")
        return

    # Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f"{ticker} Stock Price from {start} to {end}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#* ------------------------------------------------ main functions --------------------------------------------------

#* use the agent to get a prediction of the next day
def predict_next_day(ticker, model_name):
    #* load up the saved agent
    agent, agent_info = load_saved_agent(model_name)
    
    #* fetch the ticker's data to predict on (last [lookback] timesteps)
    #* -- get more than [lookback] days and slice later
    stock_df = yf.download(ticker, period=f"{3*agent_info['lookback']}d", interval='1d')
    #* -- get the required features for the last [lookback] days 
    stock_df = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(agent_info['lookback']).values 
    
    #* predict the next day's direction (scaling is baked into .predict_direction())
    return agent.predict_direction(stock_df)


#* backtest a ticker on a given model
def backtest(ticker, model_name, look_back=60, initial_balance=10000, close_index=3):
    """
    Backtest an LSTM model on historical stock data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol
        model_name (str): Path to trained LSTM model (.h5)
        look_back (int): Number of past timesteps the model expects
        initial_balance (float): Starting cash for backtest
        close_index (int): Index of 'Close' in OHLCV data (default=3)

    Returns:
        portfolio_values (list): Portfolio value progression over time
        trades (list): List of executed trades (buy/sell events)
    """    
    #* load up the saved agent
    agent, agent_info = load_saved_agent(model_name)

    #* Load saved scaler
    scaler_path = f"./backend/scalers/{model_name}.save"
    scaler = joblib.load(scaler_path)
    
    #* Download historical data
    stock_df = yf.download(ticker, period="1y")
    data = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    dates = stock_df.index
    
    #* Scale data using the saved scaler
    scaled_data = scaler.transform(data)

    #* Create dataset for backtesting
    dataset = []
    
    for i in range(look_back, len(scaled_data)):
        dataset.append(scaled_data[i-look_back:i])
        
    dataset = np.array(dataset)

    #* Get predictions
    predictions = agent.model.predict(dataset)
    predicted_classes = np.argmax(predictions, axis=1)

    #* Plot histogram of predictions
    # plt.hist(predicted_classes, bins=[0,1,2,3], align='left', rwidth=0.8)
    # plt.xticks([0,1,2], ['Down','Neutral','Up'])
    # plt.show()

    #* Trading simulation
    balance = initial_balance
    position = 0
    portfolio_values = []
    trades = []

    #* simple trading strategy using buy and sell signals
    for i in range(len(predicted_classes)):
        current_close = data[look_back + i][close_index]
        predicted_class = predicted_classes[i]

        #* if price is going up -- buy all
        if predicted_class == 2 and position == 0: 
            position = balance / current_close
            balance = 0
            trades.append(('buy', current_close, str(dates[look_back + i])))

        #* if price is going down -- sell all
        elif predicted_class == 0 and position > 0: 
            balance = position * current_close
            position = 0
            trades.append(('sell', current_close, str(dates[look_back + i])))

        #* Track portfolio value
        if position > 0:
            portfolio_values.append(position * current_close)
        else:
            portfolio_values.append(balance)

    return portfolio_values, trades

#* plot the backtesting results given an input portfolio timeline
def portfolio_timeline_plot(portfolio_values):
    """Plot the backtesting results"""
    
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Backtesting Results')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    
    #* save the plot so it can be returned
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close() #* to avoid errors
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    #* return the plot
    return img_base64
        
    