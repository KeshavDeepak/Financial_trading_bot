import json
import matplotlib.pyplot as plt

import yfinance as yf

from backend.lstm_nn_3 import LSTMTradingAgent

#* use the agent to get a prediction of the next day
def predict_next_day(ticker, model_name):
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
    
    #* fetch the ticker's data to predict on (last [lookback] timesteps)
    #* -- get more than [lookback] days and slice later
    stock_df = yf.download(ticker, period=f"{3*agent_info['lookback']}d", interval='1d')
    #* -- get the required features for the last [lookback] days 
    stock_df = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(agent_info['lookback']).values 
    
    #* predict the next day's direction 
    return agent.predict_direction(stock_df)

# visualize the stock line of a particular ticker and timeline
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