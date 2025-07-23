'''
this tests the second variant of the neural network model in which the predictions are binary (up/down)
'''

from build_nn import LSTMTradingAgent

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import json

#* load up a saved agent
model_name = "23 Jul - 21 24"
# model_name = "24 Jul - 02 39"

# import the agent's saved info
with open("lstm_nn/model_metadata.json", "r") as file:
    agent_info = next((model for model in json.load(file) if model["name"] == model_name), None)

if not agent_info:
    raise Exception("Model not found")  

#* initialize a trading agent instance with the agent_info parameters
agent = LSTMTradingAgent(look_back=agent_info['lookback'], train_test_split=agent_info['train_test_split'])

# load the saved model
agent.load_model(f"lstm_nn/{model_name}.h5", f"scalers/{model_name}.save")
df = agent.fetch_data(agent_info['ticker'], agent_info['start_date'], agent_info['end_date'], agent_info['interval'])
agent.prepare_data()

#* backtest
portfolio_value, trades = agent.backtest(initial_balance=agent_info['initial_balance'])

#* plot backtest results
agent.plot_backtest_results(portfolio_value)

agent.plot_trade_signals(trades)

#* evaluate the model on its own (without any trading strategy)
# Fetch test data
test_df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
test_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
dates = test_df.index

# Store predictions and true values
predictions = []
true_values = []

# rollow window prediction for the year
window_size = agent_info["lookback"]  # look_back period
step = 1          # move forward 1 day at a time

for i in range(window_size, len(test_data) - 1):
    # Get the last [lookback] days of data
    input_data = test_data[i-window_size:i]
    
    # Predict if price is going to go up or down
    pred = agent.predict_direction(input_data)
    predictions.append(pred)
    
    # Store true price change (binary)
    true_values.append(1 if test_data[i + 1][3] > test_data[i][3] else 0)

# directional accuracy
accuracy = np.mean(np.array(true_values) == np.array(predictions)) * 100

print(f"Directional Accuracy: {accuracy:.2f}%")

# print trade summary
agent.print_trade_summary(trades, portfolio_value, agent_info["initial_balance"], agent_info["ticker"], agent_info["start_date"], agent_info["end_date"])

'''
6th june - 00 34 (3) -- kind of good
14th june -- 21 17 (10) -- horrendous
14th june - 21 23 (68 [early stopped])-- best!

23 Jul - 21 24 (3 epochs) -- pretty decent actually
24 Jul - 02 39 (13 epochs ES) -- 
'''
