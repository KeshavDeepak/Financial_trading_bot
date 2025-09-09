'''
this tests the third variant of the neural network model in which the predictions are up/down/neutral
'''

from backend.lstm_nn_3 import LSTMTradingAgent

import yfinance as yf
import numpy as np

import json

#* load up a saved agent
model_name = "05 Aug - 17 57"

#* import the agent's saved info
with open("./backend/lstm_nn/model_metadata.json", "r") as file:
    agent_info = next((model for model in json.load(file) if model["name"] == model_name), None)

if not agent_info:
    raise Exception("Model not found")  

#* initialize a trading agent instance with the agent_info parameters
agent = LSTMTradingAgent(look_back=agent_info['lookback'], train_test_split=agent_info['train_test_split'])

# load the saved model
agent.load_model(f"./backend/lstm_nn/{model_name}.h5", f"./backend/scalers/{model_name}.save")
df = agent.fetch_data(agent_info['ticker'], agent_info['start_date'], agent_info['end_date'], agent_info['interval'])
agent.prepare_data()

#* view the training and validation losses
agent.plot_training_history(agent_info['history'])

#* backtest
portfolio_value, trades = agent.backtest(initial_balance=agent_info['initial_balance'])

#* plot backtest results
agent.plot_backtest_results(portfolio_value)

#* plot trade signals on the stock line
agent.plot_trade_signals(trades)

#* evaluate the model on its own (without any trading strategy)
# Fetch test data
test_df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

test_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Store predictions and true values
predictions = []
true_values = []

# Rolling window prediction for the year
window_size = agent_info["lookback"] # look_back period
step = 1 # move forward 1 day at a time

for i in range(window_size, len(test_data) - 1):
    # Get the last [lookback] days of data
    input_data = test_data[i-window_size:i]
    
    # Predict if price is going to go up or down or will stay the same
    pred = np.argmax(agent.predict_direction(input_data)[1])
    predictions.append(pred)
    
    # Store true price change -- match this to its original in the lstm_nn_3 bot
    current_close = test_data[i][agent.close_index]
    next_close = test_data[i + 1][agent.close_index]
    
    # 0 is down, 1 is neutral, 2 is up
    threshold = 0.01
    
    change = (next_close - current_close) / current_close
    
    if change > threshold:
        label = 2
    elif change < -threshold:
        label = 0
    else:
        label = 1
    
    true_values.append(label)

# directional accuracy
print(predictions, true_values)
accuracy = np.mean(np.array(true_values) == np.array(predictions)) * 100

print(f"Directional Accuracy: {accuracy:.2f}%")

# print trade summary
agent.print_trade_summary(trades, portfolio_value, agent_info["initial_balance"], agent_info["ticker"], agent_info["start_date"], agent_info["end_date"])

'''
05 Aug - 17 57 (100 epochs no early stopping) -- 
    with the 3rd lstm bot, does decent but still no convergence and validation loss seems to be increasing for some reason
'''
