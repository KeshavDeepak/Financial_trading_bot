from build_nn import LSTMTradingAgent

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import json

#* load up a saved agent
# model_name = "06 Jun - 00 34"
# model_name = "14 Jun - 21 17"
# model_name = "14 Jun - 21 23"
model_name = "23 Jul - 21 24"

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

for i in range(window_size, len(test_data)):
    # Get the last [lookback] days of data
    input_data = test_data[i-window_size:i]
    
    # Preprocess for prediction
    scaled_data = agent.scaler.transform(input_data)
    X = scaled_data.reshape(1, window_size, 5)  # Reshape to [1, [lookback], 5]
    
    # Predict next day's price
    pred = agent.predict(X)
    predictions.append(pred)
    
    # Store true value (next day's close)
    true_values.append(float(test_data[i]))

# calculate metrics
mae = mean_absolute_error(true_values, predictions)
# rmse = np.sqrt(mean_squared_error(true_values, predictions))

# directional accuracy
true_values_numpy = np.array(true_values)
predictions_numpy = np.array(predictions)

true_directions = np.sign(true_values_numpy[1:] - true_values_numpy[:-1])  # True price changes (Δ)
pred_directions = np.sign(predictions_numpy[1:] - true_values_numpy[:-1])  # Predicted changes (ŷ - y_prev)

correct_directions = (true_directions == pred_directions)
directional_accuracy = np.mean(correct_directions) * 100


# theils_u
def theils_u(y_true, y_pred):
    """Calculate Theil's U statistic."""
    # Compute RMSE of your model
    rmse_model = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Naive forecast: y_hat = y_prev (shift true values by 1)
    naive_pred = np.roll(y_true, shift=1)  # Shift array by 1
    naive_pred[0] = np.nan  # Remove first NaN (no prediction)
    naive_pred = naive_pred[1:]  # Align with y_true[1:]
    y_true_aligned = y_true[1:]
    
    # Compute RMSE of naive model
    rmse_naive = np.sqrt(np.mean((y_true_aligned - naive_pred) ** 2))
    
    # Avoid division by zero
    if rmse_naive == 0:
        return np.inf
    
    return rmse_model / rmse_naive

theilu = theils_u(true_values_numpy, predictions_numpy)
print(f"Theil's U: {theilu:.4f}")

if theilu < 1:
    print("Model beats naive forecast!")
elif theilu == 1:
    print("Model = naive forecast.")
else:
    print("Model worse than naive forecast.")

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

'''
6th june - 00 34 (3) -- kind of good
14th june -- 21 17 (10) -- horrendous
14th june - 21 23 (68 [early stopped])-- best!
'''
