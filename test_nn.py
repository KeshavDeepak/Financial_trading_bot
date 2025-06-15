from lstm_neural_network import LSTMTradingAgent

import json

#* load up a saved agent
# model_name = "06 Jun - 00 34"
# model_name = "14 Jun - 21 17"
model_name = "14 Jun - 21 23"

# import the agent's saved info
with open("lstm_nn/model_metadata.json", "r") as file:
    agent_info = next((model for model in json.load(file) if model["name"] == model_name), None)

if not agent_info:
    raise Exception("Model not found")

#* initialize a trading agent instance with the agent_info parameters
agent = LSTMTradingAgent(look_back=agent_info['lookback'], train_test_split=agent_info['train_test_split'])

# load the saved model
agent.load_model(f"lstm_nn/{model_name}.h5", f"scalers/{model_name}.save")
df = agent.fetch_data(agent_info['ticker'], agent_info['start_date'], agent_info['end_date'])
agent.prepare_data()

#* backtest
portfolio_value, trades = agent.backtest(initial_balance=agent_info['initial_balance'])

#* plot backtest results
agent.plot_backtest_results(portfolio_value)

'''
6th june - 00 34 (3) -- kind of good
14th june -- 21 17 (10) -- horrendous
14th june - 21 23 (68 [early stopped])-- best!
'''
