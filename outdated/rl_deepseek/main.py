import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from environment import TradingEnv
from agent import DQNAgent

def prepare_data():
    # Example with synthetic data
    dates = pd.date_range(start='2020-01-01', end='2023-01-01')
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    return df

def train_agent():
    # Prepare data
    df = prepare_data()
    
    # Initialize environment and agent
    env = TradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    batch_size = 32
    episodes = 100
    
    # Training loop
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        for time in range(len(df)-1):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Portfolio Value: {info['portfolio_value']:.2f}, Trades: {info['total_trades']}")
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
        # monitor progress
        print(f"Episode {e}: ε={agent.epsilon:.2f}, Portfolio=${info['portfolio_value']:.2f}")
        
    
    # Save the trained model
    agent.save("trading_dqn.h5")

if __name__ == "__main__":
    train_agent()
    