from environment import TradingEnv
from agent import DQNAgent
from main import prepare_data

def evaluate_agent():
    # Prepare data (use different data than training)
    df = prepare_data()  # In practice, use test data here
    
    # Initialize environment and agent
    env = TradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Load trained weights
    agent.load("trading_dqn.h5")
    
    # Evaluation parameters
    agent.epsilon = 0  # Disable exploration
    
    # Run evaluation
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    
    portfolio_values = []
    
    while not done:
        action = agent.act(state)
        next_state, _, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        portfolio_values.append(info['portfolio_value'])
        
        if done:
            print(f"Final Portfolio Value: {info['portfolio_value']:.2f}")
            print(f"Total Profit: {info['total_profit']:.2f}")
            print(f"Total Trades: {info['total_trades']}")
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.show()

# Run evaluation
evaluate_agent()