import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple

class TradingEnv(gym.Env):
    """A custom trading environment for OpenAI Gym"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)  # Reset index to use integer-based indexing
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: [balance, shares_held, current_price, ...other features]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_profit = 0
        self.total_trades = 0
        
        return self._next_observation()
    
    def _next_observation(self) -> np.ndarray:
        """Get the next observation from the data"""
        obs = np.array([
            self.balance,
            self.shares_held,
            self._get_current_price(),
            self._get_current_price() * self.shares_held,  # Portfolio value
            self.total_profit,
            self.total_trades
        ], dtype=np.float32)
        return obs
    
    def _get_current_price(self) -> float:
        """Get current price from the data"""
        return float(self.df.loc[self.current_step, 'close'])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment"""
        current_price = self._get_current_price()
        reward = 0
        done = False
        info = {}
        
        # Action logic
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.total_trades += 1
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                self.total_trades += 1
                reward = (current_price - self._get_avg_buy_price()) * self.shares_held
                self.total_profit += reward
        
        # Update step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        
        # Calculate reward if not already set (for hold action)
        if action == 0:
            reward = 0
        
        # Update portfolio value
        portfolio_value = self.balance + (self.shares_held * current_price)
        
        # Additional info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': portfolio_value,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades
        }
        
        return self._next_observation(), reward, done, info
    
    def _get_avg_buy_price(self) -> float:
        """Calculate average buy price (simplified)"""
        # In a real implementation, you'd track buy prices
        return self._get_current_price()
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass