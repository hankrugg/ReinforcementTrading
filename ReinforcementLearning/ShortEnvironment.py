import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data, close_prices, window_size):
        super(TradingEnv, self).__init__()
        self.data = np.array(data)
        self.prices = np.array(close_prices)
        self.window_size = window_size
        self.current_step = window_size  # Start at window_size to have enough data for the first window
        self.max_step = len(self.data) - 1

        print(len(self.data), len(self.prices))

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Sell/Short, 2: Hold
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.data.shape[1],), dtype=np.float32
        )

        # Initialize trading variables
        self.balance = 25000  # Initial balance
        self.stock_count = 0  # Number of stocks held (positive for long, 0 for no position)
        self.short_stock_count = 0  # Number of stocks held short (negative for short positions)
        self.buy_price = 0  # Price at which stock was bought
        self.short_price = 0  # Price at which stock was shorted
        self.balance_history = []  # Track balance history
        self.buy_indices = []  # Track buy actions (time step, price)
        self.sell_indices = []  # Track sell actions (time step, price)
        self.short_sell_indices = []  # Track short sell actions (time step, price)
        self.short_cover_indices = []  # Track short cover actions (time step, price)

        # Initialize state
        self.state = None

    def reset(self):
        self.current_step = self.window_size
        self.balance = 25000
        self.stock_count = 0
        self.short_stock_count = 0
        self.buy_price = 0
        self.short_price = 0
        self.balance_history = []
        self.buy_indices = []  # Reset buy indices when environment is reset
        self.sell_indices = []  # Reset sell indices when environment is reset
        self.short_sell_indices = []  # Reset short sell indices
        self.short_cover_indices = []  # Reset short cover indices
        self.state = self.data[self.current_step]
        return self.state

    def step(self, action):
        done = False
        reward = 0

        # Current price is the price at the current_step
        current_price = self.prices[self.current_step]

        # Implement action logic
        if action == 0:  # Buy
            if self.short_stock_count > 0:  # Cover short position if currently short
                self.balance -= self.short_stock_count * current_price
                profit = (self.short_price - current_price) * self.short_stock_count
                reward = profit / self.short_price  # Normalized profit from covering
                self.short_stock_count = 0
                self.short_cover_indices.append((self.current_step, current_price))  # Track short cover
            elif self.stock_count == 0:  # Only buy if not holding any stock
                num_stocks = self.balance // current_price
                if num_stocks > 0:
                    self.stock_count = num_stocks
                    self.balance -= self.stock_count * current_price
                    self.buy_price = current_price
                    self.buy_indices.append((self.current_step, current_price))  # Track buy

        elif action == 1:  # Sell/Short
            if self.stock_count > 0:  # Sell if holding long stock
                self.balance += self.stock_count * current_price
                profit = (current_price - self.buy_price) * self.stock_count
                reward = profit / self.buy_price  # Normalized ROI from selling
                self.stock_count = 0
                self.sell_indices.append((self.current_step, current_price))  # Track sell
            elif self.stock_count == 0 and self.short_stock_count == 0:  # Short if not holding any stock
                num_stocks = self.balance // current_price
                if num_stocks > 0:
                    self.short_stock_count = num_stocks
                    self.balance += self.short_stock_count * current_price
                    self.short_price = current_price
                    self.short_sell_indices.append((self.current_step, current_price))  # Track short sell

        elif action == 2:  # Hold
            if self.stock_count > 0:
                unrealized_gain = (current_price - self.buy_price) * self.stock_count
                reward = unrealized_gain / self.buy_price  # Normalized unrealized ROI
            elif self.short_stock_count > 0:
                unrealized_loss = (self.short_price - current_price) * self.short_stock_count
                reward = unrealized_loss / self.short_price  # Normalized unrealized short ROI
            reward *= 0  # Trying with no reward on the hold

        # Move to the next step
        self.current_step += 1

        # Check if we've reached the end of the data
        if self.current_step >= self.max_step:
            done = True
            if self.stock_count > 0:  # Sell any remaining long stocks
                self.balance += self.stock_count * current_price
                profit = (current_price - self.buy_price) * self.stock_count
                reward += profit / self.buy_price  # Final normalized ROI
                self.stock_count = 0
            if self.short_stock_count > 0:  # Cover any remaining short stocks
                self.balance -= self.short_stock_count * current_price
                profit = (self.short_price - current_price) * self.short_stock_count
                reward += profit / self.short_price  # Final normalized short ROI
                self.short_stock_count = 0

        # Update state: use the window of data up to the current step
        if not done:
            window_data = self.data[self.current_step - self.window_size:self.current_step]
            self.state = window_data[-1]  # Latest data point in the window
        else:
            self.state = self.data[self.current_step - 1]

        # Record balance history
        self.balance_history.append(self.balance + (self.stock_count * current_price) - (self.short_stock_count * current_price))

        return self.state, reward, done, {"balance": self.balance_history}

    def render(self, mode='human'):
        # Optionally implement visualization
        pass
