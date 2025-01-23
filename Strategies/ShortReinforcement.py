from Strategies.TradingStrategy import TradingStrategy
from Candle import Candle
import pandas as pd
import pickle
from Preprocessing.FeatureEngineering import calculate_macd, calculate_rsi, \
    calculate_cci, calculate_adx, calculate_moving_velocity_acceleration
from Preprocessing.CreateTensors import create_most_recent_window

from stable_baselines3 import PPO, A2C

def create_tensors(data, window_size):
    tensors = create_most_recent_window(data, window_size)
    return tensors


class ShortReinforcementStrategy(TradingStrategy):
    """
    A concrete trading strategy that uses a reinforcement learning model to decide trades in real-time.
    This strategy executes buy, sell, short, or hold actions based on the RL model's predictions.
    """

    def __init__(self, initial_balance=25000, verbose=True):
        super().__init__(initial_balance)
        self.candle_history = []
        self.current_candle = None
        self.model = PPO.load('ppo_trading_model_short')
        self.scaler = self.load_scaler()
        self.window_size = 15
        self.data = pd.DataFrame()
        self.verbose = verbose
        self.short_stock_count = 0  # Track number of stocks sold short
        self.short_price = 0  # Track the price at which short positions are opened
        self.trades = []

    def make_decision(self, obs, price, **kwargs):
        """
        Use the RL model to make a decision and execute trades.
        - obs: The current observation, which could be price data and technical indicators.
        - rl_model: The trained RL model that outputs actions.
        :param **kwargs:
        """
        decision = ''
        # Predict the action using the RL model (deterministic=True for real-time decision making)
        action, _ = self.model.predict(obs, deterministic=True)

        # Actions: 0 = Buy, 1 = Sell/Short, 2 = Hold
        if action == 0:  # Buy
            if self.short_stock_count > 0:  # Cover short if in a short position
                self.cover_short(price)
                self.trades.append(f"Cover short stock at {price}.")
                decision = 'Cover Short'
                print(f"Cover short at price: {price}")
            elif self.can_buy(price):  # Only buy if not holding any stock
                self.buy(price)
                self.trades.append(f"Buy at price: {price}.")
                decision = 'Buy'
                print(f"Buy at price: {price}")
            else:
                if self.verbose:
                    self.trades.append(f"Cannot buy at price: {price}.")
                    decision = 'Cannot Buy'
                    print(f"Cannot buy at price {price}, insufficient balance.")

        elif action == 1:  # Sell/Short
            if self.stock_count > 0:  # Sell long position
                self.sell(price)
                self.trades.append(f"Sell at price: {price}.")
                decision = 'Sell'
                print(f"Sell at price: {price}")
            elif self.short_stock_count == 0:  # Open short if no position is held
                if self.can_short(price):  # Ensure balance is enough to short
                    self.short(price)
                    self.trades.append(f"Short stock at price: {price}.")
                    decision = 'Short'
                    print(f"Short at price: {price}")
            else:
                if self.verbose:
                    self.trades.append(f"Cannot short at price: {price}.")
                    decision = 'Cannot Short'
                    print(f"Cannot short at price {price}, already in a short position.")

        elif action == 2:  # Hold
            if self.verbose:
                self.trades.append(f"Hold at price: {price}.")
                decision = 'Hold'
                print(f"Holding at price {price}")

        return decision
    def run(self, price, volume, time):
        """Run the strategy for each incoming price tick."""
        # Initialize the first candle if none exists
        if self.current_candle is None:
            self.start(price, volume, time)

        # Update the current candle with the latest price and volume
        self.current_candle.update_candle(price[-1], volume[-1], time[-1])
        ### Change this back to plus
        self.current_candle.close_time -= 86400

        # if self.verbose:
        #     print(f"Price: {self.current_candle.close} | Volume: {self.current_candle.volume}")
        #     if len(self.data) < self.window_size:
        #         print(f"Data Points collected: {len(self.data)} and data points needed {self.window_size}")

        # Check if it's time to close the current candle and start a new one

        print(time[-1] > self.current_candle.close_time)

        if time[-1] > self.current_candle.close_time:
            data = self.current_candle.to_dataframe()
            self.data = pd.concat([self.data, data], ignore_index=True)
            if len(self.data) > self.window_size:
                processed_data = self.process_data(self.data)
                scaled_data = self.scale_data(processed_data)
                tensors = create_tensors(scaled_data, self.window_size)
                # Make a decision before starting a new candle
                # if self.verbose:
                print(f"Making Decision")
                decision = self.make_decision(tensors, price[-1])

            # Archive the current candle and start a new one
            self.candle_history.append(self.current_candle)
            self.current_candle = Candle(price[-1], volume[-1], time[-1])
            self.current_candle.close_time += 86400

        if self.verbose:
            portfolio_value = self.calculate_portfolio_value(price)
            self.trades.append(f"Portfolio Value: {portfolio_value}")
            print(f"Portfolio Value: {portfolio_value}")

        return decision

    def start(self, price, volume, time):
        """Initialize the first candle and start tracking."""

        for i in range(15):
            c = Candle(price[-i], volume[-i], time[-i]).to_dataframe()
            self.data = pd.concat([self.data, c], ignore_index=True)
        self.current_candle = Candle(price[-1], volume[-1], time[-1])
        self.current_candle.update_candle(price[-1], volume[-1], time[-1])


    def process_data(self, data):
        data = calculate_macd(data)
        data = calculate_rsi(data)
        data = calculate_cci(data)
        data = calculate_adx(data)
        data = calculate_moving_velocity_acceleration(data)
        data = data.fillna(0)

        return data

    def scale_data(self, data):
        # Handle missing values and scale features
        data = self.scaler.transform(
            data[['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration']])
        data = pd.DataFrame(data, columns=['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration'])

        return data

    def load_scaler(self):
        """Load the scaler once when initializing the strategy."""
        try:
            with open('trading_scaler.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise ValueError("Scaler file 'trading_scaler.pkl' not found. Ensure the file is available.")

    def short(self, price):
        """Open a short position by selling stocks."""
        num_stocks = self.balance // price
        if num_stocks > 0:
            self.short_stock_count = num_stocks
            self.balance += self.short_stock_count * price
            self.short_price = price
            print(f"Opened short position with {num_stocks} stocks at price {price}")

    def cover_short(self, price):
        """Close the short position by buying back the stocks."""
        if self.short_stock_count > 0:
            self.balance -= self.short_stock_count * price
            profit = (self.short_price - price) * self.short_stock_count
            self.short_stock_count = 0
            print(f"Covered short position at price {price} for profit: {profit}")

    def can_short(self, price):
        """Check if there's enough balance to open a short position."""
        return self.balance >= price

    def calculate_portfolio_value(self, price):
        """Calculate portfolio value including long and short positions."""
        portfolio_value = self.balance + (self.stock_count * price) - (self.short_stock_count * price)
        return portfolio_value
