# trading_strategy.py

import abc

class TradingStrategy(abc.ABC):

    def __init__(self, initial_balance=25000):
        self.balance = initial_balance
        self.stock_count = 0

    def _get_buy_amount(self, price):
        return int(self.balance // price)  # Whole shares only

    def _get_sell_amount(self):
        return self.stock_count

    def buy(self, price):
        amount_to_buy = self._get_buy_amount(price)
        if amount_to_buy > 0:
            self.stock_count += amount_to_buy
            self.balance -= amount_to_buy * price

    def sell(self, price):
        amount_to_sell = self._get_sell_amount()
        if amount_to_sell > 0:
            self.balance += amount_to_sell * price
            self.stock_count = 0

    def can_buy(self, price):
        return self.balance >= price

    def can_sell(self):
        return self.stock_count > 0

    def calculate_portfolio_value(self, price):
        return self.balance + (self.stock_count * price)

    @abc.abstractmethod
    def make_decision(self, price, time):
        pass
