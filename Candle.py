from enum import Enum
import pandas as pd

class CandlePeriod(Enum):
    ONE_MINUTE = 60  # 60 seconds
    FIVE_MINUTES = 300  # 300 seconds
    ONE_HOUR = 3600  # 3600 seconds

class Candle:
    def __init__(self, price, volume, time):
        self.low = price
        self.high = price
        self.open = price
        self.close = price
        self.volume = volume
        self.open_time = time
        self.time_period = 60  # Time period in seconds

        # Set the close time in milliseconds
        self.close_time = time + 60

    def update_candle(self, new_price, volume, new_time):
        self.current_time = new_time  # Update the current time
        self.low = min(self.low, new_price)
        self.high = max(self.high, new_price)
        self.close = new_price
        added_volume = volume - self.volume
        self.volume += added_volume

    def to_dataframe(self):
        # Create a dictionary with candle data
        data = {
            'open_time': [self.open_time],
            'close_time': [self.close_time],
            'open': [self.open],
            'high': [self.high],
            'low': [self.low],
            'close': [self.close],
            'volume': [self.volume]
        }
        # Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame(data)
        return df