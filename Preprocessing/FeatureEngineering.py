"""
Created on
@author: <NAME>
  
"""

import numpy as np


def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """
    Calculates the MACD indicator for the given data, avoiding data leakage.

    :param data: DataFrame with a 'close' price column.
    :param short_period: Short EMA period, default is 12.
    :param long_period: Long EMA period, default is 26.
    :param signal_period: Signal line EMA period, default is 9.
    :return: DataFrame with 'MACD' and 'signal' columns.
    """
    # Ensure that the index is sorted
    data = data.sort_index()

    # Calculate EMA values
    data['EMA_short'] = data['close'].ewm(span=short_period, adjust=False).mean()
    data['EMA_long'] = data['close'].ewm(span=long_period, adjust=False).mean()

    # Calculate MACD and signal line
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

    return data


def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame, avoiding data leakage.
    This implementation uses Wilder's smoothing to avoid data leakage.
    """
    data = data.sort_index()

    # Calculate price changes
    data['price_change'] = data['close'].diff()

    # Calculate gains and losses
    data['gain'] = data['price_change'].clip(lower=0)
    data['loss'] = -data['price_change'].clip(upper=0)

    # Use Wilder's smoothing for gains and losses
    data['avg_gain'] = data['gain'].ewm(alpha=1/period, adjust=False).mean()
    data['avg_loss'] = data['loss'].ewm(alpha=1/period, adjust=False).mean()

    # Calculate RS and RSI
    data['rs'] = data['avg_gain'] / data['avg_loss']
    data['rsi'] = 100 - (100 / (1 + data['rs']))

    return data



def calculate_cci(data, period=14):
    """
    Calculate the Commodity Channel Index (CCI) for a given DataFrame, avoiding data leakage.

    :param data: DataFrame with 'high', 'low', and 'close' columns.
    :param period: The period over which to calculate CCI, default is 14.
    :return: DataFrame with an additional 'CCI' column.
    """
    data = data.sort_index()

    # Calculate Typical Price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3

    # Calculate rolling mean of Typical Price
    data['sma'] = data['typical_price'].rolling(window=period, min_periods=1).mean()

    # Calculate Mean Absolute Deviation (MAD)
    data['mad'] = data['typical_price'].rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=False)

    # Calculate CCI
    data['cci'] = (data['typical_price'] - data['sma']) / (0.015 * data['mad'])

    # Drop intermediate columns
    data.drop(columns=['typical_price', 'sma', 'mad'], inplace=True)

    return data


def calculate_adx(data, period=14):
    """
    Calculate the Average Directional Index (ADX) for a given DataFrame, avoiding data leakage.
    This implementation uses Wilder's smoothing to avoid data leakage.
    """
    data = data.sort_index()

    # Calculate True Range (TR)
    data['prev_close'] = data['close'].shift(1)
    data['tr'] = data[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )

    # Calculate Directional Movement (DM+ and DM-)
    data['dm_plus'] = np.where((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
                               np.maximum(data['high'] - data['high'].shift(1), 0), 0)
    data['dm_minus'] = np.where((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
                                np.maximum(data['low'].shift(1) - data['low'], 0), 0)

    # Use Wilder's smoothing for TR, DM+, and DM-
    data['smoothed_tr'] = data['tr'].ewm(alpha=1/period, adjust=False).mean()
    data['smoothed_dm_plus'] = data['dm_plus'].ewm(alpha=1/period, adjust=False).mean()
    data['smoothed_dm_minus'] = data['dm_minus'].ewm(alpha=1/period, adjust=False).mean()

    # Calculate +DI and -DI
    data['plus_di'] = (data['smoothed_dm_plus'] / data['smoothed_tr']) * 100
    data['minus_di'] = (data['smoothed_dm_minus'] / data['smoothed_tr']) * 100

    # Calculate DX and ADX
    data['dx'] = (abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])) * 100
    data['adx'] = data['dx'].ewm(alpha=1/period, adjust=False).mean()

    # Drop intermediate columns
    data.drop(columns=['prev_close', 'tr', 'dm_plus', 'dm_minus', 'smoothed_tr', 'smoothed_dm_plus',
                       'smoothed_dm_minus', 'dx'], inplace=True)

    return data


def calculate_moving_velocity_acceleration(data, period=20):
    """
    Calculate the 20-day moving velocity and acceleration, ensuring no data leakage.

    :param data: DataFrame with a 'close' price column.
    :param period: The period over which to calculate the velocity and acceleration, default is 20.
    :return: DataFrame with 'velocity' and 'acceleration' columns.
    """
    data = data.sort_index()

    # Calculate the velocity as the price difference over the period, using past data
    data['velocity'] = (data['close'] - data['close'].shift(period)) / period

    # Calculate the acceleration as the change in velocity over the period, using past data
    data['acceleration'] = (data['velocity'] - data['velocity'].shift(period)) / period

    return data

