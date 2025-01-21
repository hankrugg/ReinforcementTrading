import numpy as np


def create_moving_windows(data, window_size, stride=1):
    """
    Create moving windows from the past data, ensuring only past data (e.g., the previous 15 minutes) is used.

    Parameters:
    - data: DataFrame containing the data
    - window_size: Size of each window (e.g., 15 for the past 15 minutes)
    - stride: Stride length for moving the window

    Returns:
    - windows_array: A numpy array containing the flattened windows of past data
    """
    flattened_windows = []

    # Start from window_size to ensure we always have the past window_size data points
    for start in range(0, len(data) - window_size + 1):
        # Get the past 'window_size' rows (i.e., from start - window_size to start)
        window = data.iloc[start: start + window_size]

        # Flatten the window DataFrame to a 1D array
        flattened = window.values.flatten()
        flattened_windows.append(flattened)

    # Convert to numpy arrays
    windows_array = np.array(flattened_windows)

    return windows_array



def create_most_recent_window(data, window_size, stride=1):
    """
    Create the most recent moving window from the data.

    Parameters:
    - data: DataFrame containing the data
    - window_size: Size of the window
    - stride: (Optional) Stride length for consistency, though it's not needed here.

    Returns:
    - window_array: A numpy array containing the most recent window, flattened if necessary.
    """
    # Ensure the window size is not larger than the data
    if len(data) < window_size:
        raise ValueError(f"Data length ({len(data)}) is smaller than window size ({window_size}).")

    # Select the most recent window_size rows
    most_recent_window = data.iloc[-window_size:]

    # Convert the window DataFrame to a NumPy array
    window_array = most_recent_window.values

    return np.array(window_array).flatten()


def create_labels(data, window_size, stride=1):
    """
    Create labels for each moving window.

    Parameters:
    - data: DataFrame containing the data
    - window_size: Size of each window
    - stride: Stride length for moving the window

    Returns:
    - labels_array: A numpy array containing the labels
    """
    labels = []

    for start in range(0, len(data) - window_size + 1, stride):
        end = start + window_size
        window = data.iloc[start:end]

        # Use the label of the last time step in the window
        labels.append(window['decision'].iloc[-1])

    # Convert to numpy arrays
    labels_array = np.array(labels)

    return labels_array