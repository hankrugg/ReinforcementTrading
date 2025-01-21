import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import pickle
import gym
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from schwab.auth import client_from_token_file
from stable_baselines3 import PPO, A2C
from ReinforcementLearning.ShortEnvironment import TradingEnv
from ReinforcementLearning.EarlyStopping import EarlyStoppingCallback
from Preprocessing.FeatureEngineering import calculate_macd, calculate_rsi,\
calculate_cci, calculate_adx, calculate_moving_velocity_acceleration
from sklearn.preprocessing import StandardScaler, OneHotEncoder



script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

load_dotenv()

# Email credentials from .env file
email_user = os.getenv('email_user')  # Your email address
email_password = os.getenv('email_password')  # Your email password

def send_email(subject, body):
    # Check if required fields are None
    if email_user is None or email_password is None:
        print("Error: email_user or email_password is not set.")
        return

    if subject is None or body is None:
        print("Error: Email subject or body is None.")
        return

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_user
    msg['Subject'] = subject

    # Attach the body as plain text
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Setup the server connection
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Log in to the server
        server.login(email_user, email_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(email_user, email_user, text)

        # Close the connection
        server.quit()

        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def fetch_daily_data(symbol, start_datetime, end_datetime, client):
    """
    Fetch minute-by-minute historical price data using the schwab-py API.
    """
    # Use the get_price_history_every_minute method
    resp = client.get_price_history_every_day(
        symbol=symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        need_extended_hours_data=True,
        need_previous_close=False
    )

    data = resp.json()
    candles = data.get('candles', [])
    if not candles:
        raise Exception(f"No price data available for {symbol} for the given period.")

    # Convert data to DataFrame
    df = pd.DataFrame(candles)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df


load_dotenv()
api_key = os.getenv('api_key')
app_secret = os.getenv('app_secret')
account_id = os.getenv('account_id')
token_path = "../Lamar/tokens/tokens.json"

client = client_from_token_file(token_path, api_key, app_secret)

testing_start = datetime.datetime.now()
validating_start = testing_start - datetime.timedelta(days=100)
training_start = validating_start - datetime.timedelta(days=3000)
print(f"Testing from {testing_start} to {datetime.datetime.now()}")
print(f"Validating from {validating_start} to {testing_start}")
print(f"Training from {training_start} to {validating_start}")

# Define backtest parameters
symbol = 'TSLA'  # Example stock symbol

# Fetch data for the symbol
training_data = fetch_daily_data(symbol, training_start, validating_start, client)
print(f"Training from {training_start} to {validating_start}")

end = datetime.datetime.now()
start = datetime.datetime.now() - datetime.timedelta(days=1)
fetch_daily_data("TSLA", start, end, client)
# Feature Engineering
training_data = calculate_macd(training_data)
training_data = calculate_rsi(training_data)
training_data = calculate_cci(training_data)
training_data = calculate_adx(training_data)
training_data = calculate_moving_velocity_acceleration(training_data)
training_data = training_data.fillna(0)

# Handle missing values and scale features
scaler = StandardScaler()
scaler.fit(training_data[['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration']])
# Save the scaler
with open('trading_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
training_scaled_features = scaler.transform(training_data[['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration']])
training_scaled_features = pd.DataFrame(training_scaled_features, columns = ['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration'])
from Preprocessing.CreateTensors import create_moving_windows

window_size = 15  # 30 minutes
stride = 1  # Move by 1 minute

# Apply the moving window function to your data
training_windows = create_moving_windows(training_scaled_features, window_size, stride)

# Convert numpy arrays to DataFrames
training_X_windows = np.array(training_windows)

# Print shapes to verify
print(f"Shape of X_windows: {training_X_windows.shape}")
training_prices = training_data['close']

training_data = []
for x in training_X_windows:
    training_flattened = x.flatten()
    training_data.append(training_flattened)


# Create the environment
training_env = TradingEnv(training_data, training_prices, window_size)


# Instantiate the model with the custom policy
training_model = PPO("MlpPolicy", training_env, verbose=0)


# Create an early stopping callback
early_stopping_callback = EarlyStoppingCallback(patience=100000)


# Train the model
training_model.learn(total_timesteps=10000, callback=early_stopping_callback)

# Save the model
training_model.save("ppo_trading_model_short")

# Fetch data for the symbol
validating_data = fetch_daily_data(symbol, validating_start, testing_start, client)
print(f"Validating from {validating_start} to {testing_start}")

# Feature Eng
validating_data = calculate_macd(validating_data)
validating_data = calculate_rsi(validating_data)
validating_data = calculate_cci(validating_data)
validating_data = calculate_adx(validating_data)
validating_data = calculate_moving_velocity_acceleration(validating_data)
validating_data = validating_data.fillna(0)
validating_scaled_features = scaler.transform(validating_data[['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration']])
validating_scaled_features = pd.DataFrame(validating_scaled_features, columns = ['close', 'high', 'low', 'volume', 'MACD', 'rsi', 'cci', 'adx', 'velocity', 'acceleration'])

# Apply the moving window function to your data
validating_windows = create_moving_windows(validating_scaled_features, window_size, stride)
# Convert numpy arrays to DataFrames
validating_X_windows = np.array(validating_windows)
validating_prices = validating_data['close']

validating_data = []
for x in validating_X_windows:
    validating_flattened = x.flatten()
    validating_data.append(validating_flattened)

# Load the previously saved model
validating_model = PPO.load("ppo_trading_model_short")

# Create the updated environment
validating_env = TradingEnv(validating_data, validating_prices, window_size)

# Create an early stopping callback
early_stopping_callback = EarlyStoppingCallback(patience=10000)

# Continue training the model
validating_model.set_env(validating_env)
validating_model.learn(total_timesteps=10000000, callback=early_stopping_callback, reset_num_timesteps=False)

# Optionally save the model again if needed
validating_model.save("ppo_trading_model_short")


send_email(subject="Model Validation Complete", body="The PPO trading model has finished validating and has been saved.")
print("Done Training")