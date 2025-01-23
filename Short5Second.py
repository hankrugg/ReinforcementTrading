import asyncio
import os
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

import dotenv
import pytz
import schwab as schwab
from schwab.auth import client_from_token_file
from dotenv import load_dotenv
import Strategies
import pandas as pd
import httpx
from Strategies.ShortReinforcement import ShortReinforcementStrategy
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

load_dotenv()

# Load environment variables
api_key = os.getenv("api_key")
app_secret = os.getenv("app_secret")
account_id = os.getenv("account_id")
token_path = "../Lamar/tokens/tokens.json"
email_user = os.getenv("email_user")  # Your email
email_password = os.getenv("email_password")  # Your email password
starting_balance = os.getenv("balance")

# Initialize the Schwab client
client = client_from_token_file(token_path, api_key, app_secret)

# Initialize your strategy
strategy_1 = ShortReinforcementStrategy(verbose=True)  # Example strategy

trades_1 = []
portfolio_value_1 = float(starting_balance)  # Initial portfolio value
previous_volume = 0
eastern = pytz.timezone('US/Eastern')


# Function to fetch historical data (latest minute)
def fetch_daily_data(symbol, start_datetime, end_datetime):
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

    # Check if the response is successful
    if resp.status_code != httpx.codes.OK:
        raise Exception(f"Failed to fetch data for {symbol}: {resp.status_code} - {resp.text}")

    data = resp.json()
    candles = data.get('candles', [])
    if not candles:
        raise Exception(f"No price data available for {symbol} for the given period.")

    # Convert data to DataFrame
    df = pd.DataFrame(candles)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

def process_latest_data(symbol):

    ## Change this to get 15 at once

    # Set start time to 5 minutes ago and end time as the current time
    start_time = datetime.now(eastern) - timedelta(days=30)
    end_time = datetime.now(eastern)

    # Fetch the most recent minute's price data
    data = fetch_daily_data(symbol, start_time, end_time)
    # Use the latest row from the fetched data
    latest_data = data.iloc[-15:]

    # Extract price and volume data
    bid_price = latest_data['close'].values
    volume = latest_data['volume'].values
    time = latest_data.index.to_pydatetime()
    time = [x.timestamp() for x in time]

    return bid_price, volume, time

# Email functions
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_user
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_user, email_password)
        server.sendmail(email_user, email_user, msg.as_string())
        server.quit()
        print(f"Email sent successfully: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Function to get the latest price and run the strategy
async def run_trading_strategy():
    symbol = 'TSLA'  # Replace with the desired stock symbol
    bid_price, volume, time = process_latest_data(symbol)
    # Run the strategy
    decision = strategy_1.run(bid_price, volume, time)
    send_email('Johnny Decision', f'{decision} {symbol} at {bid_price}: {time}')
    sys.exit(1)


# Main function
async def main():

    await asyncio.gather(
        run_trading_strategy(),
    )

# Run the event loop
asyncio.run(main())
