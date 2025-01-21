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

    # Log the extracted and calculated data
    # print(f"Bid Price: {bid_price}")
    # print(f"Current Volume: {volume}")
    # print(f'Current time:{time}')

    # print(bid_price, volume, time)

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


def send_trade_summary_email(trades_1, portfolio_value_1):
    subject = "Daily Trade Summary"
    body = f"Daily Reinforcement Strategy - Final Portfolio Value: ${portfolio_value_1:.2f}\n\nTrades:\n"
    for trade in trades_1:
        body += trade

    send_email(subject, body)


# Time calculations
def calculate_time_until_next_market_open():
    now = datetime.now(eastern)
    today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= today_open:
        tomorrow = now + timedelta(days=1)
        next_open = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
    else:
        next_open = today_open
    time_until_open = next_open - now
    return time_until_open

def send_intro_email():
    time_until_open = calculate_time_until_next_market_open()
    subject = "Daily Program Starting"
    body = f"The trading program has started successfully.\nTime until next market open: {time_until_open}"
    send_email(subject, body)


def send_ending_email(portfolio_value_1):
    time_until_open = calculate_time_until_next_market_open()
    subject = "Program Ending"
    body = f"Daily ShortReinforcementStrategy - Final Portfolio Value: ${portfolio_value_1:.2f}\nTime until next market open: {time_until_open}"
    send_email(subject, body)

async def hourly_email_update():
    while True:
        subject = "Daily Portfolio Update"
        body = f"ShortReinforcementStrategy Daily - Current Portfolio Value: ${portfolio_value_1[-1]:.2f}"
        send_email(subject, body)
        await asyncio.sleep(3600)  # Send email every hour


# Market close email
async def schedule_email_at_market_close():
    market_close_time = datetime.now(eastern).replace(hour=16, minute=0, second=0, microsecond=0)
    while True:
        now = datetime.now(eastern)
        if now >= market_close_time:
            print("Daily ShortReinforcementStrategy Market closed. Sending email summary...")
            send_trade_summary_email(strategy_1.trades, portfolio_value_1[-1])
            send_ending_email(portfolio_value_1[-1])
            dotenv.set_key('.env', 'balance', str(portfolio_value_1[-1]))
            sys.exit()
        await asyncio.sleep(60)


# Function to get the latest price and run the strategy
async def run_trading_strategy():
    symbol = 'TSLA'  # Replace with the desired stock symbol
    while True:
        # try:
            bid_price, volume, time = process_latest_data(symbol)
            if bid_price is not None:
                # Run the strategy
                strategy_1.run(bid_price, volume, time)
                global portfolio_value_1
                portfolio_value_1 = strategy_1.calculate_portfolio_value(bid_price)

            # Wait for the next 5 seconds (you can adjust this to match the timing of minute data updates)
            await asyncio.sleep(60)

        # except Exception as e:
        #     print(f"An error occurred during strategy execution: {e}")
        #     await asyncio.sleep(60)


# Main function
async def main():
    print("Program started.")
    send_intro_email()

    await asyncio.gather(
        run_trading_strategy(),
        schedule_email_at_market_close(),
        hourly_email_update()
    )


# Run the event loop
asyncio.run(main())
