import yfinance as yf
import pandas as pd
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
os.makedirs(DATA_DIR, exist_ok=True)

# Define date range (past 1 year)
from datetime import datetime, timedelta
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

def fetch_gold_prices():
    """Fetch historical gold prices (GC=F)"""
    gold = yf.download("GC=F", start=start_date, end=end_date)

    # Reset index to make 'Date' a normal column
    gold.reset_index(inplace=True)

    # Ensure correct column names
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    gold.columns = expected_columns[:len(gold.columns)]  # Trim column names if needed

    # Save correctly formatted data
    file_path = os.path.join(DATA_DIR, "gold_prices.csv")
    gold.to_csv(file_path, index=False)
    print(f"Gold prices saved to {file_path}")

def fetch_dxy_index():
    """Fetch US Dollar Index (DXY)"""
    dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date)

    # Reset index to make 'Date' a normal column
    dxy.reset_index(inplace=True)

    # Ensure correct column names
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    dxy.columns = expected_columns[:len(dxy.columns)]  # Trim column names if needed

    # Save correctly formatted data
    file_path = os.path.join(DATA_DIR, "dxy_index.csv")
    dxy.to_csv(file_path, index=False)
    print(f"DXY Index saved to {file_path}")

if __name__ == "__main__":
    print(f"Fetching financial data from {start_date} to {end_date}...")
    fetch_gold_prices()
    fetch_dxy_index()
    print("Data fetching complete.")
