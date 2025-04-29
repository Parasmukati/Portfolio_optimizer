# app/data_loader/download_data.py

import yfinance as yf
import os
from datetime import datetime

# List of tickers to download
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS",
    "ICICIBANK.NS", "SBIN.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "ITC.NS", "TATAMOTORS.NS"
]

START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')  # Automatically today's date
# END_DATE="2021-12-31"

def download_and_save_data(tickers=TICKERS, start=START_DATE, end=END_DATE, save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        df["Ticker"] = ticker
        df.to_csv(f"{save_dir}/{ticker}.csv")
    
    print("\n✅ Data download complete.")

if __name__ == "__main__":
    download_and_save_data()
