# app/data_loader/feature_engineering.py

import pandas as pd
import numpy as np
import os

from download_data import TICKERS

def compute_features(df):
    df = df.copy()
    
    # Log Returns
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # Moving Averages
    df["MA13"] = df["Close"].rolling(window=13).mean()
    df["MA55"] = df["Close"].rolling(window=55).mean()
    df["MA233"] = df["Close"].rolling(window=233).mean()

    # Volatility
    df["Volatility"] = df["Log_Returns"].rolling(window=10).std()

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    return df.dropna()

def process_and_save_all(tickers=TICKERS, raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)

    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = pd.read_csv(f"{raw_dir}/{ticker}.csv", index_col=0, parse_dates=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # safety
        df_feat = compute_features(df)
        df_feat.to_csv(f"{processed_dir}/{ticker}_features.csv")

    print("\n✅ Feature engineering complete.")

if __name__ == "__main__":
    process_and_save_all()
