# app/services/model_training.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from app.data_loader.download_data import TICKERS

def create_target_labels(df, horizon=5):
    df = df.copy()
    df["target_price"] = df["Close"].shift(-horizon)
    return df.dropna()

def create_lstm_data(df, window=60):
    X, y = [], []
    df = df.select_dtypes(include=["number"])
    feature_cols = [col for col in df.columns if col not in ["target_price"]]

    for i in range(len(df) - window):
        X.append(df.iloc[i:i+window][feature_cols].values)
        y.append(df.iloc[i+window]["target_price"])

    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def train_and_save_model(ticker, processed_dir="data/processed", model_dir="models"):
    df = pd.read_csv(f"{processed_dir}/{ticker}_features.csv", index_col=0, parse_dates=True)
    df = create_target_labels(df)
    df.dropna(inplace=True)

    feature_cols = [col for col in df.columns if col not in ["target_price", "Ticker"]]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    df[feature_cols] = scaler_X.fit_transform(df[feature_cols])
    df["target_price"] = scaler_y.fit_transform(df["target_price"].values.reshape(-1, 1))

    X, y = create_lstm_data(df)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_model((X.shape[1], X.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, f"{ticker}_lstm.h5"))
    joblib.dump(feature_cols, os.path.join(model_dir, f"{ticker}_features.pkl"))
    joblib.dump(scaler_X, os.path.join(model_dir, f"{ticker}_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, f"{ticker}_scaler_y.pkl"))

    print(f"✅ Model saved for {ticker}")

if __name__ == "__main__":
    for ticker in TICKERS:
        train_and_save_model(ticker)