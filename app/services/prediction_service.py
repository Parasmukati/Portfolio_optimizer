# app/services/prediction_service.py (corrected version)

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

def load_artifacts(ticker, model_dir="models"):
    model = load_model(f"{model_dir}/{ticker}_lstm.h5", compile=False)
    scaler_X = joblib.load(f"{model_dir}/{ticker}_scaler_X.pkl")
    scaler_y = joblib.load(f"{model_dir}/{ticker}_scaler_y.pkl")
    feature_cols = joblib.load(f"{model_dir}/{ticker}_features.pkl")
    return model, scaler_X, scaler_y, feature_cols

def create_lstm_input(df, feature_cols, window=60):
    df = df[feature_cols].reset_index(drop=True)
    if len(df) < window:
        raise ValueError("Not enough data for LSTM input window.")
    input_slice = df.iloc[-window:]  # 🔥 Always pick LAST 60 rows!
    return np.array([input_slice.values])

def predict_future_prices(customer_holdings: dict,offset: int = 0):
    predictions = {}
    for ticker in customer_holdings:
        model, scaler_X, scaler_y, feature_cols= load_artifacts(ticker)
        df = pd.read_csv(f"data/processed/{ticker}_features.csv", index_col=0, parse_dates=True)

        # 🔥 Print to check date range
        # print(f"\nTicker: {ticker}")
        # print("Last 5 dates in dataset:", df.index[-5:].tolist())
        if offset > 0:
            df = df.iloc[:-offset]
        
        window_size=60

        if len(df) < window_size:
            continue

        close_price = df["Close"].iloc[-1] 
        df[feature_cols] = scaler_X.transform(df[feature_cols])
        # print(close_price)
        X = create_lstm_input(df, feature_cols)  # No offset needed now
        predicted_price_scaled = model.predict(X, verbose=0)
        predicted_price = scaler_y.inverse_transform(predicted_price_scaled)[0][0]

        predictions[ticker] = {
            "current_price": float(close_price),
            "predicted_price": float(predicted_price),
            "potential_change": float((predicted_price - close_price) / close_price)
        }
    return predictions

# customer_holdings={
#     "TCS.NS": 100,
# }


# predict_future_prices(customer_holdings)