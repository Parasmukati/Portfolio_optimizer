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
import mlflow
import mlflow.keras
from datetime import datetime
from pathlib import Path

from app.data_loader.download_data import TICKERS

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5001")  # or your MLflow server URI
mlflow.set_experiment("Stock_Prediction_LSTM")

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
    with mlflow.start_run(run_name=f"{ticker}_LSTM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "ticker": ticker,
            "horizon": 5,
            "window_size": 60,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "learning_rate": 0.001
        })
        
        # Load and prepare data
        df = pd.read_csv(f"{processed_dir}/{ticker}_features.csv", index_col=0, parse_dates=True)
        df = create_target_labels(df)
        df.dropna(inplace=True)

        feature_cols = [col for col in df.columns if col not in ["target_price", "Ticker"]]
        mlflow.log_param("feature_columns", feature_cols)
        
        # Scale data
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        df[feature_cols] = scaler_X.fit_transform(df[feature_cols])
        df["target_price"] = scaler_y.fit_transform(df["target_price"].values.reshape(-1, 1))

        X, y = create_lstm_data(df)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Log dataset information
        mlflow.log_metrics({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "total_samples": len(X)
        })

        # Build and train model
        model = build_lstm_model((X.shape[1], X.shape[2]))

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        # Log metrics
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metrics({
                "train_loss": history.history['loss'][epoch],
                "val_loss": history.history['val_loss'][epoch],
                "train_mae": history.history['mae'][epoch],
                "val_mae": history.history['val_mae'][epoch]
            }, step=epoch)

        # Log final metrics
        final_metrics = {
            "final_train_loss": history.history['loss'][-1],
            "final_val_loss": history.history['val_loss'][-1],
            "final_train_mae": history.history['mae'][-1],
            "final_val_mae": history.history['val_mae'][-1],
            "best_epoch": len(history.history['loss']) - early_stop.patience - 1
        }
        mlflow.log_metrics(final_metrics)

        # Save artifacts
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{ticker}_lstm.h5")
        model.save(model_path)
        
        # Log model with MLflow
        mlflow.keras.log_model(
            model,
            artifact_path=f"{ticker}_model",
            registered_model_name=f"LSTM_{ticker}"
        )

        # Save and log additional artifacts
        artifacts = {
            f"{ticker}_features.pkl": joblib.dump(feature_cols, os.path.join(model_dir, f"{ticker}_features.pkl")),
            f"{ticker}_scaler_X.pkl": joblib.dump(scaler_X, os.path.join(model_dir, f"{ticker}_scaler_X.pkl")),
            f"{ticker}_scaler_y.pkl": joblib.dump(scaler_y, os.path.join(model_dir, f"{ticker}_scaler_y.pkl"))
        }

        # Log all artifacts
        mlflow.log_artifacts(model_dir, artifact_path="model_artifacts")

        print(f"✅ Model saved for {ticker}")

if __name__ == "__main__":
    # Ensure MLflow server is running or configure remote tracking URI
    for ticker in TICKERS:
        try:
            train_and_save_model(ticker)
        except Exception as e:
            print(f"❌ Failed to train model for {ticker}: {str(e)}")
            mlflow.log_param(f"{ticker}_error", str(e))
            continue