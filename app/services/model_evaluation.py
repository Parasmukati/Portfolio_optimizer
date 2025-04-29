import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import mlflow
from datetime import datetime

from app.data_loader.download_data import TICKERS

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Stock_Prediction_LSTM_Eval")

def evaluate_model(ticker, processed_dir="data/processed", model_dir="models"):
    # Load artifacts
    model_path = os.path.join(model_dir, f"{ticker}_lstm.h5")
    model = load_model(model_path)

    scaler_X = joblib.load(os.path.join(model_dir, f"{ticker}_scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, f"{ticker}_scaler_y.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, f"{ticker}_features.pkl"))

    # Load and prepare data
    df = pd.read_csv(os.path.join(processed_dir, f"{ticker}_features.csv"), index_col=0, parse_dates=True)
    df = df.dropna().copy()
    df["target_price"] = df["Close"].shift(-5)
    df.dropna(inplace=True)

    df[feature_cols] = scaler_X.transform(df[feature_cols])
    df["target_price"] = scaler_y.transform(df["target_price"].values.reshape(-1, 1))

    window = 60
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df.iloc[i:i + window][feature_cols].values)
        y.append(df.iloc[i + window]["target_price"])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    # Predict
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()

    rmse = mean_squared_error(y_test_inv, y_pred_inv, squared=False)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{ticker}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_param("ticker", ticker)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label="Actual")
        plt.plot(y_pred_inv, label="Predicted")
        plt.title(f"{ticker} - LSTM Predictions vs Actuals")
        plt.legend()
        plot_path = os.path.join(model_dir, f"{ticker}_prediction_plot.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        print(f"✅ Evaluation done for {ticker} | RMSE: {rmse:.2f}, MAE: {mae:.2f}")

if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            evaluate_model(ticker)
        except Exception as e:
            print(f"❌ Evaluation failed for {ticker}: {str(e)}")
