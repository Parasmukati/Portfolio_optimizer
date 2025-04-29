# app/main.py

from fastapi import FastAPI
from app.api import predict, backtest

app = FastAPI(
    title="AI Portfolio Optimizer",
    description="Predict stock price movements and suggest portfolio reallocations.",
    version="1.0.0"
)

# Register API routes
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtesting"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Portfolio Optimizer API!"}