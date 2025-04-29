# app/services/backtest_service.py

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for server environments
import matplotlib.pyplot as plt
from app.services.prediction_service import predict_future_prices
from app.services.reallocation_service import reallocate_portfolio

def plot_performance(history, bh_history):
    days = [entry['day_offset'] for entry in history]
    total_values = [entry['total'] for entry in history]
    bh_total_values = [entry['total'] for entry in bh_history]

    plt.figure(figsize=(12, 6))
    plt.plot(days, total_values, marker='o', label='AI Portfolio Strategy', linewidth=2)
    plt.plot(days, bh_total_values, marker='x', linestyle='--', label='Buy and Hold Baseline', linewidth=2)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Days Ago')
    plt.ylabel('Total Portfolio Value (INR)')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.close()

def simulate_backtest(customer_holdings: dict, starting_cash: float = 0, start_offset: int = 90, frequency: int = 5):
    portfolio_history = []
    bh_history = []
    cash = starting_cash
    holdings = customer_holdings["customer_holdings"].copy()

    print(holdings)

    # Store initial buy-and-hold quantities
    buy_and_hold_qty = holdings.copy()

    for offset in range(start_offset, -1, -frequency):
        print(f"\n=== Simulating for Offset Day: {offset} ===")

        # Predict and Reallocate for AI strategy
        predictions = predict_future_prices(holdings, offset=offset)
        reallocation_result = reallocate_portfolio(holdings, predictions, leftover_fund=cash)

        holdings = {ticker: int(info["quantity"]) for ticker, info in reallocation_result["portfolio"].items()}
        cash = reallocation_result["leftover_fund"]

        total_value = sum(info["value"] for info in reallocation_result["portfolio"].values())

        portfolio_snapshot = {
            "day_offset": offset,
            "total_value": round(total_value, 2),
            "cash": round(cash, 2),
            "total": round(total_value + cash, 2),
            "holdings": reallocation_result["portfolio"]
        }
        portfolio_history.append(portfolio_snapshot)

        # Buy-and-hold baseline calculation
        bh_total = 0
        for ticker, qty in buy_and_hold_qty.items():
            df = pd.read_csv(f"data/processed/{ticker}_features.csv", index_col=0, parse_dates=True)
            close_price = df["Close"].iloc[-1 - offset]
            bh_total += close_price * qty

        bh_snapshot = {
            "day_offset": offset,
            "total": round(bh_total, 2)
        }
        bh_history.append(bh_snapshot)

    # Reverse to chronological order
    portfolio_history = sorted(portfolio_history, key=lambda x: x['day_offset'])
    bh_history = sorted(bh_history, key=lambda x: x['day_offset'])

    # Plot performance comparison
    plot_performance(portfolio_history, bh_history)

    # Final metrics
    final_ai = portfolio_history[-1]["total"]
    final_bh = bh_history[-1]["total"]

    return {
        "starting_cash": starting_cash,
        "starting_holdings": customer_holdings,
        "final_portfolio_value": final_ai,
        "final_buy_and_hold_value": final_bh,
        "return_ai_percentage": round(100 * (final_ai - starting_cash) / starting_cash, 2) if starting_cash else None,
        "return_bh_percentage": round(100 * (final_bh - starting_cash) / starting_cash, 2) if starting_cash else None,
        "history": portfolio_history,
        "buy_and_hold_history": bh_history
    }