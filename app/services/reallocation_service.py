# app/services/reallocation_service.py

def reallocate_portfolio(customer_holdings: dict, predictions: dict, leftover_fund: float = 0):
    fund_info = {}
    total_value = 0

    # Step 1: Build the fund info dictionary
    for ticker, qty in customer_holdings.items():
        close_price = predictions[ticker]["current_price"]
        predicted_price = predictions[ticker]["predicted_price"]
        potential_change = predictions[ticker]["potential_change"]

        holding_value = close_price * qty

        fund_info[ticker] = {
            "quantity": int(qty),
            "price": float(close_price),
            "value": float(holding_value),
            "predicted_price": float(predicted_price),
            "potential_change": float(potential_change),
            "withdraw": float(0)
        }
        total_value += holding_value

    # Step 2: Sort stocks by lowest to highest potential change
    sorted_stocks = sorted(fund_info.items(), key=lambda x: x[1]['potential_change'])

    num_to_withdraw = max(1, len(sorted_stocks) // 3)
    bottom_stocks = [t for t, _ in sorted_stocks[:num_to_withdraw]]
    top_stocks = [t for t, _ in sorted_stocks[num_to_withdraw:]]

    total_withdrawn = leftover_fund

    # Step 3: Withdraw from bottom 1/3rd stocks
    if bottom_stocks:
        max_withdraw_per_stock = 0.2 * total_value / len(bottom_stocks)

        for ticker in bottom_stocks:
            stock = fund_info[ticker]
            if stock["value"] == 0:
                continue

            forty_percent = 0.4 * stock["value"]
            to_withdraw = min(forty_percent, max_withdraw_per_stock)

            withdraw_qty = int(to_withdraw // stock["price"])
            actual_withdraw = withdraw_qty * stock["price"]

            stock["withdraw"] = actual_withdraw
            stock["value"] -= actual_withdraw
            stock["quantity"] = int(stock["value"] // stock["price"])
            stock["value"] = stock["quantity"] * stock["price"]

            total_withdrawn += actual_withdraw

    # Step 4: Distribute withdrawn cash across top 2/3rd stocks
    if top_stocks and total_withdrawn > 0:
        per_stock_alloc = total_withdrawn / len(top_stocks)

        for ticker in top_stocks:
            stock = fund_info[ticker]
            stock["value"] += per_stock_alloc
            stock["quantity"] = int(stock["value"] // stock["price"])
            stock["value"] = stock["quantity"] * stock["price"]

    updated_leftover_fund = total_value - sum(stock["value"] for stock in fund_info.values()) + leftover_fund

    return {
        "portfolio": fund_info,
        "leftover_fund": float(round(updated_leftover_fund, 2))
    }
