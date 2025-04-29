from fastapi import APIRouter
from app.services.backtest_service import simulate_backtest

router = APIRouter()

@router.post("/simulate")
def backtest_simulation(
    customer_holdings: dict,
    starting_cash: float = 0,
    start_offset: int = 90,
    frequency: int = 5
):
    result = simulate_backtest(customer_holdings, starting_cash, start_offset, frequency)
    return result
