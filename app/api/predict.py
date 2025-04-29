# app/api/predict.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.services.prediction_service import predict_future_prices
from app.services.reallocation_service import reallocate_portfolio

router = APIRouter()

class PortfolioInput(BaseModel):
    customer_holdings: Dict[str, int]
    leftover_fund: float = 0

@router.post("/reallocate")
def reallocate_portfolio_api(input_data: PortfolioInput):
    """
    Input: 
    - customer_holdings: {"TCS.NS": 100, "RELIANCE.NS": 50, ...}
    - leftover_fund: Optional cash available for reallocation
    
    Output:
    - portfolio: Reallocated portfolio dict
    - leftover_fund: Updated remaining cash after reallocation
    """
    predictions = predict_future_prices(input_data.customer_holdings)
    allocation_result = reallocate_portfolio(input_data.customer_holdings, predictions, input_data.leftover_fund)
    return allocation_result
