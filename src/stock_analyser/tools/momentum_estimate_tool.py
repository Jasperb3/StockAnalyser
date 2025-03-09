# momentum_estimate_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class MomentumEstimateInput(BaseModel):
    """
    Input schema for MomentumEstimateTool.
    Provide a single stock ticker symbol.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")

class MomentumEstimateTool(BaseTool):
    name: str = "Momentum Estimate Tool"
    description: str = "Retrieves the momentum estimate for a given stock ticker."
    args_schema: Type[BaseModel] = MomentumEstimateInput

    def _run(self, ticker: str) -> str:
        """
        Run method for the MomentumEstimateTool.
        Retrieves momentum estimate data for the ticker from nasdaq-data.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_data(ticker, 3)  # 3 => Momentum Estimate
        return df.to_json(orient="records")