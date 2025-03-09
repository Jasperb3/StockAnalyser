# historical_prices_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class HistoricalPricesInput(BaseModel):
    """
    Input schema for HistoricalPricesTool.
    Provide a stock ticker plus start and end dates in ISO format (YYYY-MM-DD).
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")
    start_date: str = Field(..., description="Start date in ISO format, e.g. '2024-01-01'.")
    end_date: str = Field(..., description="End date in ISO format, e.g. '2024-12-31'.")

class HistoricalPricesTool(BaseTool):
    name: str = "Historical Prices Tool"
    description: str = (
        "Retrieves historical price data (date, open, high, low, close, volume) "
        "for a specified ticker and date range."
    )
    args_schema: Type[BaseModel] = HistoricalPricesInput

    def _run(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Run method for the HistoricalPricesTool.
        Retrieves the historical price data from 'start_date' to 'end_date'.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_historical_price(ticker, start_date, end_date)
        return df.to_json(orient="records")