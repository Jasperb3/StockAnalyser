# analyst_target_price_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class AnalystTargetPriceInput(BaseModel):
    """
    Input schema for AnalystTargetPriceTool.
    Provide a single stock ticker symbol.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")

class AnalystTargetPriceTool(BaseTool):
    name: str = "Analyst Target Price Tool"
    description: str = (
        "Retrieves analyst target price and ratings for a given stock ticker "
        "from the nasdaq-data SDK."
    )
    args_schema: Type[BaseModel] = AnalystTargetPriceInput

    def _run(self, ticker: str) -> str:
        """
        Run method for the AnalystTargetPriceTool.
        Retrieves analyst data (target price, buy/hold/sell counts) for the ticker.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_data(ticker, 1)  # 1 => Analyst Target Price & Ratings
        return df.to_json(orient="records")