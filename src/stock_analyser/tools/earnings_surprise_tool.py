# earnings_surprise_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class EarningsSurpriseInput(BaseModel):
    """
    Input schema for EarningsSurpriseTool.
    Provide a single stock ticker symbol.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")

class EarningsSurpriseTool(BaseTool):
    name: str = "Earnings Surprise Tool"
    description: str = "Retrieves the earnings surprise data for a given stock ticker."
    args_schema: Type[BaseModel] = EarningsSurpriseInput

    def _run(self, ticker: str) -> str:
        """
        Run method for the EarningsSurpriseTool.
        Retrieves earnings surprise data for the ticker from nasdaq-data.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_data(ticker, 5)  # 5 => Earnings Surprise
        return df.to_json(orient="records")