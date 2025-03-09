# stocks_financial_data_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class StocksFinancialDataInput(BaseModel):
    """
    Input schema for StocksFinancialDataTool.
    frequency=1 for annual, frequency=2 for semi-annual.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")
    frequency: int = Field(..., description="1 for annual, 2 for semi-annual financial data.")

class StocksFinancialDataTool(BaseTool):
    name: str = "Stocks Financial Data Tool"
    description: str = (
        "Fetches annual or semi-annual financial data (Income Statement, Balance Sheet, "
        "Cash Flow, Financial Ratios) for a specific ticker."
    )
    args_schema: Type[BaseModel] = StocksFinancialDataInput

    def _run(self, ticker: str, frequency: int) -> str:
        """
        Run method for the StocksFinancialDataTool.
        Retrieves financial data for the given ticker and frequency.
        Returns JSON-formatted results (one row with multiple columns).
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_financals(ticker, frequency)
        return df.to_json(orient="records")