from typing import Type

import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class YFinanceFinancialDataToolInput(BaseModel):
    """Input schema for YFinanceFinancialDataTool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")


class YFinanceFinancialDataTool(BaseTool):
    name: str = "YFinance Financial Data Tool"
    description: str = (
        "Fetches the latest financial data for a given stock ticker using yfinance."
    )
    args_schema: Type[BaseModel] = YFinanceFinancialDataToolInput

    def _run(self, ticker: str) -> str:
        stock = yf.Ticker(ticker)
        financial_data = stock.financials.iloc[:, [0]].to_dict()
        return financial_data
    

if __name__ == "__main__":
    tool = YFinanceFinancialDataTool()
    print(tool.run("AAPL"))
