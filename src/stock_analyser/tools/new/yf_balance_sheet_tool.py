from typing import Type

import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class YFinanceBalanceSheetToolInput(BaseModel):
    """Input schema for YFinanceBalanceSheetTool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")


class YFinanceBalanceSheetTool(BaseTool):
    name: str = "YFinance Balance Sheet Tool"
    description: str = (
        "Fetches the latest balance sheet for a given stock ticker using yfinance."
    )
    args_schema: Type[BaseModel] = YFinanceBalanceSheetToolInput

    def _run(self, ticker: str) -> str:
        stock = yf.Ticker(ticker)
        balance_sheet_data = stock.balance_sheet.iloc[:, [0]].to_dict()
        return balance_sheet_data
    

if __name__ == "__main__":
    tool = YFinanceBalanceSheetTool()
    print(tool.run("AAPL"))
