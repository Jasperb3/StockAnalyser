from typing import Type

import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class YFinanceCashFlowToolInput(BaseModel):
    """Input schema for YFinanceCashFlowTool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")


class YFinanceCashFlowTool(BaseTool):
    name: str = "YFinance Cash Flow Tool"
    description: str = (
        "Fetches the latest cash flow data for a given stock ticker using yfinance."
    )
    args_schema: Type[BaseModel] = YFinanceCashFlowToolInput

    def _run(self, ticker: str) -> str:
        stock = yf.Ticker(ticker)
        cash_flow_data = stock.cashflow.iloc[:, [0]].to_dict()
        return cash_flow_data
    

if __name__ == "__main__":
    tool = YFinanceCashFlowTool()
    print(tool.run("AAPL"))
