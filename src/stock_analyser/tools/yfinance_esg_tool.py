from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Type

# Define the input schema using Pydantic
class YFinanceESGToolInput(BaseModel):
    """Input schema for YFinanceESGTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceESGTool(BaseTool):
    name: str = "YFinance ESG Tool"
    description: str = "Fetches Environmental, Social, and Governance (ESG) data for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceESGToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches ESG data for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's ESG data.
        """
        company_ticker = yf.Ticker(ticker)
        esg_data = company_ticker.sustainability

        if not esg_data.empty:
            return esg_data.to_markdown()
        else:
            return "No ESG data found for this ticker."
        

if __name__ == '__main__':
    esg_tool = YFinanceESGTool()
    test_ticker = 'AAPL'
    result = esg_tool.run(test_ticker)
    print(result)