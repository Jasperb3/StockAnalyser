from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Type

# Define the input schema using Pydantic
class YFinanceSustainabilityToolInput(BaseModel):
    """Input schema for YFinanceSustainabilityTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceSustainabilityTool(BaseTool):
    name: str = "YFinance Sustainability Tool"
    description: str = "Fetches sustainability data for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceSustainabilityToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches sustainability data for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's sustainability data.
        """
        company_ticker = yf.Ticker(ticker)
        sustainability_data = company_ticker.sustainability

        if not sustainability_data.empty:
            return sustainability_data.to_markdown()
        else:
            return "No sustainability data found for this ticker."
        

if __name__ == '__main__':
    sustainability_tool = YFinanceSustainabilityTool()
    test_ticker = 'AAPL'
    result = sustainability_tool.run(test_ticker)
    print(result)