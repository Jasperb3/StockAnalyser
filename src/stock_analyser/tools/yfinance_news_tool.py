from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Type

# Define the input schema using Pydantic
class YFinanceNewsToolInput(BaseModel):
    """Input schema for YFinanceNewsTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceNewsTool(BaseTool):
    name: str = "YFinance News Tool"
    description: str = "Fetches and analyzes news for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceNewsToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches news and research for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's news.
        """
        # get list of news
        news = yf.Search(ticker, news_count=12).news

        # get list of related research
        research = yf.Search(ticker, include_research=True).research
        
        return f"News: {news}\nResearch: {research}"