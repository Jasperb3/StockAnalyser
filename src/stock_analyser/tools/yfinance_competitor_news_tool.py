from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Type, List
from pprint import pprint

# Define the input schema using Pydantic
class YFinanceCompetitorNewsToolInput(BaseModel):
    """Input schema for YFinanceCompetitorNewsTool."""
    tickers: List[str] = Field(..., description="List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'] for Apple Inc., NVIDIA Corp., and Alphabet Inc.)")

# Define the tool class
class YFinanceCompetitorNewsTool(BaseTool):
    name: str = "YFinance Competitor News Tool"
    description: str = "Fetches and analyzes news for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceCompetitorNewsToolInput

    def _run(self, tickers: List[str]) -> str:
        """
        Fetches news and research for given tickers using yfinance.
        :param tickers: List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'] for Apple Inc., NVIDIA Corp., and Alphabet Inc.)
        :return: A string containing the stock's news.
        """
        competitor_news = {}
        for ticker in tickers:
            
            # get list of news
            news = yf.Search(ticker, news_count=10).news

            # get list of related research
            research = yf.Search(ticker, include_research=True).research

            competitor_news[ticker] = {
                "news": news,
                "research": research
            }
        
        return competitor_news
    

if __name__ == "__main__":
    tool_instance = YFinanceCompetitorNewsTool()
    competitor_news = tool_instance.run(tickers=['NVDA', 'AAPL', 'GOOG'])
    pprint(competitor_news)