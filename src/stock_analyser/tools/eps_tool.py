# eps_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class EPSToolInput(BaseModel):
    """
    Input schema for EPSTool.
    Provide a single stock ticker symbol.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")

class EPSTool(BaseTool):
    name: str = "EPS Tool"
    description: str = "Retrieves the EPS (Earnings Per Share) data for a given stock ticker."
    args_schema: Type[BaseModel] = EPSToolInput

    def _run(self, ticker: str) -> str:
        """
        Run method for the EPSTool.
        Retrieves EPS data for the ticker from nasdaq-data.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_data(ticker, 6)  # 6 => EPS
        return df.to_json(orient="records")

if __name__ == '__main__':
    # Create an instance of the EPSTool
    eps_tool = EPSTool()
    
    # Test with a sample ticker
    test_ticker = 'GOOG'
    result = eps_tool.run(test_ticker)
    
    # Print the results
    print(f"EPS data for {test_ticker}:")
    print(result)
