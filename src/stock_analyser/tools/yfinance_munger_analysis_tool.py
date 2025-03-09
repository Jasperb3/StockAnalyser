from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.charlie_munger_analysis import analyse_charlie_munger_valuation
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceMungerAnalysisToolInput(BaseModel):
    """Input schema for YFinanceMungerAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")


# Define the tool class
class YFinanceMungerAnalysisTool(BaseTool):
    name: str = "YFinance Munger Analysis Tool"
    description: str = "Produces a Charlie Munger-style analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceMungerAnalysisToolInput

    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Produces a Charlie Munger-style analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A dictionary containing the analysis results.
        """
        
        return analyse_charlie_munger_valuation(ticker)



if __name__ == "__main__":
    tool_instance = YFinanceMungerAnalysisTool()
    munger_analysis = tool_instance.run(ticker= 'NVDA')
    print(munger_analysis)