from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.ben_graham_analysis import calculate_graham_analysis_data
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceGrahamAnalysisToolInput(BaseModel):
    """Input schema for YFinanceGrahamAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")


# Define the tool class
class YFinanceGrahamAnalysisTool(BaseTool):
    name: str = "YFinance Graham Analysis Tool"
    description: str = "Produces a Graham analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceGrahamAnalysisToolInput

    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Produces a Graham analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A dictionary containing the analysis results.
        """
        
        return calculate_graham_analysis_data(ticker)



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceGrahamAnalysisTool()
    graham_analysis = tool_instance.run(ticker= 'NVDA')
    print(graham_analysis)