from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.stanley_druckenmiller_analysis import calculate_druckenmiller_data
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceDruckenmillerAnalysisToolInput(BaseModel):
    """Input schema for YFinanceDruckenmillerAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")


# Define the tool class
class YFinanceDruckenmillerAnalysisTool(BaseTool):
    name: str = "YFinance Druckenmiller Analysis Tool"
    description: str = "Produces a Druckenmiller analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceDruckenmillerAnalysisToolInput

    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Produces a Druckenmiller analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A dictionary containing the analysis results.
        """
        
        return calculate_druckenmiller_data(ticker)



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceDruckenmillerAnalysisTool()
    druckenmiller_analysis = tool_instance.run(ticker= 'NVDA')
    print(druckenmiller_analysis)