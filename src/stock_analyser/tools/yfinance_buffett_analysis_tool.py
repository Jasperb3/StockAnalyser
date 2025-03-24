from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.warren_buffett_analysis import calculate_buffett_analysis_data
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceBuffettAnalysisToolInput(BaseModel):
    """Input schema for YFinanceBuffettAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol.")


# Define the tool class
class YFinanceBuffettAnalysisTool(BaseTool):
    name: str = "YFinance Buffett Analysis Tool"
    description: str = "Produces a Buffett analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceBuffettAnalysisToolInput

    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Produces a Buffet analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol, e.g. 'AAPL' for Apple Inc.
        :return: A dictionary containing the analysis results.
        """
        
        try:
            return calculate_buffett_analysis_data(ticker)
        except Exception as e:
            return f"Error: {e}"



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceBuffettAnalysisTool()
    buffet_analysis = tool_instance.run(ticker='MANU')
    print(buffet_analysis)