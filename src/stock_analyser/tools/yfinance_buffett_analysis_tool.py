from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.warren_buffett_analysis import calculate_buffett_analysis_data
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceBuffettAnalysisToolInput(BaseModel):
    """Input schema for YFinanceBuffettAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")
    growth_rate: float = Field(default=0.05, description="Growth rate for the company. The default is 0.05.")
    discount_rate: float = Field(default=0.09, description="Discount rate for the company. The default is 0.09.")
    terminal_multiple: float = Field(default=12, description="Terminal multiple for the company. The default is 12.")
    projection_years: int = Field(default=10, description="Number of years to project for the company. The default is 10.")


# Define the tool class
class YFinanceBuffettAnalysisTool(BaseTool):
    name: str = "YFinance Buffett Analysis Tool"
    description: str = "Produces a Buffett analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceBuffettAnalysisToolInput

    
    def _run(self, ticker: str, growth_rate: float = 0.05, discount_rate: float = 0.09, terminal_multiple: float = 12, projection_years: int = 10) -> Dict[str, Any]:
        """
        Produces a Buffet analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :param growth_rate: Growth rate for the company. Default is 0.05.
        :param discount_rate: Discount rate for the company. Default is 0.09.
        :param terminal_multiple: Terminal multiple for the company. Default is 12.
        :param projection_years: Number of years to project for the company. Default is 10.
        :return: A dictionary containing the analysis results.
        """
        
        return calculate_buffett_analysis_data(ticker, growth_rate, discount_rate, terminal_multiple, projection_years)



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceBuffettAnalysisTool()
    buffet_analysis = tool_instance.run(ticker='FGBI')
    print(buffet_analysis)