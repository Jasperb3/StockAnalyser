from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.cathie_wood_analysis import calculate_cathie_wood_analysis_data
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceCathieWoodAnalysisToolInput(BaseModel):
    """Input schema for YFinanceCathieWoodAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")
    growth_rate: float = Field(0.20, description="The growth rate of the company. The default is 0.20.")
    discount_rate: float = Field(0.15, description="The discount rate of the company. The default is 0.15.")
    terminal_multiple: float = Field(25, description="The terminal multiple of the company. The default is 25.")
    projection_years: int = Field(5, description="The number of years to project the company's growth. The default is 5.")


# Define the tool class
class YFinanceCathieWoodAnalysisTool(BaseTool):
    name: str = "YFinance Cathie Wood Analysis Tool"
    description: str = "Produces a Cathie Wood analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceCathieWoodAnalysisToolInput

    
    def _run(self, ticker: str, growth_rate: float = 0.20, discount_rate: float = 0.15, terminal_multiple: float = 25, projection_years: int = 5) -> Dict[str, Any]:
        """
        Produces a Cathie Wood analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :param growth_rate: The growth rate of the company. The default is 0.20.
        :param discount_rate: The discount rate of the company. The default is 0.15.
        :param terminal_multiple: The terminal multiple of the company. The default is 25.
        :param projection_years: The number of years to project the company's growth. The default is 5.
        :return: A dictionary containing the analysis results.
        """
        
        return calculate_cathie_wood_analysis_data(ticker, growth_rate, discount_rate, terminal_multiple, projection_years)



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceCathieWoodAnalysisTool()
    cathie_wood_analysis = tool_instance.run(ticker='MANU', growth_rate=0.20, discount_rate=0.15, terminal_multiple=25, projection_years=5)
    print(cathie_wood_analysis)