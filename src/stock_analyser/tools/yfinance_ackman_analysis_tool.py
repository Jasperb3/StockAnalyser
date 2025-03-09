from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from stock_analyser.tools.tool_utils.bill_ackman_analysis import calculate_bill_ackman_analysis_data
from typing import Type, Dict, Any


# Define the input schema using Pydantic
class YFinanceAckmanAnalysisToolInput(BaseModel):
    """Input schema for YFinanceAckmanAnalysisTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")
    growth_rate: float = Field(0.06, description="The growth rate of the company. Default is 0.06.")
    discount_rate: float = Field(0.10, description="The discount rate of the company. Default is 0.10.")
    terminal_multiple: float = Field(15, description="The terminal multiple of the company. Default is 15.")
    projection_years: int = Field(5, description="The number of years to project the company's cash flows. Default is 5.")


# Define the tool class
class YFinanceAckmanAnalysisTool(BaseTool):
    name: str = "YFinance Ackman Analysis Tool"
    description: str = "Produces a Bill Ackman analysis for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceAckmanAnalysisToolInput

    
    def _run(self, ticker: str, growth_rate: float = 0.06, discount_rate: float = 0.10, terminal_multiple: float = 15, projection_years: int = 5) -> Dict[str, Any]:
        """
        Produces a Bill Ackman analysis for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :param growth_rate: The growth rate of the company. Default is 0.06
        :param discount_rate: The discount rate of the company. Default is 0.10
        :param terminal_multiple: The terminal multiple of the company. Default is 15
        :param projection_years: The number of years to project the company's cash flows. Default is 5
        :return: A dictionary containing the analysis results.
        """
        
        return calculate_bill_ackman_analysis_data(ticker, growth_rate, discount_rate, terminal_multiple, projection_years)



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceAckmanAnalysisTool()
    ackman_analysis = tool_instance.run(ticker= 'NVDA', growth_rate=0.06, discount_rate=0.10, terminal_multiple=15, projection_years=5)
    print(ackman_analysis)