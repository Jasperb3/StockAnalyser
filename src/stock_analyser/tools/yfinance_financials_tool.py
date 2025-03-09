from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime
from typing import Type
import numpy as np
# Define the input schema using Pydantic
class YFinanceStockFinancialsToolInput(BaseModel):
    """Input schema for YFinanceStockFinancialsTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceStockFinancialsTool(BaseTool):
    name: str = "YFinance Stock Financials Tool"
    description: str = "Fetches most recent financial data for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceStockFinancialsToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches financial data for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's financial data.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)

        # Get the info dictionary
        most_recent = stock.financials.columns[0]

        date = datetime.strftime(most_recent, '%B %d, %Y')

        data = stock.financials[most_recent].to_dict()

        # Convert percentage values
        for key, value in data.items():
            if isinstance(value, float):
                if not np.isnan(value) and key in ['Tax Rate For Calcs']:
                    data[key] = str(round(value * 100, 2)) + '%'

        # Format monetary values and other numbers
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if not np.isnan(value) and key in ['Total Revenue', 'Cost Of Revenue', 'Gross Profit',
                           'Operating Expense', 'Selling General And Administration',
                           'Research And Development', 'Operating Income',
                           'Interest Income Non Operating', 'Interest Expense Non Operating',
                           'Net Non Operating Interest Income Expense',
                           'Restructuring And Mergern Acquisition', 'Special Income Charges',
                           'Other Non Operating Income Expenses', 'Other Income Expense',
                           'Pretax Income', 'Tax Provision', 'Net Income Continuous Operations',
                           'Net Income Including Noncontrolling Interests', 'Net Income',
                           'Net Income Common Stockholders', 'Diluted NI Availto Com Stockholders',
                           'Net Income From Continuing And Discontinued Operation',
                           'Normalized Income', 'Interest Income', 'Interest Expense',
                           'Net Interest Income', 'EBIT', 'EBITDA', 'Reconciled Cost Of Revenue',
                           'Reconciled Depreciation', 'Net Income From Continuing Operation Net Minority Interest',
                           'Total Unusual Items Excluding Goodwill', 'Total Unusual Items',
                           'Normalized EBITDA', 'Tax Effect Of Unusual Items', 'Total Expenses',
                           'Total Operating Income As Reported', 'Diluted EPS', 'Basic EPS']:
                    data[key] = f"${value:,.2f}"
                elif key in ['Diluted Average Shares', 'Basic Average Shares']:
                    data[key] = f"{value:,.0f}"

        # Format the analysis results for output
        output = f"Date of financials: {date}\n" + "\n".join([f"{key}: {value}" for key, value in data.items()])
        
        return output



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceStockFinancialsTool()
    nvidia_analysis = tool_instance.run(ticker='NVDA')
    print(nvidia_analysis)