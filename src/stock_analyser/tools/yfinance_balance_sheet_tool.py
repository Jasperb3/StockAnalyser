from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime
from typing import Type
import numpy as np


# Define the input schema using Pydantic
class YFinanceStockBalanceSheetToolInput(BaseModel):
    """Input schema for YFinanceStockBalanceSheetTool."""
    ticker: str = Field(..., description="Stock ticker symbol for the company.")

# Define the tool class
class YFinanceStockBalanceSheetTool(BaseTool):
    name: str = "YFinance Stock Balance Sheet Tool"
    description: str = "Fetches most recent balance sheet data for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceStockBalanceSheetToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches balance sheet data for a given ticker using yfinance.
        :param ticker: Stock ticker symbol for the company.
        :return: A string containing the stock's balance sheet data.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)

        # Get the info dictionary
        most_recent = stock.balance_sheet.columns[0]

        date = datetime.strftime(most_recent, '%B %d, %Y')

        data = stock.balance_sheet[most_recent].to_dict()

        # Format monetary values and other numbers
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if not np.isnan(value) and key in ['Net Debt',
                           'Total Debt',
                           'Tangible Book Value',
                           'Invested Capital',
                           'Working Capital',
                           'Net Tangible Assets',
                           'Capital Lease Obligations',
                           'Common Stock Equity',
                           'Total Capitalization',
                           'Total Equity Gross Minority Interest',
                           'Stockholders Equity',
                           'Gains Losses Not Affecting Retained Earnings',
                           'Other Equity Adjustments',
                           'Retained Earnings',
                           'Capital Stock',
                           'Common Stock',
                           'Total Liabilities Net Minority Interest',
                           'Total Non Current Liabilities Net Minority Interest',
                           'Other Non Current Liabilities',
                           'Tradeand Other Payables Non Current',
                           'Long Term Debt And Capital Lease Obligation',
                           'Long Term Capital Lease Obligation',
                           'Long Term Debt',
                           'Current Liabilities',
                           'Other Current Liabilities',
                           'Current Deferred Liabilities',
                           'Current Deferred Revenue',
                           'Current Debt And Capital Lease Obligation',
                           'Current Capital Lease Obligation',
                           'Current Debt',
                           'Other Current Borrowings',
                           'Commercial Paper',
                           'Payables And Accrued Expenses',
                           'Payables',
                           'Total Tax Payable',
                           'Income Tax Payable',
                           'Accounts Payable',
                           'Total Assets',
                           'Total Non Current Assets',
                           'Other Non Current Assets',
                           'Non Current Deferred Assets',
                           'Non Current Deferred Taxes Assets',
                           'Investments And Advances',
                           'Other Investments',
                           'Investmentin Financial Assets',
                           'Available For Sale Securities',
                           'Net PPE',
                           'Accumulated Depreciation',
                           'Gross PPE',
                           'Leases',
                           'Other Properties',
                           'Machinery Furniture Equipment',
                           'Land And Improvements',
                           'Properties',
                           'Current Assets',
                           'Other Current Assets',
                           'Inventory',
                           'Receivables',
                           'Other Receivables',
                           'Accounts Receivable',
                           'Cash Cash Equivalents And Short Term Investments',
                           'Other Short Term Investments',
                           'Cash And Cash Equivalents',
                           'Cash Equivalents',
                           'Cash Financial']:
                    data[key] = f"${value:,.2f}"
                elif key in ['Treasury Shares Number',
                             'Ordinary Shares Number',
                             'Share Issued',
                             'Diluted Average Shares',
                             'Basic Average Shares']:
                    data[key] = f"{value:,.0f}"

        # Format the analysis results for output
        output = f"Date of balance sheet: {date}\n" + "\n".join([f"{key}: {value}" for key, value in data.items()])
        
        return output



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceStockBalanceSheetTool()
    nvidia_analysis = tool_instance.run(ticker='NVDA')
    print(nvidia_analysis)