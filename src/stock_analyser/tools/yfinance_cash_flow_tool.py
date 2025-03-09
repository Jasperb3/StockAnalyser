from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime
from typing import Type
import numpy as np

# Define the input schema using Pydantic
class YFinanceStockCashFlowToolInput(BaseModel):
    """Input schema for YFinanceStockCashFlowTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceStockCashFlowTool(BaseTool):
    name: str = "YFinance Stock Cash Flow Tool"
    description: str = "Fetches most recent cash flow data for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceStockCashFlowToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches cash flow data for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's cash flow data.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)

        # Get the info dictionary

        cashflow = stock.cashflow
        if cashflow.empty:
            return "No cash flow data available for this ticker."
        
        most_recent = cashflow.columns[0]

        date = datetime.strftime(most_recent, '%B %d, %Y')

        data = cashflow[most_recent].to_dict()

        # Format monetary values and other numbers
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if not np.isnan(value) and key in ['Free Cash Flow',
                           'Repurchase Of Capital Stock',
                           'Repayment Of Debt',
                           'Issuance Of Debt',
                           'Capital Expenditure',
                           'Interest Paid Supplemental Data',
                           'Income Tax Paid Supplemental Data',
                           'End Cash Position',
                           'Beginning Cash Position',
                           'Changes In Cash',
                           'Financing Cash Flow',
                           'Cash Flow From Continuing Financing Activities',
                           'Net Other Financing Charges',
                           'Proceeds From Stock Option Exercised',
                           'Cash Dividends Paid',
                           'Common Stock Dividend Paid',
                           'Net Common Stock Issuance',
                           'Common Stock Payments',
                           'Net Issuance Payments Of Debt',
                           'Net Long Term Debt Issuance',
                           'Long Term Debt Payments',
                           'Long Term Debt Issuance',
                           'Investing Cash Flow',
                           'Cash Flow From Continuing Investing Activities',
                           'Net Other Investing Changes',
                           'Net Investment Purchase And Sale',
                           'Sale Of Investment',
                           'Purchase Of Investment',
                           'Net Business Purchase And Sale',
                           'Purchase Of Business',
                           'Net PPE Purchase And Sale',
                           'Purchase Of PPE',
                           'Capital Expenditure Reported',
                           'Operating Cash Flow',
                           'Cash Flow From Continuing Operating Activities',
                           'Change In Working Capital',
                           'Change In Other Working Capital',
                           'Change In Other Current Liabilities',
                           'Change In Payables And Accrued Expense',
                           'Change In Accrued Expense',
                           'Change In Payable',
                           'Change In Account Payable',
                           'Change In Prepaid Assets',
                           'Change In Inventory',
                           'Change In Receivables',
                           'Changes In Account Receivables',
                           'Other Non Cash Items',
                           'Stock Based Compensation',
                           'Deferred Tax',
                           'Deferred Income Tax',
                           'Depreciation Amortization Depletion',
                           'Depreciation And Amortization',
                           'Amortization Cash Flow',
                           'Amortization Of Intangibles',
                           'Depreciation',
                           'Operating Gains Losses',
                           'Gain Loss On Investment Securities',
                           'Net Income From Continuing Operations']:
                    data[key] = f"${value:,.2f}"

        # Format the analysis results for output
        output = f"Date of cash flow data: {date}\n" + "\n".join([f"{key}: {value}" for key, value in data.items()])
        
        return output



# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceStockCashFlowTool()
    nvidia_analysis = tool_instance.run(ticker='NVDA')
    print(nvidia_analysis)