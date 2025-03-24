from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime, timedelta
from stock_analyser.utils.convert_currency import convert_currency
from typing import Type
import pandas as pd

# Define the input schema using Pydantic
class YFinanceIncomeToolInput(BaseModel):
    """Input schema for YFinanceIncomeTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")
    years: int = Field(1, description="Number of years to fetch income statement for. Default is 1.")
    
# Define the tool class
class YFinanceIncomeTool(BaseTool):
    name: str = "YFinance Income Tool"
    description: str = "Fetches detailed income statement for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceIncomeToolInput

    def _run(self, ticker: str, years: int = 1) -> str:
        """
        Fetches income statement data for a given ticker for the last n years using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :param years: Number of years to fetch income statement for. Default is 1.
        :return: A string containing the stock's income statement.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)

        exchange_rate = convert_currency(ticker)

        income_statement = stock.income_stmt
        if income_statement is not None and not income_statement.empty and (income_statement != 0).any().any():
            income_statement = income_statement.map(lambda x: x * exchange_rate if isinstance(x, (int, float)) else x)

        # Convert the income statement to a human-readable format
        output = f"\nIncome statement for {ticker} for the last {years} years:\n\n"
        for date in income_statement.columns:
            formatted_date = date.strftime('%Y-%m-%d')
            cutoff_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
            if formatted_date >= cutoff_date:
                output += f"**Date: {formatted_date}**\n"  # Format the date
                for index, row in income_statement.iterrows():
                    value = row[date]
                    if value is None or pd.isna(value):
                        formatted_value = "Data Not Available"
                    elif isinstance(value, float) or isinstance(value, int):
                        if pd.isna(value):  # Check for NaN values
                            formatted_value = "N/A"
                        elif value == 0:
                            formatted_value = "Zero Value Reported"
                        elif index.lower().endswith("eps"):  # Format EPS with 3 decimal places
                            formatted_value = f"{value:.3f}"
                        elif index.lower().find("shares") != -1: # Format share numbers without any decimals or currency
                            formatted_value = f"{value:,.0f}"
                        else: # Assume it is currency and format it with thousands separators and 2 decimal places
                            formatted_value = f"${value:,.2f}"
                    else:
                        formatted_value = str(value)  # Keep as string if not a number

                    output += f"- {index}: {formatted_value}\n"
                output += "\n"

        return output

# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceIncomeTool()
    nvidia_analysis = tool_instance.run(ticker='AAPL', years=3)
    print(nvidia_analysis)