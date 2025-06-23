from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime
from typing import Type

# Define the input schema using Pydantic
class YFinanceCompanyInfoToolInput(BaseModel):
    """Input schema for YFinanceCompanyInfoTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceCompanyInfoTool(BaseTool):
    name: str = "YFinance Company Info Tool"
    description: str = "Fetches detailed company and management info for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceCompanyInfoToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches company and management info for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the company info.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        
        # Get the info dictionary
        info = stock.info

        ceo = [officer for officer in info.get('companyOfficers', []) if 'ceo' in officer.get('title', '').lower() or 'chief executive officer' in officer.get('title', '').lower()]
        ceo_name = ceo[0].get('name', 'N/A') if ceo else 'N/A'
        ceo_age = ceo[0].get('age', 'N/A') if ceo else 'N/A'
        ceo_title = ceo[0].get('title', 'N/A') if ceo else 'N/A'
        ceo_pay = ceo[0].get('totalPay', 'N/A') if ceo else 'N/A'
        ceo_exercised_value = ceo[0].get('exercisedValue', 'N/A') if ceo else 'N/A'
        ceo_unexercised_value = ceo[0].get('unexercisedValue', 'N/A') if ceo else 'N/A'

        other_management = sorted([officer for officer in info.get('companyOfficers', []) if ceo_name.lower() not in officer.get('name', '').lower()][:5], key=lambda x: x.get('totalPay', 0), reverse=True)
        other_management_str = "\n".join([
            f"{officer.get('name', 'N/A').replace('  ', ' ')}. ({officer.get('title', 'N/A')}). "
            f"Total Pay: ${officer.get('totalPay', 'N/A'):,}" if isinstance(officer.get('totalPay', 'N/A'), (int, float))
            else f"{officer.get('name', 'N/A').replace('  ', ' ')}. ({officer.get('title', 'N/A')}). "
                 f"Total Pay: {officer.get('totalPay', 'N/A')}"
            for officer in other_management
        ])
        
        # Format CEO pay information with proper formatting
        ceo_pay_formatted = f"${ceo_pay:,}" if isinstance(ceo_pay, (int, float)) else f"{ceo_pay}"
        ceo_exercised_formatted = f"${ceo_exercised_value:,}" if isinstance(ceo_exercised_value, (int, float)) else f"{ceo_exercised_value}"
        ceo_unexercised_formatted = f"${ceo_unexercised_value:,}" if isinstance(ceo_unexercised_value, (int, float)) else f"{ceo_unexercised_value}"
        
        # Prepare the analysis results
        company_info = {
            'Ticker Symbol': ticker,
            'Company Name': info.get('displayName', info.get('shortName', info.get('longName', ticker))),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Website': info.get('website', 'N/A'),
            'Description': info.get('longBusinessSummary', 'N/A'),
            'Headquarters': info.get('address1', 'N/A'),
            'Founded': info.get('founded', 'N/A'),
            'Employees': info.get('fullTimeEmployees', 'N/A'),
            'CEO': f"{ceo_name.replace('  ', ' ')} ({ceo_title}), {ceo_age} years old, "
                   f"{ceo_pay_formatted} total pay, "
                   f"{ceo_exercised_formatted} exercised value, "
                   f"{ceo_unexercised_formatted} unexercised value",
            'Other Management': other_management_str
        }
        
        
        # Format the analysis results for output
        output = "\n".join([f"{key}: {value}" for key, value in company_info.items()])
        
        return output

    # async def _arun(self, ticker: str) -> str:
    #     """Asynchronous version of the _run method."""
    #     return self._run(ticker)

# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceCompanyInfoTool()
    nvidia_analysis = tool_instance.run(ticker='NVDA')
    print(nvidia_analysis)