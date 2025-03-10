from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime
from typing import Type

# Define the input schema using Pydantic
class YFinanceIndustryLeadersToolInput(BaseModel):
    """Input schema for YFinanceIndustryLeadersTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceIndustryLeadersTool(BaseTool):
    name: str = "YFinance Industry Leaders Tool"
    description: str = "Fetches information about the leading companies in an industry for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceIndustryLeadersToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches info on the leading companies in an industry for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing information about the leading companies in the industry to which the ticker belongs.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        
        # Get the industry
        industry = yf.Industry(stock.info['industryKey'])

        industry_name = industry.name if industry.name else "N/A"
        industry_overview = industry.overview
        description = industry_overview.get('description', 'No description available.')
        market_weight = industry_overview.get('market_weight', 'N/A')
        market_cap = industry_overview.get('market_cap', 'N/A')
        if isinstance(market_cap, (int, float)):
            if market_cap == 0:
                market_cap = "N/A"
            else:
                market_cap = f"${market_cap:,}"
        companies_count = industry_overview.get('companies_count', 'N/A')
        if isinstance(companies_count, (int, float)):
            if companies_count == 0:
                companies_count = 'N/A'
            else:
                companies_count = f"{companies_count:,}"

        employee_count = industry_overview.get('employee_count', 'N/A')
        if isinstance(employee_count, (int, float)):
            if employee_count == 0:
                employee_count = 'N/A'
            else:
                employee_count = f"{employee_count:,}"
        industry_top_companies = industry.top_companies.to_dict(orient='index') if industry.top_companies is not None else {}
        top_growth_companies = industry.top_growth_companies.to_dict(orient='index') if industry.top_growth_companies is not None else {}
        top_performing_companies = industry.top_performing_companies.to_dict(orient='index') if industry.top_performing_companies is not None else {}
        
        analysis = ""
        analysis += f"**Industry Name:** {industry_name}\n"
        analysis += f"**Industry Overview:**\n"
        analysis += f"Description: {description}\n"
        analysis += f"Market Weight: {market_weight}\n"
        analysis += f"Market Cap: {market_cap}\n"
        analysis += f"Companies Count: {companies_count}\n"
        analysis += f"Employee Count: {employee_count}\n"
        analysis += f"\n**Top Companies in the {industry_name.capitalize()} Industry:**\n"
        for ticker, company_data in industry_top_companies.items():
            company_name = company_data.get('name', 'N/A')
            rating = company_data.get('rating', 'N/A')
            market_weight_c = company_data.get('market weight', 'N/A')
            analysis += f"- **{company_name} ({ticker})**\n"
            analysis += f"  - Rating: {rating}\n"
            analysis += f"  - Market Weight: {market_weight_c}\n"
        analysis += f"\n**Top Growth Companies in the {industry_name.capitalize()} Industry:**\n"
        for ticker, company_data in top_growth_companies.items():
            company_name = company_data.get('name', 'N/A')
            ytd_return = company_data.get('ytd return', 'N/A')
            if isinstance(ytd_return, (int, float)):
                ytd_return = f"{ytd_return:.4f}"
            growth_estimate = company_data.get(' growth estimate', 'N/A')
            if isinstance(growth_estimate, (int, float)):
                growth_estimate = f"${growth_estimate:.2f}"

            analysis += f"- **{company_name} ({ticker})**\n"
            analysis += f"  - YTD Return: {ytd_return}\n"
            analysis += f"  - Growth Estimate: {growth_estimate}\n"
        analysis += f"\n**Top Performing Companies in the {industry_name.capitalize()} Industry:**\n"
        for ticker, company_data in top_performing_companies.items():
            company_name = company_data.get('name', 'N/A')
            ytd_return = company_data.get('ytd return', 'N/A')
            if isinstance(ytd_return, (int, float)):
                ytd_return = f"{ytd_return:.4f}"
            last_price = company_data.get(' last price', 'N/A')
            if isinstance(last_price, (int, float)):
                last_price = f"${last_price:.2f}"
            target_price = company_data.get('target price', 'N/A')
            if isinstance(target_price, (int, float)):
                target_price = f"${target_price:.2f}"
            analysis += f"- **{company_name} ({ticker})**\n"
            analysis += f"  - YTD Return: {ytd_return}\n"
            analysis += f"  - Last Price: {last_price}\n"
            analysis += f"  - Target Price: {target_price}\n"

        return analysis
    

# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceIndustryLeadersTool()
    nvidia_analysis = tool_instance.run(ticker='NVDA')
    print(nvidia_analysis)