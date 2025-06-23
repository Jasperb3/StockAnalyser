from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from pprint import pprint
from typing import Type, List


# Define the input schema using Pydantic
class YFinanceCompetitorFinancialMetricsToolInput(BaseModel):
    """Input schema for YFinanceCompetitorFinancialMetricsTool."""
    tickers: List[str] = Field(..., description="A Python list of ticker symbols")

# Define the tool class
class YFinanceCompetitorFinancialMetricsTool(BaseTool):
    name: str = "YFinance Competitor Financial Metrics Tool"
    description: str = "Fetches detailed financial metrics for given tickers using yfinance."
    args_schema: Type[BaseModel] = YFinanceCompetitorFinancialMetricsToolInput

    def _run(self, tickers: List[str]) -> str:
        """
        Fetches financial data for given tickers using yfinance.
        :param tickers: List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'])
        :return: A string containing the competitors' financial KPIs.
        """
        # Fetch the stock data
        competitor_data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            
            # Get the info dictionary
            info = stock.info
            
            # Prepare the analysis results
            analysis = {
                'Ticker Symbol': ticker,
                'Company Name': info.get('displayName', info.get('shortName', info.get('longName', ticker))),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),

                # Price Information
                'Current Price': f"${info.get('currentPrice')}" if info.get('currentPrice') else 'N/A',

                # Valuation
                'P/E Ratio (Trailing)': info.get('trailingPE', 'N/A'),
                'Price to Sales Trailing 12 Months': f"${info.get('priceToSalesTrailing12Months'):.2f}" if info.get('priceToSalesTrailing12Months') else 'N/A',
                'P/B Ratio': info.get('priceToBook', 'N/A'),

                # Profitability
                'Net Margin': info.get('profitMargins', 'N/A'),

                # Returns
                'Return on Equity': info.get('returnOnEquity', 'N/A'),

                # Revenue Growth
                'Revenue Growth': info.get('revenueGrowth', 'N/A')

            }
            
            # Convert percentage values
            for key, value in analysis.items():
                if isinstance(value, float):
                    if key in ['Net Margin', 'Return on Equity', 'Revenue Growth']:
                        analysis[key] = str(round(value * 100, 2)) + '%'
            
            competitor_data[ticker] = analysis
        
        return competitor_data
    


if __name__ == "__main__":
    tool_instance = YFinanceCompetitorFinancialMetricsTool()
    analysis_data = tool_instance.run(tickers=['UNH', 'ELV', 'CVS', 'HUM', 'CNC'])
    pprint(analysis_data)