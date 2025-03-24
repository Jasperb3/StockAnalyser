from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from pprint import pprint
from datetime import datetime
from typing import Type, List
from stock_analyser.utils.convert_currency import convert_currency

# Define the input schema using Pydantic
class YFinanceCompetitorKPIsToolInput(BaseModel):
    """Input schema for YFinanceCompetitorKPIsTool."""
    tickers: List[str] = Field(..., description="List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'] for Apple Inc., NVIDIA Corp., and Alphabet Inc.)")

# Define the tool class
class YFinanceCompetitorKPIsTool(BaseTool):
    name: str = "YFinance Competitor KPIs Tool"
    description: str = "Fetches detailed financial KPIs for given tickers using yfinance."
    args_schema: Type[BaseModel] = YFinanceCompetitorKPIsToolInput

    def _run(self, tickers: List[str]) -> str:
        """
        Fetches financial data for given tickers using yfinance.
        :param tickers: List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'] for Apple Inc., NVIDIA Corp., and Alphabet Inc.)
        :return: A string containing the competitors' financial KPIs.
        """
        # Fetch the stock data
        competitor_data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            
            # Get the info dictionary
            info = stock.info

            exchange_rate = convert_currency(ticker)
            
            # Get historical data
            history = stock.history(period="5y")
        
            # Calculate 52-Week High/Low
            week_52_high = history['High'].tail(252).max()
            week_52_low = history['Low'].tail(252).min()
        
            # Calculate 5-year revenue growth rate
            financials = stock.financials
            if not financials.empty and 'Total Revenue' in financials.index:
                revenue_5y = financials.loc['Total Revenue'].iloc[:5]
                revenue_growth = (revenue_5y.iloc[0] / revenue_5y.iloc[-1]) ** (1/5) - 1 if len(revenue_5y) >= 5 else None
            else:
                revenue_growth = None
            
            # Prepare the analysis results
            analysis = {
                'Ticker Symbol': ticker,
                'Company Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),

                # Price Information
                'Current Price': f"${info.get('currentPrice')}" if info.get('currentPrice') else 'N/A',
                '52-Week High': f"${week_52_high}" if week_52_high else 'N/A',
                '52-Week Low': f"${week_52_low}" if week_52_low else 'N/A',
                '50 Day Average': f"${info.get('fiftyDayAverage')}" if info.get('fiftyDayAverage') else 'N/A',
                '50 Day Average Change': info.get('fiftyDayAverageChangePercent', 'N/A'),
                '200 Day Average': f"${info.get('twoHundredDayAverage')}" if info.get('twoHundredDayAverage') else 'N/A',
                '200 Day Average Change': info.get('twoHundredDayAverageChangePercent', 'N/A'),
                'Previous Close': f"${info.get('previousClose')}" if info.get('previousClose') else 'N/A',

                # Valuation
                'Market Cap': f"${info.get('marketCap'):,}" if info.get('marketCap') else 'N/A',
                'Enterprise Value': f"${info.get('enterpriseValue'):,}" if info.get('enterpriseValue') else 'N/A',
                'P/E Ratio (Trailing)': info.get('trailingPE', 'N/A'),
                'Forward P/E Ratio': info.get('forwardPE', 'N/A'),
                'Price to Sales Trailing 12 Months': f"${info.get('priceToSalesTrailing12Months'):.2f}" if info.get('priceToSalesTrailing12Months') else 'N/A',
                'P/B Ratio': info.get('priceToBook', 'N/A'),
                'Enterprise to Revenue': info.get('enterpriseToRevenue', 'N/A'),
                'Enterprise to EBITDA': info.get('enterpriseToEbitda', 'N/A'),
                'Trailing EPS': info.get('trailingEps', 'N/A'),
                'Forward EPS': info.get('forwardEps', 'N/A'),

                # Profitability
                'Profit Margin': info.get('profitMargins', 'N/A'),
                'Operating Margin': info.get('operatingMargins', 'N/A'),
                'Gross Margins': info.get('grossMargins', 'N/A'),
                "EBITDA": info.get('ebitda', 'N/A'),
                "EBITDA Margin": info.get('ebitdaMargins', 'N/A'),

                # Financial Health
                'Total Cash': f"${info.get('totalCash') * exchange_rate:,.0f}" if info.get('totalCash') else 'N/A',
                'Total Debt': f"${info.get('totalDebt') * exchange_rate:,.0f}" if info.get('totalDebt') else 'N/A',
                'Debt-to-Equity Ratio': info.get('debtToEquity', 'N/A'),
                'Current Ratio': info.get('currentRatio', 'N/A'),
                'Quick Ratio': info.get('quickRatio', 'N/A'),
                'Operating Cashflow': f"${info.get('operatingCashflow') * exchange_rate:,.0f}" if info.get('operatingCashflow') else 'N/A',
                'Free Cash Flow': f"${info.get('freeCashflow') * exchange_rate:,.0f}" if info.get('freeCashflow') else 'N/A',

                # Dividends
                'Dividend Rate': info.get('dividendRate', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                'Trailing Annual Dividend Rate': info.get('trailingAnnualDividendRate', 'N/A'),
                'Trailing Annual Dividend Yield': info.get('trailingAnnualDividendYield', 'N/A'),
                'Payout Ratio': info.get('payoutRatio', 'N/A'),
                'Next Dividend Date': datetime.fromtimestamp(info.get('dividendDate')).strftime('%Y-%m-%d') if info.get('dividendDate') else 'N/A',
                'Ex-Dividend Date': datetime.fromtimestamp(info.get('exDividendDate')).strftime('%Y-%m-%d') if info.get('exDividendDate') else 'N/A',

                # Growth
                '5-Year Revenue Growth Rate': revenue_growth,
                'Revenue Growth': info.get('revenueGrowth', 'N/A'),
                'Earnings Growth': info.get('earningsGrowth', 'N/A'),
                'Earnings Quarterly Growth': info.get('earningsQuarterlyGrowth', 'N/A'),

                # Ownership
                'Shares Outstanding': f"{info.get('sharesOutstanding', 'N/A'):,}" if info.get('sharesOutstanding') else 'N/A',
                'Float Shares': f"{info.get('floatShares', 'N/A'):,}" if info.get('floatShares') else 'N/A',
                'Held Percent Insiders': info.get('heldPercentInsiders', 'N/A'),
                'Held Percent Institutions': info.get('heldPercentInstitutions', 'N/A'),

                # Analyst Ratings and Targets
                'Analyst Target Price': f"${info.get('targetMedianPrice')}" if info.get('targetMedianPrice') else 'N/A',
                'Average Analyst Rating': info.get('averageAnalystRating', 'N/A'),
                'Recommendation': info.get('recommendationKey', 'N/A'),
                'Number of Analyst Opinions': info.get('numberOfAnalystOpinions', 'N/A'),

                # Returns
                'Return on Assets': info.get('returnOnAssets', 'N/A'),
                'Return on Equity': info.get('returnOnEquity', 'N/A')

            }
            
            # Convert percentage values
            for key, value in analysis.items():
                if isinstance(value, float):
                    if key in ['200 Day Average Change', '50 Day Average Change', 'Dividend Yield', '5-Year Revenue Growth Rate',
                            'Profit Margin', 'Operating Margin', 'Earnings Growth', 'Revenue Growth', 'Return on Equity',
                            'EBITDA Margin', 'Trailing Annual Dividend Yield', 'Payout Ratio', 'Held Percent Insiders',
                            'Held Percent Institutions', 'Earnings Quarterly Growth','Return on Assets', 'Gross Margins']:
                        analysis[key] = str(round(value * 100, 2)) + '%'
                elif isinstance(value, str):
                    if key in ['Held Percent Insiders', 'Held Percent Institutions']:
                        try:
                            # Check if the value can be converted to a float before multiplying by 100
                            num_value = float(value)
                            analysis[key] = str(round(num_value * 100, 2)) + '%'
                        except ValueError:
                            pass  # Keep original value if not convertible to float
            
            competitor_data[ticker] = analysis
        
        return competitor_data
    


if __name__ == "__main__":
    tool_instance = YFinanceCompetitorKPIsTool()
    nvidia_analysis = tool_instance.run(tickers=['NVDA', 'MANU', 'HMC', 'SKBSY'])
    pprint(nvidia_analysis)