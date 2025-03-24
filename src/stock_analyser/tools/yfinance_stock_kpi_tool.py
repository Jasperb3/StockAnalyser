from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
import numpy as np
from datetime import datetime
from typing import Type
from stock_analyser.utils.convert_currency import convert_currency


# Define the input schema using Pydantic
class YFinanceStockKPIToolInput(BaseModel):
    """Input schema for YFinanceStockKPITool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceStockKPITool(BaseTool):
    name: str = "YFinance Stock KPI Tool"
    description: str = "Fetches detailed financial KPIs for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceStockKPIToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches financial data for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's financial KPIs.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)

        exchange_rate = convert_currency(ticker)
        
        # Get the data
        info = stock.info

        financials = stock.financials
        financials = financials.apply(lambda x: x * exchange_rate)

        balance_sheet = stock.balance_sheet
        balance_sheet = balance_sheet.apply(lambda x: x * exchange_rate)

        cashflow = stock.cashflow
        cashflow = cashflow.apply(lambda x: x * exchange_rate)

        # Get historical data
        history = stock.history(period="5y")
        history = history.apply(lambda x: x * exchange_rate)


        
        # Calculate 52-Week High/Low
        week_52_high = history['High'].tail(252).max() if not history.empty else None
        week_52_low = history['Low'].tail(252).min() if not history.empty else None
        
        # Calculate 5-year revenue growth rate
        five_yr_revenue_growth = None
        if financials is not None and not financials.empty and 'Total Revenue' in financials.index:
            revenue_5y = financials.loc['Total Revenue'].iloc[:5]
            if len(revenue_5y) >= 5 and revenue_5y.iloc[-1] != 0 and not revenue_5y.isnull().any():
                five_yr_revenue_growth = (revenue_5y.iloc[0] / revenue_5y.iloc[-1]) ** (1/5) - 1
        
        # Calculate 2-year revenue growth rate
        two_yr_revenue_growth = None
        if financials is not None and not financials.empty and 'Total Revenue' in financials.index:
            revenue_2y = financials.loc['Total Revenue'].iloc[:2]
            if len(revenue_2y) >= 2 and revenue_5y.iloc[-1] != 0 and not revenue_2y.isnull().any():
                two_yr_revenue_growth = (revenue_2y.iloc[0] / revenue_2y.iloc[-1]) ** (1/2) - 1

        # Calculate FCF growth

        fcf = cashflow.loc['Free Cash Flow']
        fcf_changes = fcf.pct_change(fill_method=None)
        for i in range(len(fcf_changes)):
            if not np.isnan(fcf_changes.iloc[i]):
                fcf_growth = fcf_changes.iloc[i]
                break
        fcf_growth_rate = f"{-fcf_growth:.2%}" if fcf_growth is not None else 'N/A'
        




        # Calculate Return on Invested Capital
        return_on_invested_capital = None
        try:
            invested_capital = balance_sheet.loc["Invested Capital"].iloc[0]
            operating_income = financials.loc["Operating Income"].iloc[0]
            tax_rate_for_calcs = financials.loc["Tax Rate For Calcs"].iloc[0]

            if (invested_capital is not None
                    and operating_income is not None
                    and invested_capital != 0):

                if tax_rate_for_calcs is None:
                    tax_provision = financials.loc["Tax Provision"].iloc[0]
                    pretax_income = financials.loc["Pretax Income"].iloc[0]
                    if (tax_provision is not None and pretax_income is not None
                            and pretax_income != 0):
                        estimated_tax_rate = tax_provision / pretax_income
                        nopat = operating_income * (1 - estimated_tax_rate)
                    else:
                        nopat = operating_income * (1 - 0.20)  # Default tax rate
                else:
                    nopat = operating_income * (1 - tax_rate_for_calcs)

                return_on_invested_capital = nopat / invested_capital
        except (KeyError, TypeError, ZeroDivisionError):
            return_on_invested_capital = None
        
        # Prepare the analysis results
        analysis = {
            'Date': datetime.now().strftime('%B %d, %Y'),
            'Ticker Symbol': ticker,
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),

            '💲': 'Price Information ->',
            'Current Price': f"${info.get('currentPrice', 'N/A')}" if info.get('currentPrice') else 'N/A',
            '52-Week High': f"${round(week_52_high, 2)}" if week_52_high else 'N/A',
            '52-Week Low': f"${round(week_52_low, 2)}" if week_52_low else 'N/A',
            '50 Day Average': f"${info.get('fiftyDayAverage', 'N/A'):.2f}" if info.get('fiftyDayAverage') else 'N/A',
            '50 Day Average Change': info.get('fiftyDayAverageChangePercent', 'N/A'),
            '200 Day Average': f"${info.get('twoHundredDayAverage', 'N/A'):.2f}" if info.get('twoHundredDayAverage') else 'N/A',
            '200 Day Average Change': info.get('twoHundredDayAverageChangePercent', 'N/A'),
            'Previous Close': f"${info.get('previousClose', 'N/A')}" if info.get('previousClose') else 'N/A',

            '💲': 'Valuation ->',
            'Market Cap': f"${info.get('marketCap') * exchange_rate:,}" if info.get('marketCap') else 'N/A',
            'Enterprise Value': f"${info.get('enterpriseValue') * exchange_rate:,}" if info.get('enterpriseValue') else 'N/A',
            'P/E Ratio (Trailing)': info.get('trailingPE', 'N/A'),
            'Forward P/E Ratio': info.get('forwardPE', 'N/A'),
            'Price to Sales Trailing 12 Months': f"${info.get('priceToSalesTrailing12Months') * exchange_rate :.2f}" if info.get('priceToSalesTrailing12Months') else 'N/A',
            'P/B Ratio': info.get('priceToBook', 'N/A'),
            'Book Value': f"${info.get('bookValue') * exchange_rate:,}" if info.get('bookValue') else 'N/A',
            'Enterprise to Revenue': info.get('enterpriseToRevenue', 'N/A'),
            'Enterprise to EBITDA': info.get('enterpriseToEbitda', 'N/A'),
            'Trailing EPS': info.get('trailingEps', 'N/A'),
            'Forward EPS': info.get('forwardEps', 'N/A'),

            '💲': 'Profitability ->',
            'Profit Margin': info.get('profitMargins', 'N/A'),
            'Operating Margin': info.get('operatingMargins', 'N/A'),
            'Gross Margins': info.get('grossMargins', 'N/A'),
            "EBITDA": info.get('ebitda', 'N/A'),
            "EBITDA Margin": info.get('ebitdaMargins', 'N/A'),

            '💲': 'Financial Health ->',
            'Total Cash': f"${info.get('totalCash') * exchange_rate:,}" if info.get('totalCash') else 'N/A',
            'Total Debt': f"${info.get('totalDebt') * exchange_rate:,}" if info.get('totalDebt') else 'N/A',
            'Debt-to-Equity Ratio': info.get('debtToEquity', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Quick Ratio': info.get('quickRatio', 'N/A'),
            'Operating Cashflow': f"${info.get('operatingCashflow') * exchange_rate:,}" if info.get('operatingCashflow') else 'N/A',
            'Free Cash Flow': f"${info.get('freeCashflow') * exchange_rate:,}" if info.get('freeCashflow') else 'N/A',

            '💲': 'Dividends ->',
            'Dividend Rate': info.get('dividendRate', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Trailing Annual Dividend Rate': info.get('trailingAnnualDividendRate', 'N/A'),
            'Trailing Annual Dividend Yield': info.get('trailingAnnualDividendYield', 'N/A'),
            'Payout Ratio': info.get('payoutRatio', 'N/A'),
            'Next Dividend Date': datetime.fromtimestamp(info.get('dividendDate')).strftime('%Y-%m-%d') if info.get('dividendDate') else 'N/A',
            'Ex-Dividend Date': datetime.fromtimestamp(info.get('exDividendDate')).strftime('%Y-%m-%d') if info.get('exDividendDate') else 'N/A',

            '📈': 'Growth ->',
            '5-Year Revenue Growth Rate': five_yr_revenue_growth,
            '2-Year Revenue Growth Rate': two_yr_revenue_growth,
            'Revenue Growth': info.get('revenueGrowth', 'N/A'),
            'Earnings Growth': info.get('earningsGrowth', 'N/A'),
            'Earnings Quarterly Growth': info.get('earningsQuarterlyGrowth', 'N/A'),
            'Free Cash Flow Growth': fcf_growth_rate,

            '👥': 'Ownership ->',
            'Shares Outstanding': f"{info.get('sharesOutstanding', 'N/A'):,}" if info.get('sharesOutstanding') else 'N/A',
            'Float Shares': f"{info.get('floatShares', 'N/A'):,}" if info.get('floatShares') else 'N/A',
            'Held Percent Insiders': info.get('heldPercentInsiders', 'N/A'),
            'Held Percent Institutions': info.get('heldPercentInstitutions', 'N/A'),

            '💲': 'Analyst Ratings and Targets ->',
            'Analyst Target Price': f"${info.get('targetMedianPrice', 'N/A')}" if info.get('targetMedianPrice') else 'N/A',
            'Average Analyst Rating': info.get('averageAnalystRating', 'N/A'),
            'Recommendation': info.get('recommendationKey', 'N/A').replace("_", " "),
            'Number of Analyst Opinions': info.get('numberOfAnalystOpinions', 'N/A'),

            '💰': 'Returns ->',
            'Return on Assets': info.get('returnOnAssets', 'N/A'),
            'Return on Equity': info.get('returnOnEquity', 'N/A'),
            'Return on Invested Capital': return_on_invested_capital,
            
            '🔍': 'Other ->',
            'Last Earnings Call': datetime.fromtimestamp(info.get('earningsTimestamp')).strftime('%Y-%m-%d') if info.get('earningsTimestamp') else 'N/A',
            'Next Earnings Call': f"between {datetime.fromtimestamp(info.get('earningsTimestampStart')).strftime('%Y-%m-%d')} and {datetime.fromtimestamp(info.get('earningsTimestampEnd')).strftime('%Y-%m-%d')}" if info.get('earningsTimestampStart') and info.get('earningsTimestampEnd') else 'N/A',
            'Audit Risk': info.get('auditRisk', 'N/A'),
            'Last Split Factor': info.get('lastSplitFactor', 'N/A'),
            'Last Split Date': datetime.fromtimestamp(info.get('lastSplitDate')).strftime('%Y-%m-%d') if info.get('lastSplitDate') else 'N/A',

        }
        
        # Convert percentage values
        for key, value in analysis.items():
            if isinstance(value, float):
                if key in ['200 Day Average Change', '50 Day Average Change', 'Dividend Yield', '5-Year Revenue Growth Rate',
                           'Profit Margin', 'Operating Margin', 'Earnings Growth', 'Revenue Growth', 'Return on Equity', 'Return on Invested Capital',
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
        
        
        # Format the analysis results for output
        output = "\n".join([f"{key}: {value}" for key, value in analysis.items()])
        
        return output

    # async def _arun(self, ticker: str) -> str:
    #     """Asynchronous version of the _run method."""
    #     return self._run(ticker)

# Example usage within CrewAI
if __name__ == "__main__":
    tool_instance = YFinanceStockKPITool()
    stock_kpi_data = tool_instance.run(ticker='ACN')
    print(stock_kpi_data)