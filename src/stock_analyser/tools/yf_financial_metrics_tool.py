from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
import numpy as np
from datetime import datetime
from stock_analyser.utils.convert_currency import convert_currency
from typing import Type


# Define the input schema using Pydantic
class YFinanceFinancialMetricsToolInput(BaseModel):
    """Input schema for FinancialMetricsTool."""
    ticker: str = Field(..., description="Stock ticker symbol for the company to analyze")

# Define the tool class
class YFinanceFinancialMetricsTool(BaseTool):
    name: str = "YFinance Financial Metrics Tool"
    description: str = "Fetches detailed financial data for a given ticker."
    args_schema: Type[BaseModel] = YFinanceFinancialMetricsToolInput

    def _run(self, ticker: str) -> str:
        """
        Fetches key financial data from YFinance for a given ticker.
        :param ticker: Stock ticker symbol for the company to analyze.
        :return: A string containing the stock's financial data.
        """
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        
        # Get the info dictionary
        info = stock.info

        exchange_rate = convert_currency(ticker)

        # Get the balance sheet data
        balance_sheet = stock.balance_sheet
        if not balance_sheet.empty:
            balance_sheet = balance_sheet.apply(lambda x: x * exchange_rate)
            most_recent_balance_sheet = balance_sheet.columns[0]
            balance_sheet_data = balance_sheet[most_recent_balance_sheet].to_dict()

        # Get the cash flow data
        cashflow = stock.cashflow
        if not cashflow.empty:
            cashflow = cashflow.apply(lambda x: x * exchange_rate)
            most_recent_cashflow = cashflow.columns[0]
            cash_flow_data = cashflow[most_recent_cashflow].to_dict()

        # Get the financials data
        financials = stock.financials
        if not financials.empty:
            financials = financials.apply(lambda x: x * exchange_rate)
            most_recent_financials = financials.columns[0]
            financials_data = financials[most_recent_financials].to_dict()

        
        # INCOME METRICS

        # net_income
        net_income = info.get('netIncome')
        if net_income not in [None, 'N/A', 'nan']:
            net_income *= exchange_rate
        else:
            net_income = financials_data.get('Net Income', 'N/A')
        net_income_string = f"Net Income: ${net_income:,.2f}" if isinstance(net_income, (float, int)) else "Net Income: N/A"

        # total_revenue
        total_revenue = info.get('totalRevenue')
        if total_revenue not in [None, 'N/A', 'nan']:
            total_revenue *= exchange_rate
        else:
            total_revenue = financials_data.get('Total Revenue', 'N/A')
        total_revenue_string = f"Total Revenue: ${total_revenue:,.2f}" if isinstance(total_revenue, (float, int)) else "Total Revenue: N/A"

        # gross_profit
        gross_profit = info.get('grossProfit')
        if gross_profit not in [None, 'N/A', 'nan']:
            gross_profit *= exchange_rate
        else:
            gross_profit = financials_data.get('Gross Profit', 'N/A')
        gross_profit_string = f"Gross Profit: ${gross_profit:,.2f}" if isinstance(gross_profit, (float, int)) else "Gross Profit: N/A"

        # operating_income
        operating_income = info.get('operatingIncome')
        if operating_income not in [None, 'N/A', 'nan']:
            operating_income *= exchange_rate
        else:
            operating_income = financials_data.get('Operating Income', 'N/A')
        operating_income_string = f"Operating Income: ${operating_income:,.2f}" if isinstance(operating_income, (float, int)) else "Operating Income: N/A"

        # ebitda
        ebitda = info.get('ebitda')
        if ebitda not in [None, 'N/A', 'nan']:
            ebitda *= exchange_rate
        else:
            ebitda = financials_data.get('EBITDA', 'N/A')
        ebitda_string = f"EBITDA: ${ebitda:,.2f}" if isinstance(ebitda, (float, int)) else "EBITDA: N/A"

        # basic_eps
        basic_eps = info.get('Basic EPS')
        if basic_eps not in [None, 'N/A', 'nan']:
            basic_eps *= exchange_rate
        else:
            basic_eps = financials_data.get('basicEps', 'N/A')
        basic_eps_string = f"Basic EPS: {basic_eps}" if isinstance(basic_eps, (float, int)) else "Basic EPS: N/A"

        # diluted_eps
        diluted_eps = info.get('Diluted EPS')
        if diluted_eps not in [None, 'N/A', 'nan']:
            diluted_eps *= exchange_rate
        else:
            diluted_eps = financials_data.get('dilutedEps', 'N/A')
        diluted_eps_string = f"Diluted EPS: {diluted_eps}" if isinstance(diluted_eps, (float, int)) else "Diluted EPS: N/A"

        # gross_margins
        gross_margins = info.get('grossMargins', 'N/A')
        if gross_margins not in [None, 'N/A', 'nan']:
            gross_margins *= exchange_rate
        gross_margins_string = f"Gross Margins: {gross_margins:.2%}" if isinstance(gross_margins, (float, int)) else "Gross Margins: N/A"

        # operating_margin
        operating_margin = info.get('operatingMargins', 'N/A')
        if operating_margin not in [None, 'N/A', 'nan']:
            operating_margin *= exchange_rate
        operating_margin_string = f"Operating Margin: {operating_margin:.2%}" if isinstance(operating_margin, (float, int)) else "Operating Margin: N/A"

        # profit_margin
        profit_margin = info.get('profitMargins', 'N/A')
        if profit_margin not in [None, 'N/A', 'nan']:
            profit_margin *= exchange_rate
        profit_margin_string = f"Profit Margin: {profit_margin:.2%}" if isinstance(profit_margin, (float, int)) else "Profit Margin: N/A"
        profit_margin_string = f"Profit Margin: {profit_margin:.2%}" if isinstance(profit_margin, (float, int)) else "Profit Margin: N/A"



        # BALANCE SHEET METRICS

        # total_assets
        total_assets = balance_sheet_data.get('Total Assets', 'N/A')
        total_assets_string = f"Total Assets: ${total_assets:,.2f}" if isinstance(total_assets, (float, int)) else "Total Assets: N/A"

        # total_liabilities
        total_liabilities = balance_sheet_data.get('Total Liabilities Net Minority Interest', 'N/A')
        total_liabilities_string = f"Total Liabilities: ${total_liabilities:,.2f}" if isinstance(total_liabilities, (float, int)) else "Total Liabilities: N/A"

        # total_equity
        total_equity = balance_sheet_data.get('Stockholders Equity', 'N/A')
        total_equity_string = f"Total Equity: ${total_equity:,.2f}" if isinstance(total_equity, (float, int)) else "Total Equity: N/A"

        # total_debt
        total_debt = info.get('totalDebt')
        if total_debt not in [None, 'N/A', 'nan']:
            total_debt *= exchange_rate
        else:
            total_debt = balance_sheet_data.get('Total Debt', 'N/A')
        total_debt_string = f"Total Debt: ${total_debt:,.2f}" if isinstance(total_debt, (float, int)) else "Total Debt: N/A"

        # net_debt
        net_debt = info.get('netDebt')
        if net_debt not in [None, 'N/A', 'nan']:
            net_debt *= exchange_rate
        else:
            net_debt = balance_sheet_data.get('Net Debt', 'N/A')
        net_debt_string = f"Net Debt: ${net_debt:,.2f}" if isinstance(net_debt, (float, int)) else "Net Debt: N/A"

        # cash_and_equivalents
        cash_and_equivalents = balance_sheet_data.get('Cash And Cash Equivalents', 'N/A')
        cash_and_equivalents_string = f"Cash and Equivalents: ${cash_and_equivalents:,.2f}" if isinstance(cash_and_equivalents, (float, int)) else "Cash and Equivalents: N/A"

        # working_capital
        working_capital = balance_sheet_data.get('Working Capital', 'N/A')
        working_capital_string = f"Working Capital: ${working_capital:,.2f}" if isinstance(working_capital, (float, int)) else "Working Capital: N/A"
        
        # current_ratio
        current_ratio = info.get('currentRatio', 'N/A')
        current_ratio_string = f"Current Ratio: {current_ratio:.2f}" if isinstance(current_ratio, (float, int)) else "Current Ratio: N/A"

        # quick_ratio
        quick_ratio = info.get('quickRatio', 'N/A')
        quick_ratio_string = f"Quick Ratio: {quick_ratio:.2f}" if isinstance(quick_ratio, (float, int)) else "Quick Ratio: N/A"

        # debt_to_equity
        debt_to_equity = info.get('debtToEquity', 'N/A')
        debt_to_equity_string = f"Debt to Equity: {debt_to_equity:.2f}" if isinstance(debt_to_equity, (float, int)) else "Debt to Equity: N/A"

        # retained_earnings
        retained_earnings = balance_sheet_data.get('Retained Earnings', 'N/A')
        retained_earnings_string = f"Retained Earnings: ${retained_earnings:,.2f}" if isinstance(retained_earnings, (float, int)) else "Retained Earnings: N/A"



        # CASH FLOW METRICS

        # total_cash
        total_cash = info.get('totalCash')
        if total_cash not in [None, 'N/A', 'nan']:
            total_cash *= exchange_rate
        else:
            total_cash = balance_sheet_data.get('Cash And Cash Equivalents', 'N/A')
        total_cash_string = f"Total Cash: ${total_cash:,.2f}" if isinstance(total_cash, (float, int)) else "Total Cash: N/A"

        # free_cash_flow
        free_cash_flow = info.get('freeCashflow')
        if free_cash_flow not in [None, 'N/A', 'nan']:
            free_cash_flow *= exchange_rate
        else:
            free_cash_flow = cash_flow_data.get('Free Cash Flow', 'N/A')
        free_cash_flow_string = f"Free Cash Flow: ${free_cash_flow:,.2f}" if isinstance(free_cash_flow, (float, int)) else "Free Cash Flow: N/A"

        # operating_cash_flow
        operating_cash_flow = info.get('operatingCashflow')
        if operating_cash_flow not in [None, 'N/A', 'nan']:
            operating_cash_flow *= exchange_rate
        else:
            operating_cash_flow = cash_flow_data.get('Operating Cash Flow', 'N/A')
        operating_cash_flow_string = f"Operating Cash Flow: ${operating_cash_flow:,.2f}" if isinstance(operating_cash_flow, (float, int)) else "Operating Cash Flow: N/A"

        # capital_expenditure
        capital_expenditure = cash_flow_data.get('Capital Expenditure', 'N/A')
        capital_expenditure_string = f"Capital Expenditure: ${capital_expenditure:,.2f}" if isinstance(capital_expenditure, (float, int)) else "Capital Expenditure: N/A"

        # repayment_of_debt
        repayment_of_debt = cash_flow_data.get('Repayment Of Debt', 'N/A')
        repayment_of_debt_string = f"Repayment of Debt: ${repayment_of_debt:,.2f}" if isinstance(repayment_of_debt, (float, int)) else "Repayment of Debt: N/A"

        # debt_issuance
        debt_issuance = cash_flow_data.get('Issuance Of Debt', 'N/A')
        debt_issuance_string = f"Debt Issuance: ${debt_issuance:,.2f}" if isinstance(debt_issuance, (float, int)) else "Debt Issuance: N/A"

        # cash_changes
        cash_changes = cash_flow_data.get('Changes In Cash', 'N/A')
        cash_changes_string = f"Cash Changes: ${cash_changes:,.2f}" if isinstance(cash_changes, (float, int)) else "Cash Changes: N/A"

        # investing_cash_flow
        investing_cash_flow = cash_flow_data.get('Investing Cash Flow', 'N/A')
        investing_cash_flow_string = f"Investing Cash Flow: ${investing_cash_flow:,.2f}" if isinstance(investing_cash_flow, (float, int)) else "Investing Cash Flow: N/A"

        # financing_cash_flow
        financing_cash_flow = cash_flow_data.get('Financing Cash Flow', 'N/A')
        financing_cash_flow_string = f"Financing Cash Flow: ${financing_cash_flow:,.2f}" if isinstance(financing_cash_flow, (float, int)) else "Financing Cash Flow: N/A"

        # repurchases_of_stock
        repurchases_of_stock = cash_flow_data.get('Repurchase Of Capital Stock', 'N/A')
        repurchases_of_stock_string = f"Repurchases of Stock: ${repurchases_of_stock:,.2f}" if isinstance(repurchases_of_stock, (float, int)) else "Repurchases of Stock: N/A"

        # depreciation_and_amortization
        depreciation_and_amortization = cash_flow_data.get('Depreciation And Amortization', 'N/A')
        depreciation_and_amortization_string = f"Depreciation and Amortization: ${depreciation_and_amortization:,.2f}" if isinstance(depreciation_and_amortization, (float, int)) else "Depreciation and Amortization: N/A"

        # GROWTH METRICS

        # revenue_growth
        revenue_growth = info.get('revenueGrowth', 'N/A')
        revenue_growth_string = f"Latest Revenue Growth: {revenue_growth:.2%}" if isinstance(revenue_growth, (float, int)) else "Latest Revenue Growth: N/A"

        # earnings_growth
        earnings_growth = info.get('earningsGrowth', 'N/A')
        earnings_growth_string = f"Latest Earnings Growth: {earnings_growth:.2%}" if isinstance(earnings_growth, (float, int)) else "Latest Earnings Growth: N/A"

        # five_year_revenue_growth_rate
        five_yr_revenue_growth = None
        revenue_5y = financials.loc['Total Revenue'].iloc[:5]
        if len(revenue_5y) >= 5 and revenue_5y.iloc[-1] != 0 and not revenue_5y.isnull().any():
            five_yr_revenue_growth = (revenue_5y.iloc[0] / revenue_5y.iloc[-1]) ** (1/5) - 1
        five_year_revenue_growth_rate_string = f"Five Year Revenue Growth Rate: {five_yr_revenue_growth:.2%}" if isinstance(five_yr_revenue_growth, (float, int)) else "Five Year Revenue Growth Rate: N/A"

        # two_year_revenue_growth_rate
        two_yr_revenue_growth = None
        if financials is not None and not financials.empty and 'Total Revenue' in financials.index:
            revenue_2y = financials.loc['Total Revenue'].iloc[:2]
            if len(revenue_2y) >= 2 and revenue_2y.iloc[-1] != 0 and not revenue_2y.isnull().any():
                two_yr_revenue_growth = (revenue_2y.iloc[0] / revenue_2y.iloc[-1]) ** (1/2) - 1
        two_year_revenue_growth_rate_string = f"Two Year Revenue Growth Rate: {two_yr_revenue_growth:.2%}" if isinstance(two_yr_revenue_growth, (float, int)) else "Two Year Revenue Growth Rate: N/A"

        # free_cash_flow_growth
        fcf = cashflow.loc['Free Cash Flow']
        fcf_changes = fcf.pct_change(fill_method=None)
        for i in range(len(fcf_changes)):
            if not np.isnan(fcf_changes.iloc[i]):
                fcf_growth = fcf_changes.iloc[i]
                break
        fcf_growth_rate_string = f"Free Cash Flow Growth Rate: {fcf_growth:.2%}" if isinstance(fcf_growth, (float, int)) else "Free Cash Flow Growth Rate: N/A"


        # VALUATION METRICS

        # market_cap
        market_cap = info.get('marketCap', 'N/A')
        if market_cap not in [None, 'N/A', 'nan']:
            market_cap *= exchange_rate
        market_cap_string = f"Market Cap: ${market_cap:,.2f}" if isinstance(market_cap, (float, int)) else "Market Cap: N/A"

        # enterprise_value
        enterprise_value = info.get('enterpriseValue', 'N/A')
        if enterprise_value not in [None, 'N/A', 'nan']:
            enterprise_value *= exchange_rate
        enterprise_value_string = f"Enterprise Value: ${enterprise_value:,.2f}" if isinstance(enterprise_value, (float, int)) else "Enterprise Value: N/A"

        # trailing_pe_ratio
        trailing_pe_ratio = info.get('trailingPE', 'N/A')
        trailing_pe_ratio_string = f"Trailing P/E Ratio: {trailing_pe_ratio:.2f}" if isinstance(trailing_pe_ratio, (float, int)) else "Trailing P/E Ratio: N/A"

        # forward_pe_ratio
        forward_pe_ratio = info.get('forwardPE', 'N/A')
        forward_pe_ratio_string = f"Forward P/E Ratio: {forward_pe_ratio:.2f}" if isinstance(forward_pe_ratio, (float, int)) else "Forward P/E Ratio: N/A"

        # price_to_sales
        price_to_sales = info.get('priceToSalesTrailing12Months', 'N/A')
        price_to_sales_string = f"Price to Sales: {price_to_sales:.2f}" if isinstance(price_to_sales, (float, int)) else "Price to Sales: N/A"

        # price_to_book
        price_to_book = info.get('priceToBook', 'N/A')
        price_to_book_string = f"Price to Book: {price_to_book:.2f}" if isinstance(price_to_book, (float, int)) else "Price to Book: N/A"

        # enterprise_to_ebitda
        enterprise_to_ebitda = info.get('enterpriseToEbitda', 'N/A')
        enterprise_to_ebitda_string = f"Enterprise to EBITDA: {enterprise_to_ebitda:.2f}" if isinstance(enterprise_to_ebitda, (float, int)) else "Enterprise to EBITDA: N/A"

        # enterprise_to_revenue
        enterprise_to_revenue = info.get('enterpriseToRevenue', 'N/A')
        enterprise_to_revenue_string = f"Enterprise to Revenue: {enterprise_to_revenue:.2f}" if isinstance(enterprise_to_revenue, (float, int)) else "Enterprise to Revenue: N/A"

        # trailing_eps
        trailing_eps = info.get('trailingEps', 'N/A')
        trailing_eps_string = f"Trailing EPS: {trailing_eps:.2f}" if isinstance(trailing_eps, (float, int)) else "Trailing EPS: N/A"

        # forward_eps
        forward_eps = info.get('forwardEps', 'N/A')
        forward_eps_string = f"Forward EPS: {forward_eps:.2f}" if isinstance(forward_eps, (float, int)) else "Forward EPS: N/A"


        # PRICE METRICS

        # current_price
        current_price = info.get('currentPrice', 'N/A')
        current_price_string = f"Current Price: ${current_price:.2f}" if isinstance(current_price, (float, int)) else "Current Price: N/A"

        # fifty_two_week_high
        history = stock.history(period="5y")
        week_52_high = history['High'].tail(252).max() if not history.empty else None
        fifty_two_week_high_string = f"52 Week High: ${round(week_52_high, 2)}" if isinstance(week_52_high, (float, int)) else "52 Week High: N/A"

        # fifty_two_week_low
        week_52_low = history['Low'].tail(252).min() if not history.empty else None
        fifty_two_week_low_string = f"52 Week Low: ${round(week_52_low, 2)}" if isinstance(week_52_low, (float, int)) else "52 Week Low: N/A"

        # fifty_day_average
        fifty_day_average = info.get('fiftyDayAverage', 'N/A')
        fifty_day_average_string = f"50 Day Average: ${fifty_day_average:.2f}" if isinstance(fifty_day_average, (float, int)) else "50 Day Average: N/A"

        # two_hundred_day_average
        two_hundred_day_average = info.get('twoHundredDayAverage', 'N/A')
        two_hundred_day_average_string = f"200 Day Average: ${two_hundred_day_average:.2f}" if isinstance(two_hundred_day_average, (float, int)) else "200 Day Average: N/A"


        # RETURN METRICS

        # return_on_equity
        return_on_equity = info.get('returnOnEquity', 'N/A')
        if return_on_equity not in [None, 'N/A', 'nan']:
            return_on_equity *= exchange_rate
        return_on_equity_string = f"Return on Equity: {return_on_equity:.2%}" if isinstance(return_on_equity, (float, int)) else "Return on Equity: N/A"

        # return_on_assets
        return_on_assets = info.get('returnOnAssets', 'N/A')
        if return_on_assets not in [None, 'N/A', 'nan']:
            return_on_assets *= exchange_rate
        return_on_assets_string = f"Return on Assets: {return_on_assets:.2%}" if isinstance(return_on_assets, (float, int)) else "Return on Assets: N/A"

        # return_on_invested_capital
        return_on_invested_capital = None
        try:
            invested_capital = balance_sheet.loc["Invested Capital"].iloc[0]
            operating_income_result = financials.loc["Operating Income"].iloc[0]
            tax_rate_for_calcs = financials.loc["Tax Rate For Calcs"].iloc[0]

            if (invested_capital is not None
                    and operating_income_result is not None
                    and invested_capital != 0):

                if tax_rate_for_calcs is None:
                    tax_provision = financials.loc["Tax Provision"].iloc[0]
                    pretax_income = financials.loc["Pretax Income"].iloc[0]
                    if (tax_provision is not None and pretax_income is not None
                            and pretax_income != 0):
                        estimated_tax_rate = tax_provision / pretax_income
                        nopat = operating_income_result * (1 - estimated_tax_rate)
                    else:
                        nopat = operating_income_result * (1 - 0.20)  # Default tax rate
                else:
                    nopat = operating_income_result * (1 - tax_rate_for_calcs)

                return_on_invested_capital = nopat / invested_capital
        except (KeyError, TypeError, ZeroDivisionError):
            return_on_invested_capital = 'N/A'
        return_on_invested_capital_string = f"Return on Invested Capital: {return_on_invested_capital:.2%}" if isinstance(return_on_invested_capital, (float, int)) else "Return on Invested Capital: N/A"


        # DIVIDEND METRICS

        # cash_dividends_paid
        cash_dividends_paid = cash_flow_data.get('Cash Dividends Paid', 'N/A')
        if cash_dividends_paid not in [None, 'N/A', 'nan']:
            cash_dividends_paid *= exchange_rate
        cash_dividends_paid_string = f"Cash Dividends Paid: ${cash_dividends_paid:,.2f}" if isinstance(cash_dividends_paid, (float, int)) else "Cash Dividends Paid: N/A"

        # dividend_yield
        dividend_yield = info.get('dividendYield', 'N/A')
        dividend_yield_string = f"Dividend Yield: {dividend_yield:.2%}" if isinstance(dividend_yield, (float, int)) else "Dividend Yield: N/A"

        # dividend_rate
        dividend_rate = info.get('dividendRate', 'N/A')
        dividend_rate_string = f"Dividend Rate: {dividend_rate:.2f}" if isinstance(dividend_rate, (float, int)) else "Dividend Rate: N/A"

        # payout_ratio
        payout_ratio = info.get('payoutRatio', 'N/A')
        payout_ratio_string = f"Payout Ratio: {payout_ratio:.2%}" if isinstance(payout_ratio, (float, int)) else "Payout Ratio: N/A"

        # ex_dividend_date
        ex_dividend_date = datetime.fromtimestamp(info.get('exDividendDate')).strftime('%Y-%m-%d') if info.get('exDividendDate') else 'N/A'
        ex_dividend_date_string = f"Ex-Dividend Date: {ex_dividend_date}" if isinstance(ex_dividend_date, str) else "Ex-Dividend Date: N/A"

        # next_dividend_date
        next_dividend_date = datetime.fromtimestamp(info.get('dividendDate')).strftime('%Y-%m-%d') if info.get('dividendDate') else 'N/A'
        next_dividend_date_string = f"Next Dividend Date: {next_dividend_date}" if isinstance(next_dividend_date, str) else "Next Dividend Date: N/A"


        financial_data = [
            net_income_string,
            total_revenue_string,
            gross_profit_string,
            operating_income_string,
            ebitda_string,
            basic_eps_string,
            diluted_eps_string,
            revenue_growth_string,
            earnings_growth_string,
            gross_margins_string,
            operating_margin_string,
            profit_margin_string,
            total_assets_string,
            total_liabilities_string,
            total_equity_string,
            total_debt_string,
            net_debt_string,
            cash_and_equivalents_string,
            working_capital_string,
            current_ratio_string,
            quick_ratio_string,
            debt_to_equity_string,
            retained_earnings_string,
            total_cash_string,
            total_debt_string,
            free_cash_flow_string,
            operating_cash_flow_string,
            capital_expenditure_string,
            repayment_of_debt_string,
            debt_issuance_string,
            cash_changes_string,
            investing_cash_flow_string, 
            financing_cash_flow_string,
            repurchases_of_stock_string,
            depreciation_and_amortization_string,
            revenue_growth_string,
            earnings_growth_string,
            five_year_revenue_growth_rate_string,
            two_year_revenue_growth_rate_string,
            fcf_growth_rate_string,     
            market_cap_string,
            enterprise_value_string,
            trailing_pe_ratio_string,
            forward_pe_ratio_string,
            price_to_sales_string,
            price_to_book_string,
            enterprise_to_ebitda_string,
            enterprise_to_revenue_string,
            trailing_eps_string,
            forward_eps_string,
            current_price_string,
            fifty_two_week_high_string,
            fifty_two_week_low_string,
            fifty_day_average_string,
            two_hundred_day_average_string,
            return_on_equity_string,
            return_on_assets_string,
            return_on_invested_capital_string,
            cash_dividends_paid_string,
            dividend_yield_string,
            dividend_rate_string,
            payout_ratio_string,
            ex_dividend_date_string,
            next_dividend_date_string,
        ]

        
        output_string = "\n".join(financial_data)

        return output_string
    


if __name__ == "__main__":
    financial_metrics_tool = FinancialMetricsTool()
    tool_results = financial_metrics_tool.run(ticker='ACN')
    print(tool_results)