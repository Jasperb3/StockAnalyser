import yfinance as yf
import pandas as pd
import numpy as np
import math

def calculate_tangible_book_value_per_share(ticker):
    try:
        balance_sheet = yf.Ticker(ticker).balance_sheet
        info = yf.Ticker(ticker).info

        stockholders_equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
        print(f"stockholders_equity: {stockholders_equity:,.2f}")
        goodwill = balance_sheet.loc['Goodwill'].iloc[0]
        other_intangible_assets = balance_sheet.loc['Other Intangible Assets'].iloc[0]
        shares_outstanding = info.get('sharesOutstanding')
        print(f"shares_outstanding: {shares_outstanding:,.2f}")

        if shares_outstanding is None:
            return "Shares Outstanding data not found"

        total_intangible_assets = goodwill + other_intangible_assets
        tangible_book_value = stockholders_equity - total_intangible_assets
        tangible_book_value_per_share = tangible_book_value / shares_outstanding

        return tangible_book_value_per_share

    except KeyError as e:
        return f"KeyError: {e}. Data might be missing for {ticker}"
    except Exception as e:
        return f"An error occurred: {e}"


def calculate_eps_without_nri(ticker):
    try:
        income_statement = yf.Ticker(ticker).income_stmt
        info = yf.Ticker(ticker).info

        net_income = income_statement.loc['Net Income'].iloc[0]
        print(f"net_income: {net_income:,.2f}")
        # Preferred dividends, often not found, so we will set to zero.
        preferred_dividends = income_statement.loc['Preferred Dividends'].iloc[0] if 'Preferred Dividends' in income_statement.index else 0
        print(f"preferred_dividends: {preferred_dividends:,.2f}")
        # Non-recurring items, very hard to find, so we will set to zero.
        non_recurring_items = 0
        print(f"non_recurring_items: {non_recurring_items:,.2f}")
        # Weighted average shares outstanding, yfinance might not have this, so we will use shares outstanding.
        shares_outstanding = info.get('sharesOutstanding')
        print(f"shares_outstanding: {shares_outstanding:,.2f}")
        if shares_outstanding is None:
            return "Shares Outstanding data not found"

        eps_without_nri = (net_income - preferred_dividends - non_recurring_items) / shares_outstanding
        print(f"eps_without_nri: {eps_without_nri:,.2f}")
        return eps_without_nri

    except KeyError as e:
        return f"KeyError: {e}. Data might be missing for {ticker}"
    except Exception as e:
        return f"An error occurred: {e}"



def compute_growth(df: pd.DataFrame, row_name: str, start_date: str, end_date: str):
    """
    Calculates the percentage growth for a given row_name in a yfinance DataFrame
    between start_date and end_date columns.
    
    Args:
        df (pd.DataFrame): DataFrame (e.g., financials or balance_sheet) where rows are line items
                           and columns are dates.
        row_name (str)   : Name of the row (e.g., "Operating Income", "Basic EPS").
        start_date (str) : Column label for the start period (e.g., "2021-09-30").
        end_date (str)   : Column label for the end period (e.g., "2022-09-30").
    
    Returns:
        float or None: The calculated growth rate (as a fraction). None if data is unavailable.
    """
    try:
        start_val = df.loc[row_name, start_date]
        end_val = df.loc[row_name, end_date]
        if pd.isna(start_val) or pd.isna(end_val) or start_val == 0 or np.isnan(start_val) or np.isnan(end_val):
            return None
        
        growth = (end_val - start_val) / abs(start_val)
        return growth
    except KeyError:
        return None  # If row_name or date columns don't exist
    except Exception:
        return None

def compute_financial_metrics(ticker_symbol: str):
    """
    Compute a suite of financial metrics for the given ticker symbol using yfinance data fields.
    If a metric cannot be computed with the provided data fields, it will be set to None
    along with a note explaining the limitation.
    
    Returns:
        dict: A dictionary mapping each metric name to its computed value (or None with a note).
    """
    
    # Fetch data from yfinance
    ticker = yf.Ticker(ticker_symbol)
    
    # Cache .info, .balance_sheet, .financials, .cashflow for repeated use
    info_data = ticker.info
    try:
        bs_data = ticker.balance_sheet  # balance_sheet is a DataFrame with columns for each reported period
    except Exception:
        bs_data = pd.DataFrame()
    try:
        fin_data = ticker.financials    # financials is a DataFrame with rows for line items and columns for periods
    except Exception:
        fin_data = pd.DataFrame()
    try:
        cf_data = ticker.cashflow       # cashflow is a DataFrame with rows for line items and columns for periods
    except Exception:
        cf_data = pd.DataFrame()

    try:
        quarterly_balancesheet = ticker.quarterly_balancesheet
    except Exception:
        quarterly_balancesheet = pd.DataFrame()

    # Helper function to find the most recent common date between DataFrames
    def find_most_recent_common_date(*dataframes):
        """
        Find the most recent common date (column) across multiple DataFrames.
        
        Args:
            *dataframes: Variable number of pandas DataFrames to compare
        
        Returns:
            str or None: The most recent common date, or None if no common dates exist
        """
        # Filter out empty DataFrames
        valid_dfs = [df for df in dataframes if not df.empty]
        if not valid_dfs:
            return None
            
        # Find common dates across all DataFrames
        common_dates = valid_dfs[0].columns
        for df in valid_dfs[1:]:
            common_dates = common_dates.intersection(df.columns)
            
        if common_dates.empty:
            return None
            
        # Return the most recent date (assuming columns are sorted chronologically)
        return common_dates[0]
    
    # Helper function to get a value from a specific date and row
    def get_value_for_date(df, row_name, date):
        """Get a value from a DataFrame for a specific row and date"""
        try:
            return df.loc[row_name, date]
        except Exception:
            return None
    
    # Get dates for growth calculations. Assumes columns are sorted chronologically.
    try:
        dates = bs_data.columns.to_list()
        end_period = dates[0]  # Most recent
        start_period = dates[1] if len(dates) > 1 else None # Second most recent
    except:
        start_period = None
        end_period = None
    
    # Helper function to safely get a single data point from a DataFrame row (most recent column)
    # Returns None if row or column not found.
    def get_most_recent(df, row_name):
        try:
            return df.loc[row_name].iloc[0]
        except Exception:
            return None
        
    def get_next_most_recent(df, row_name):
        try:
            return df.loc[row_name].iloc[1]
        except Exception:
            return None
    
    # Prepare a results dictionary
    metrics = {}
    
    # 1. Market Cap
    #   Direct from info: 'marketCap'
    metrics["market_cap"] = info_data.get("marketCap", None)

    # 2. Enterprise Value
    #   Direct from info: 'enterpriseValue'
    metrics["enterprise_value"] = info_data.get("enterpriseValue", None)

    # 3. Price to Earnings Ratio (Trailing P/E)
    #   Direct from info: 'trailingPE'
    metrics["price_to_earnings_ratio"] = info_data.get("trailingPE", None)

    # 4. Price to Book Ratio
    #   Direct from info: 'priceToBook'
    metrics["price_to_book_ratio"] = info_data.get("priceToBook", None)

    # 5. Price to Sales Ratio
    #   Direct from info: 'priceToSalesTrailing12Months'
    metrics["price_to_sales_ratio"] = info_data.get("priceToSalesTrailing12Months", None)

    # 6. Enterprise Value to EBITDA Ratio
    #   If both EV and EBITDA are available, compute EV / EBITDA
    ev = info_data.get("enterpriseValue", None)
    ebitda_val = info_data.get("ebitda", None)
    if ev is not None and ebitda_val is not None and ebitda_val != 0:
        metrics["enterprise_value_to_ebitda_ratio"] = ev / ebitda_val
    else:
        # If 'enterpriseToEbitda' is provided directly, try that; otherwise None
        metrics["enterprise_value_to_ebitda_ratio"] = info_data.get("enterpriseToEbitda", None)

    # 7. Enterprise Value to Revenue Ratio
    #   If EV and totalRevenue are available, compute EV / totalRevenue
    total_revenue = info_data.get("totalRevenue", None)
    if ev is not None and total_revenue is not None and total_revenue != 0:
        metrics["enterprise_value_to_revenue_ratio"] = ev / total_revenue
    else:
        # If 'enterpriseToRevenue' is provided directly, try that; otherwise None
        metrics["enterprise_value_to_revenue_ratio"] = info_data.get("enterpriseToRevenue", None)

    # 8. Free Cash Flow Yield = Free Cash Flow / Market Cap
    #   Yahoo sometimes provides 'freeCashflow' in .info
    # fcf_val = get_most_recent(quarterly_balancesheet, "Free Cash Flow")
    fcf_val = info_data.get("freeCashflow", None)
    mc = metrics["market_cap"]
    if fcf_val is not None and mc is not None and mc != 0:
        metrics["free_cash_flow_yield"] = fcf_val / mc
    else:
        metrics["free_cash_flow_yield"] = None  # Not possible with provided fields

    # 9. PEG Ratio
    #   Yahoo often provides 'trailingPegRatio' or 'pegRatio'. If not present, it can't be directly computed here.
    metrics["peg_ratio"] = info_data.get("trailingPegRatio", None)

    # 10. Gross Margin
    #   Direct from info: 'grossMargins'
    metrics["gross_margin"] = info_data.get("grossMargins", None)

    # 11. Operating Margin
    #   Direct from info: 'operatingMargins'
    metrics["operating_margin"] = info_data.get("operatingMargins", None)

    # 12. Net Margin
    #   Yahoo calls it 'profitMargins'
    metrics["net_margin"] = info_data.get("profitMargins", None)

    # 13. Return on Equity
    #   Direct from info: 'returnOnEquity'
    metrics["return_on_equity"] = info_data.get("returnOnEquity", None)

    # 14. Return on Assets
    #   Direct from info: 'returnOnAssets'
    metrics["return_on_assets"] = info_data.get("returnOnAssets", None)

    # 15. Return on Invested Capital
    #   Not directly available in .info. We can attempt a simple approximation:
    #   ROIC = NOPAT / InvestedCapital
    #   NOPAT ~ Operating Income * (1 - tax_rate), if we have those fields.
    
    # Find the most recent common date index
    common_dates = []
    if quarterly_balancesheet is not None and fin_data is not None:
        common_dates = quarterly_balancesheet.columns.intersection(fin_data.columns)

    if common_dates.empty:
        metrics["return_on_invested_capital"] = None
    else:
        most_recent_date = common_dates[0]  # Get the most recent common date

        invested_capital = quarterly_balancesheet.get(most_recent_date, {}).get("Invested Capital")
        operating_income = fin_data.get(most_recent_date, {}).get("Operating Income")
        tax_rate_for_calcs = fin_data.get(most_recent_date, {}).get("Tax Rate For Calcs")

        if (invested_capital is not None
                and operating_income is not None
                and invested_capital != 0):

            if tax_rate_for_calcs is not None:
                nopat = operating_income * (1 - tax_rate_for_calcs)
            else:
                tax_provision = fin_data.get(most_recent_date, {}).get("Tax Provision")
                pretax_income = fin_data.get(most_recent_date, {}).get("Pretax Income")
                if tax_provision is not None and pretax_income is not None and pretax_income != 0:
                    estimated_tax_rate = tax_provision / pretax_income
                    nopat = operating_income * (1 - estimated_tax_rate)
                else:
                    # Use a default tax rate of 20%
                    nopat = operating_income * (1 - 0.20)

            metrics["return_on_invested_capital"] = nopat / invested_capital
        else:
            metrics["return_on_invested_capital"] = None

    # 16. Asset Turnover = Total Revenue / Total Assets
    #   We can approximate by using info_data['totalRevenue'] and the most recent 'Total Assets' from balance_sheet
    total_assets = get_most_recent(quarterly_balancesheet, "Total Assets")
    if total_revenue is not None and total_assets not in (None, 0):
        metrics["asset_turnover"] = total_revenue / total_assets
    else:
        metrics["asset_turnover"] = None

    # 17. Inventory Turnover = Cost of Revenue / Inventory (approx for single period)
    # Find most recent common date between fin_data and quarterly_balancesheet
    common_date = find_most_recent_common_date(fin_data, quarterly_balancesheet)
    if common_date:
        cost_of_revenue = get_value_for_date(fin_data, "Cost Of Revenue", common_date)
        inventory = get_value_for_date(quarterly_balancesheet, "Inventory", common_date)
        if cost_of_revenue is not None and inventory not in (None, 0):
            metrics["inventory_turnover"] = cost_of_revenue / inventory
        else:
            metrics["inventory_turnover"] = None
    else:
        # Fallback to the original method if no common date
        cost_of_revenue = get_most_recent(fin_data, "Cost Of Revenue")
        inventory = get_most_recent(quarterly_balancesheet, "Inventory")
        if cost_of_revenue is not None and inventory not in (None, 0):
            metrics["inventory_turnover"] = cost_of_revenue / inventory
        else:
            metrics["inventory_turnover"] = None

    # 18. Receivables Turnover = Total Revenue / Accounts Receivable (approx)
    common_date = find_most_recent_common_date(fin_data, bs_data)
    if common_date:
        total_revenue_date = get_value_for_date(fin_data, "Total Revenue", common_date)
        accounts_receivable = get_value_for_date(quarterly_balancesheet, "Accounts Receivable", common_date)
        if total_revenue_date is not None and accounts_receivable not in (None, 0):
            metrics["receivables_turnover"] = total_revenue_date / accounts_receivable
        else:
            metrics["receivables_turnover"] = None
    else:
        # Fallback to original method
        total_revenue = info_data.get("totalRevenue", None)
        accounts_receivable = get_most_recent(bs_data, "Accounts Receivable")
        if total_revenue is not None and accounts_receivable not in (None, 0):
            metrics["receivables_turnover"] = total_revenue / accounts_receivable
        else:
            metrics["receivables_turnover"] = None

    # 19. Days Sales Outstanding = 365 / Receivables Turnover
    rt = metrics["receivables_turnover"]
    if rt not in (None, 0):
        metrics["days_sales_outstanding"] = 365 / rt
    else:
        metrics["days_sales_outstanding"] = None

    # 20. Operating Cycle = Days Inventory Outstanding + Days Sales Outstanding
    #   DIO = 365 / Inventory Turnover, DSO = 365 / Receivables Turnover
    it = metrics["inventory_turnover"]
    dso = metrics["days_sales_outstanding"]
    if it not in (None, 0) and dso is not None:
        dio = 365 / it
        metrics["operating_cycle"] = dio + dso
    else:
        metrics["operating_cycle"] = None

    # 21. Working Capital Turnover = Total Revenue / Working Capital
    #   Yahoo has "Working Capital" in balance_sheet
    working_cap = get_most_recent(quarterly_balancesheet, "Working Capital")
    if total_revenue is not None and working_cap not in (None, 0):
        metrics["working_capital_turnover"] = total_revenue / working_cap
    else:
        metrics["working_capital_turnover"] = None

    # 22. Current Ratio
    #   Direct from info: 'currentRatio'
    #   or from scratch: Current Assets / Current Liabilities
    if "currentRatio" in info_data:
        metrics["current_ratio"] = info_data["currentRatio"]
    else:
        current_assets = get_most_recent(quarterly_balancesheet, "Current Assets")
        current_liabilities = get_most_recent(quarterly_balancesheet, "Current Liabilities")
        if current_assets and current_liabilities not in (None, 0):
            metrics["current_ratio"] = current_assets / current_liabilities
        else:
            metrics["current_ratio"] = None

    # 23. Quick Ratio
    #   Direct from info: 'quickRatio'
    metrics["quick_ratio"] = info_data.get("quickRatio", None)

    # 24. Cash Ratio = (Cash and Cash Equivalents) / Current Liabilities
    cash_equiv = get_most_recent(quarterly_balancesheet, "Cash And Cash Equivalents")
    current_liab = get_most_recent(quarterly_balancesheet, "Current Liabilities")
    if cash_equiv not in (None, 0) and current_liab not in (None, 0):
        metrics["cash_ratio"] = cash_equiv / current_liab
    else:
        metrics["cash_ratio"] = None

    # 25. Operating Cash Flow Ratio = Operating Cash Flow / Current Liabilities
    ocf = get_most_recent(cf_data, "Operating Cash Flow")
    if ocf not in (None, 0) and current_liab not in (None, 0):
        metrics["operating_cash_flow_ratio"] = ocf / current_liab
    else:
        metrics["operating_cash_flow_ratio"] = None

    # 26. Debt to Equity
    #   Direct from info: 'debtToEquity'
    metrics["debt_to_equity"] = info_data.get("debtToEquity", None)

    # 27. Debt to Assets = Total Debt / Total Assets
    total_debt = info_data.get("totalDebt", None)
    if total_debt not in (None, 0) and total_assets not in (None, 0):
        metrics["debt_to_assets"] = total_debt / total_assets
    else:
        metrics["debt_to_assets"] = None

    # 28. Interest Coverage = Operating Income / Interest Expense (for the most recent period)
    op_income = get_most_recent(fin_data, "Operating Income")
    int_expense = get_most_recent(fin_data, "Interest Expense")

    if int_expense is None or np.isnan(int_expense) or op_income is None or np.isnan(op_income):
        int_expense = get_next_most_recent(fin_data, "Interest Expense")
        op_income = get_next_most_recent(fin_data, "Operating Income")

    if op_income is not None and int_expense not in (None, 0) and not np.isnan(op_income) and not np.isnan(int_expense):
        metrics["interest_coverage"] = op_income / int_expense
    else:
        metrics["interest_coverage"] = None

    # 29. Revenue Growth
    #   Direct from info: 'revenueGrowth' (usually yoy growth)
    metrics["revenue_growth"] = info_data.get("revenueGrowth", None)

    # Growth Metrics, computed using historical data if available:
    if start_period and end_period:
        # 30. Earnings Growth
        net_income_row = "Net Income"
        if net_income_row not in fin_data.index:
            net_income_row = "Net Income Common Stockholders"  # fallback
        metrics["earnings_growth"] = compute_growth(fin_data, net_income_row, start_period, end_period)

        # 31. Book Value Growth
        equity_row = "Stockholders Equity"
        if equity_row not in bs_data.index:
            equity_row = "Common Stock Equity"
        metrics["book_value_growth"] = compute_growth(bs_data, equity_row, start_period, end_period)

        # 32. Earnings Per Share Growth
        eps_row = "Basic EPS"
        if eps_row not in fin_data.index:
            eps_row = "Diluted EPS"
        metrics["earnings_per_share_growth"] = compute_growth(fin_data, eps_row, start_period, end_period)

        # 33. Free Cash Flow Growth
        fcf_row = "Free Cash Flow"
        metrics["free_cash_flow_growth"] = compute_growth(cf_data, fcf_row, start_period, end_period)

        # 34. Operating Income Growth
        op_income_row = "Operating Income"
        metrics["operating_income_growth"] = compute_growth(fin_data, op_income_row, start_period, end_period)

        # 35. EBITDA Growth
        ebitda_row = "EBITDA"
        metrics["ebitda_growth"] = compute_growth(fin_data, ebitda_row, start_period, end_period)

    else:
        metrics["earnings_growth"] = None
        metrics["book_value_growth"] = None
        metrics["earnings_per_share_growth"] = None
        metrics["free_cash_flow_growth"] = None
        metrics["operating_income_growth"] = None
        metrics["ebitda_growth"] = None

    # 36. Payout Ratio
    #   Direct from info: 'payoutRatio'
    metrics["payout_ratio"] = info_data.get("payoutRatio", None)

    # 37. Earnings Per Share (EPS)
    #   We can check .financials for the row 'Diluted EPS'. This is usually annual, not TTM.
    eps_val = get_most_recent(fin_data, "Diluted EPS")
    metrics["earnings_per_share"] = eps_val

    # # Alternatively, we might try the following (not recommended):
    # eps_without_nri = calculate_eps_without_nri(ticker_symbol)
    # metrics["earnings_per_share_without_nri"] = eps_without_nri

    # average_eps = (eps_val + eps_without_nri) / 2
    # metrics["average_earnings_per_share"] = average_eps

    # 38. Book Value Per Share
    #   From info: 'bookValue' is typically the per share figure on Yahoo

    metrics["book_value_per_share"] = quarterly_balancesheet.loc['Tangible Book Value'].iloc[0]

    # 39. Free Cash Flow Per Share
    #   If 'freeCashflow' and 'sharesOutstanding' are available in info, we can compute approximate FCF/share
    fcf_shares_out = info_data.get("sharesOutstanding", None)
    if fcf_val is not None and fcf_shares_out not in (None, 0):
        metrics["free_cash_flow_per_share"] = fcf_val / fcf_shares_out
    else:
        metrics["free_cash_flow_per_share"] = None

    # 40. Total Liabilities
    metrics["total_liabilities"] = get_most_recent(bs_data, "Total Liabilities Net Minority Interest")

    # 41. Capital Expenditure (CapEx) - using absolute value
    metrics["capex"] = abs(get_most_recent(cf_data, "Capital Expenditure")) if get_most_recent(cf_data, "Capital Expenditure") is not None else None

    # 42. Depreciation and Amortization
    metrics["depreciation_and_amortization"] = get_most_recent(cf_data, "Depreciation And Amortization")

    # 43. Net Income
    metrics["net_income"] = get_most_recent(fin_data, "Net Income")
    
    # 44. Outstanding Shares with Fallback
    metrics["outstanding_shares"] = get_most_recent(bs_data, "Share Issued")
    if metrics["outstanding_shares"] is None:
        metrics["outstanding_shares"] = get_most_recent(bs_data, "Ordinary Shares Number")

    # 45. Shares Outstanding
    metrics["shares_outstanding"] = info_data.get("sharesOutstanding", None)

    # 46. Market Cap
    metrics["market_cap"] = info_data.get("marketCap", None)

    # 47. Revenue   
    metrics["revenue"] = info_data.get("totalRevenue", None)

    # 48. Dividends and Other Cash Distributions
    metrics["dividends_and_other_cash_distributions"] = get_most_recent(cf_data, "Cash Dividends Paid")

    # 49. Current Assets
    metrics["current_assets"] = get_most_recent(bs_data, "Current Assets")

    # 50. Total Assets
    metrics["total_assets"] = get_most_recent(bs_data, "Total Assets")

    # 51. Current Liabilities
    metrics["current_liabilities"] = get_most_recent(bs_data, "Current Liabilities")

    # 52. Total Liabilities
    metrics["total_liabilities"] = get_most_recent(bs_data, "Total Liabilities Net Minority Interest")

    # 53. Tangible Book Value Per Share
    metrics["tangible_book_value_per_share"] = calculate_tangible_book_value_per_share(ticker_symbol)

    return metrics



if __name__ == "__main__":
    ticker_symbol = "SOUN"
    results = compute_financial_metrics(ticker_symbol)
    for metric_name, value in results.items():
        print(f"{metric_name}: {value}")
    #
    # print(f"tangible_book_value_per_share: {calculate_tangible_book_value_per_share(ticker_symbol)}")
    # book_value_ps = calculate_tangible_book_value_per_share(ticker_symbol)
    # print(f"eps: {metrics["earnings_per_share"]}")
    # eps = calculate_eps_without_nri(ticker_symbol)

    # graham_number = math.sqrt(22.5 * eps * book_value_ps)
    # print(f"graham_number: {graham_number:,.2f}")

    # price_per_share = yf.Ticker(ticker_symbol).info.get("regularMarketPrice", None)
    # print(f"price_per_share: {price_per_share:,.2f}")
    
    