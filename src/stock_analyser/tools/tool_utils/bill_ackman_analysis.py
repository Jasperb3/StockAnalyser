import yfinance as yf
import numpy as np
import pandas as pd


def get_ticker(ticker: str):
    """
    Get a yfinance Ticker object for the given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        yf.Ticker: A yfinance Ticker object
        
    Raises:
        Exception: If ticker data cannot be retrieved
    """
    try:
        return yf.Ticker(ticker.upper())
    except Exception as e:
        raise Exception(f"Failed to get ticker data for {ticker}: {str(e)}")


def safe_get_row(df: pd.DataFrame, row_name: str, alternative_names=None):
    """
    Safely get a row from a DataFrame, handling KeyError and empty DataFrames.
    
    Args:
        df (pd.DataFrame): The DataFrame to get the row from
        row_name (str): The name of the row to get
        alternative_names (list, optional): Alternative names to try if row_name is not found
        
    Returns:
        pd.Series or None: The row data or None if not found
    """
    if df is None or df.empty:
        return None
    
    try:
        return df.loc[row_name]
    except KeyError:
        if alternative_names:
            for alt_name in alternative_names:
                try:
                    return df.loc[alt_name]
                except KeyError:
                    continue
        return None


def filter_valid_values(series):
    """
    Filter a series to only include valid numeric values (not None or NaN).
    
    Args:
        series: A pandas Series or list-like object
        
    Returns:
        list: A list of valid numeric values
    """
    if series is None:
        return []
    
    if isinstance(series, pd.Series):
        return [val for val in series if val is not None and not np.isnan(val)]
    else:
        return [val for val in series if val is not None and not np.isnan(val)]


def analyse_business_quality(ticker: str):
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages, and potential for long-term growth.
    
    Args:
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        financials (pd.DataFrame, optional): Financial data to reuse
        cash_flow (pd.DataFrame, optional): Cash flow data to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []

    
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {"score": 0, "max_score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}
            
    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}

    # 1. Multi-period revenue growth analysis
    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    
    if revenue_df is None:
        details.append("Revenue data not available in financial statements.")
    else:
        revenues = filter_valid_values(revenue_df)
        
        if len(revenues) >= 2:
            try:
                # Check if overall revenue grew from first to last
                # Note: yfinance data is typically in reverse chronological order (newest first)
                latest = revenues[0]
                earliest = None  # Initialize to None

                # Loop backwards to find a valid (non-zero) earliest revenue
                for i in range(len(revenues) - 1, -1, -1):
                    if revenues[i] != 0:
                        earliest = revenues[i]
                        break

                if earliest is not None:
                    # Simple growth rate
                    growth_rate = (latest - earliest) / abs(earliest)
                    if growth_rate > 0.5:  # e.g., 50% growth over the available time
                        score += 2
                        details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period.")
                    else:
                        score += 1
                        details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
                elif len(revenues) > 1: # We did NOT find a valid earliest_revenue, but there are at least two values
                    details.append(f"Revenue did not grow from earliest to latest.")
                else: # We did not find a valid earliest_revenue AND there is only one value
                    details.append("Cannot calculate revenue growth (zero or negative base value)")
            except (IndexError, ZeroDivisionError, Exception) as e:
                details.append(f"Error calculating revenue growth rate: {str(e)}")
        else:
            details.append("Not enough revenue data for multi-period trend (need at least 2 periods).")
    
    max_score += 2 # Increment by the *highest* possible score for this section

    # 2. Operating margin and free cash flow consistency
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    
    if fcf_df is None:
        details.append("Free cash flow data not available in cash flow statement.")
    
    fcf_vals = filter_valid_values(fcf_df) if fcf_df is not None else []

    total_revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    operating_income_df = safe_get_row(financials, "Operating Income", ["Operating Profit"])

    if total_revenue_df is None or operating_income_df is None:
        details.append("Revenue or operating income data missing for margin analysis.")
    else:
        revenues = filter_valid_values(total_revenue_df)
        op_income = filter_valid_values(operating_income_df)
        
        op_margin_vals = []
        
        if revenues and op_income and len(revenues) == len(op_income):
            for i in range(len(revenues)):
                if revenues[i] > 0:  # Avoid division by zero
                    op_margin_vals.append(op_income[i] / revenues[i])
        
        if op_margin_vals:
            # Check if the majority of operating margins are > 15%
            above_15 = sum(1 for m in op_margin_vals if m > 0.15)
            if above_15 >= (len(op_margin_vals) // 2 + 1):
                score += 2
                details.append(f"Operating margins have exceeded 15% in {above_15} of {len(op_margin_vals)} periods.")
            else:
                details.append(f"Operating margin exceeded 15% in only {above_15} of {len(op_margin_vals)} periods.")
        else:
            details.append("Could not calculate operating margins (insufficient data).")
    
    max_score += 2 # Increment by the *highest* possible score for this section

    if fcf_vals:
        # Check if free cash flow is positive in most periods
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append(f"Positive free cash flow in {positive_fcf_count} of {len(fcf_vals)} periods.")
        else:
            details.append(f"Free cash flow positive in only {positive_fcf_count} of {len(fcf_vals)} periods.")
    else:
        details.append("No free cash flow data available for analysis.")

    max_score += 1 # Increment by the *highest* possible score for this section
    
    # 3. Return on Equity (ROE) check from the latest metrics
    return_on_equity = company_ticker.info.get('returnOnEquity', None)
    if return_on_equity is not None and not np.isnan(return_on_equity):
        if return_on_equity > 0.15:
            score += 2
            details.append(f"High ROE of {return_on_equity:.1%}, indicating potential moat.")
        else:
            details.append(f"ROE of {return_on_equity:.1%} is below 15% threshold for strong moat.")
    else:
        details.append("ROE data not available in company information.")
    
    max_score += 2 # Increment by the *highest* possible score for this section

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details)
    }


def analyse_financial_discipline(ticker: str):
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    
    Args:
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        financials (pd.DataFrame, optional): Financial data to reuse
        balance_sheet (pd.DataFrame, optional): Balance sheet data to reuse
        cash_flow (pd.DataFrame, optional): Cash flow data to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []

    
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}


    try:
        financials = company_ticker.financials
    except Exception as e:
        financials = pd.DataFrame()
        details.append(f"Error retrieving financial data: {str(e)}")
            

    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return {"score": 0, "max_score": 0, "details": "No balance sheet data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving balance sheet data: {str(e)}"}
            

    try:
        cash_flow = company_ticker.cashflow
    except Exception as e:
        cash_flow = pd.DataFrame()
        details.append(f"Error retrieving cash flow data: {str(e)}")

    # 1. Multi-period debt ratio or debt_to_equity
    # Check if the company's leverage is stable or improving
    total_liabilities_df = safe_get_row(balance_sheet, "Total Liabilities Net Minority Interest", 
                                       ["Total Liabilities", "Total Debt"])
    shareholder_equity_df = safe_get_row(balance_sheet, "Stockholders Equity", 
                                        ["Total Equity", "Shareholders Equity"])

    if total_liabilities_df is None or shareholder_equity_df is None:
        details.append("Liabilities or equity data missing for debt ratio analysis.")
    else:
        total_liabilities = filter_valid_values(total_liabilities_df)
        shareholder_equity = filter_valid_values(shareholder_equity_df)

        debt_to_equity_vals = []
        if total_liabilities and shareholder_equity and len(total_liabilities) == len(shareholder_equity):
            for i in range(len(total_liabilities)):
                if shareholder_equity[i] > 0:  # Avoid division by zero
                    debt_to_equity_vals.append(total_liabilities[i] / shareholder_equity[i])
        
        # If we have multi-year data, see if D/E ratio has gone down or stayed <1 across most periods
        if debt_to_equity_vals:
            below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
            if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
                score += 2
                details.append(f"Debt-to-equity < 1.0 for {below_one_count} of {len(debt_to_equity_vals)} periods.")
            else:
                details.append(f"Debt-to-equity >= 1.0 in {len(debt_to_equity_vals) - below_one_count} of {len(debt_to_equity_vals)} periods.")
        else:
            # Fallback to total_liabilities/total_assets if D/E not available
            total_assets_df = safe_get_row(balance_sheet, "Total Assets")
            
            if total_assets_df is None:
                details.append("Total assets data not available for leverage analysis.")
            else:
                total_assets = filter_valid_values(total_assets_df)
                
                liab_to_assets = []
                if total_liabilities and total_assets and len(total_liabilities) == len(total_assets):
                    for i in range(len(total_liabilities)):
                        if total_assets[i] > 0:  # Avoid division by zero
                            liab_to_assets.append(total_liabilities[i] / total_assets[i])
                
                if liab_to_assets:
                    below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
                    if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                        score += 2
                        details.append(f"Liabilities-to-assets < 50% for {below_50pct_count} of {len(liab_to_assets)} periods.")
                    else:
                        details.append(f"Liabilities-to-assets >= 50% in {len(liab_to_assets) - below_50pct_count} of {len(liab_to_assets)} periods.")
                else:
                    details.append("Could not calculate leverage ratios (insufficient data).")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    # 2. Capital allocation approach (dividends + share counts)
    # If the company paid dividends or reduced share count over time, it may reflect discipline
    dividends_df = safe_get_row(cash_flow, "Cash Dividends Paid", ["Dividends Paid", "Common Stock Dividend"])
    
    if dividends_df is not None:
        dividends_list = filter_valid_values(dividends_df)
        if dividends_list:
            # Check if dividends were paid (i.e., negative outflows to shareholders) in most periods
            paying_dividends_count = sum(1 for d in dividends_list if d < 0)
            if paying_dividends_count >= (len(dividends_list) // 2 + 1):
                score += 1
                details.append(f"Company paid dividends in {paying_dividends_count} of {len(dividends_list)} periods.")
            else:
                details.append(f"Dividends paid in only {paying_dividends_count} of {len(dividends_list)} periods.")
        else:
            details.append("No valid dividend data found across periods.")
    else:
        details.append("Dividend data not available in cash flow statement.")

    max_score += 1 # Increment by the *highest* possible score for this section
    
    # Check for decreasing share count (simple approach):
    # We can compare first vs last if we have at least two data points
    shares_df = safe_get_row(balance_sheet, "Share Issued", 
                            ["Common Stock Shares Outstanding", "Ordinary Shares Number"])
    
    if shares_df is None:
        details.append("Outstanding shares data not available in balance sheet.")
    else:
        shares = filter_valid_values(shares_df)
        if len(shares) >= 2:
            # Note: yfinance data is typically in reverse chronological order (newest first)
            latest = shares[0]
            earliest = None

            for i in range(len(shares) - 1, -1, -1):
                if shares[i] != 0:
                    earliest = shares[i]
                    break

            if earliest is not None:
                if latest < earliest:
                    score += 1
                    details.append(f"Outstanding shares decreased from {earliest:,.0f} to {latest:,.0f} (possible buybacks).")
                else:
                    details.append(f"Outstanding shares increased from {earliest:,.0f} to {latest:,.0f}.")
            elif len(shares) > 1:
                details.append("Cannot calculate share change due to zero or invalid values.")
            else:
                details.append("Insufficient share count data to assess buybacks (need at least 2 periods).")

        else:
            details.append("Insufficient share count data to assess buybacks (need at least 2 periods).")
    
    max_score += 1 # Increment by the *highest* possible score for this section

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details)
    }

    
def analyse_valuation(ticker: str, growth_rate: float = 0.06, discount_rate: float = 0.10, 
                     terminal_multiple: float = 15, projection_years: int = 5):
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    We can do a simplified DCF or an FCF-based approach.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Analysis results with score, details and valuation metrics
    """
    
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Failed to get ticker data: {str(e)}",
            "intrinsic_value": None
        }
    
    
    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No cash flow data available",
                "intrinsic_value": None
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving cash flow data: {str(e)}",
            "intrinsic_value": None
        }
    
    market_cap = company_ticker.info.get('marketCap', None)
    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        return {
            "score": 0,
            "max_score": 0,
            "details": "Market cap data not available or invalid",
            "intrinsic_value": None
        }

    # Get the most recent FCF value
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    if fcf_df is None:
        return {
            "score": 0,
            "max_score": 0,
            "details": "Free cash flow data not available in cash flow statement",
            "intrinsic_value": None
        }
    
    fcf_vals = filter_valid_values(fcf_df)
    if not fcf_vals:
        return {
            "score": 0,
            "max_score": 0,
            "details": "No valid free cash flow values available",
            "intrinsic_value": None
        }
    
    # Use the most recent FCF value (typically first in the list from yfinance)
    fcf = fcf_vals[0]
    
    if fcf <= 0:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Most recent FCF is negative or zero (${fcf:,.0f}), cannot perform valuation",
            "intrinsic_value": None
        }
    
    try:
        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = future_fcf / (1 + discount_rate) ** year
            present_value += pv
        
        # Terminal Value
        terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                        / ((1 + discount_rate) ** projection_years)
        intrinsic_value = present_value + terminal_value
        
        # Compare with market cap => margin of safety
        margin_of_safety = (intrinsic_value - market_cap) / intrinsic_value
        
        score = 0
        if margin_of_safety > 0.3:
            score += 3
            safety_description = "Significant margin of safety (>30%)"
        elif margin_of_safety > 0.1:
            score += 1
            safety_description = "Modest margin of safety (10-30%)"
        else:
            safety_description = "Limited or no margin of safety (<10%)"
        
        details = [
            f"Calculated intrinsic value: ~${intrinsic_value:,.2f}",
            f"Market cap: ~${market_cap:,.2f}",
            f"Margin of safety: {margin_of_safety:.2%}",
            safety_description
        ]

        max_score = 3 # Increment by the *highest* possible score for this section
        
        return {
            "score": score,
            "max_score": max_score,
            "details": "; ".join(details),
            "intrinsic_value": f"${intrinsic_value:,.2f}",
            "margin_of_safety": f"{margin_of_safety:.2%}"
        }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error in DCF calculation: {str(e)}",
            "intrinsic_value": None
        }


def calculate_bill_ackman_analysis_data(ticker: str, growth_rate: float = 0.06, 
                                       discount_rate: float = 0.10, 
                                       terminal_multiple: float = 15, 
                                       projection_years: int = 5):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data so we can analyze long-term trends.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Complete Ackman analysis with signal, score, and detailed components
    """

    try:
        quality_analysis = analyse_business_quality(ticker)
    except Exception as e:
        quality_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in business quality analysis: {str(e)}"
        }
    
    try:
        balance_sheet_analysis = analyse_financial_discipline(ticker)
    except Exception as e:
        balance_sheet_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in financial discipline analysis: {str(e)}"
        }
    
    try:
        valuation_analysis = analyse_valuation(ticker, growth_rate, discount_rate, terminal_multiple, 
                                              projection_years)
    except Exception as e:
        valuation_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in valuation analysis: {str(e)}",
            "intrinsic_value": None
        }

    # Combine partial scores or signals
    total_score = quality_analysis.get("score", 0) + balance_sheet_analysis.get("score", 0) + valuation_analysis.get("score", 0)
    max_possible_score = quality_analysis.get("max_score", 0) + balance_sheet_analysis.get("max_score", 0) + valuation_analysis.get("max_score", 0)
    
    # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"
    
    analysis_data = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "quality_analysis": quality_analysis,
        "balance_sheet_analysis": balance_sheet_analysis,
        "valuation_analysis": valuation_analysis
    }

    return analysis_data


if __name__ == "__main__":
    print(calculate_bill_ackman_analysis_data("MSFT", 0.06, 0.10, 15, 5))