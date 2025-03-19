import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

this_year = datetime.now().year

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


def analyse_fundamentals(ticker: str):
    """
    Analyse the fundamentals of a company.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving ticker: {str(e)}"}

    try:
        info = company_ticker.info
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving company info: {str(e)}"}

    
    score = 0
    max_score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    roe = info.get('returnOnEquity')
    if roe is not None and not np.isnan(roe):
        if roe > 0.15:  # 15% ROE threshold
            score += 2
            reasoning.append(f"Strong ROE of {roe:.1%}")
        else:
            reasoning.append(f"Weak ROE of {roe:.1%}")
    else:
        reasoning.append("ROE data not available")

    max_score += 2 # Increment by the *highest* possible score for this section

    # Check Debt to Equity
    debt_to_equity = info.get('debtToEquity')
    if debt_to_equity is not None and not np.isnan(debt_to_equity):
        debt_to_equity /= 100
        if debt_to_equity < 0.5:
            score += 2
            reasoning.append(f"Conservative debt levels (D/E ratio: {debt_to_equity:.2f})")
        else:
            reasoning.append(f"High debt to equity ratio of {debt_to_equity:.2f}")
    else:
        reasoning.append("Debt to equity data not available")

    max_score += 2 # Increment by the *highest* possible score for this section

    # Check Operating Margin
    operating_margin = info.get('operatingMargins')
    if operating_margin is not None and not np.isnan(operating_margin):
        if operating_margin > 0.15:
            score += 2
            reasoning.append(f"Strong operating margins of {operating_margin:.1%}")
        else:
            reasoning.append(f"Weak operating margin of {operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    max_score += 2 # Increment by the *highest* possible score for this section

    # Check Current Ratio
    current_ratio = info.get('currentRatio')
    if current_ratio is not None and not np.isnan(current_ratio):
        if current_ratio > 1.5:
            score += 1
            reasoning.append(f"Good liquidity position (current ratio: {current_ratio:.2f})")
        else:
            reasoning.append(f"Weak liquidity with current ratio of {current_ratio:.2f}")
    else:
        reasoning.append("Current ratio data not available")

    max_score += 1 # Increment by the *highest* possible score for this section


    return {"score": score, "max_score": max_score, "details": "; ".join(reasoning)}


def analyse_consistency(ticker: str):
    """
    Analyse the consistency of a company's earnings over time.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Analysis results with score and details
    """
    company_ticker = get_ticker(ticker)
    
    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {"score": 0, "max_score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}

    # Get Net Income data, trying alternative row names if needed
    net_income_df = safe_get_row(financials, "Net Income", 
                                ["Net Income Common Stockholders", "Net Profit"])
    
    if net_income_df is None:
        return {"score": 0, "max_score": 0, "details": "Net Income data not available in financial statements"}
    
    earnings_values = filter_valid_values(net_income_df)
    
    n = len(earnings_values)
    
    
    score = 0
    max_score = n - 1
    reasoning = []
    
    
    # Check if earnings are consistently growing
    if n >= 4:
        try:
            consistent_earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(n - 1))

        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Error analyzing earnings growth pattern: {str(e)}"}
        
    else:
        return {"score": 0, "max_score": 0, "details": f"Insufficient historical data (have {n} periods, need at least 4)"}
    
    if consistent_earnings_growth:
        score += n
        reasoning.append(f"Consistent earnings growth over {n} periods")
    else:
        reasoning.append(f"Inconsistent earnings growth pattern over {n} periods:")

    try:
        for i in range(n - 1):
            if earnings_values[i] > earnings_values[i + 1]:
                score += 1
                reasoning.append(f"earnings grew from period {this_year - i - 2} to {this_year - i - 1}")
            else:
                reasoning.append(f"earnings did not grow from period {this_year - i - 2} to {this_year - i - 1}")

    except Exception as e:
        return {"score": 0, "max_score": 4, "details": f"Error analyzing earnings growth pattern: {str(e)}"}


    # Calculate growth rate from earliest to latest
    try:
        latest_earnings = earnings_values[0]
        earliest_earnings = None

        # Find a valid (positive) earliest earnings value
        for i in range(n - 1, -1, -1):
            if earnings_values[i] is not None and not np.isnan(earnings_values[i]):
                earliest_earnings = earnings_values[i]
                break
        
        if earliest_earnings is not None and earliest_earnings != 0:
            growth_rate = (latest_earnings - earliest_earnings) / abs(earliest_earnings)
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {n} periods")
        else:
            reasoning.append("Cannot calculate growth rate (earliest earnings negative or zero)")
    except (ZeroDivisionError, IndexError, Exception) as e:
        reasoning.append(f"Error calculating growth rate: {str(e)}")

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(reasoning),
    }


def analyse_moat(ticker: str) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.
    """
    company_ticker = get_ticker(ticker)
    try:
       financials = company_ticker.financials
       if financials is None or financials.empty:
           return {"score": 0, "max_score": 0, "details": "No financial data available"}
    except Exception as e:
       return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}
    
    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
           return {"score": 0, "max_score": 0, "details": "No balance sheet data available"}
    except Exception as e:
       return {"score": 0, "max_score": 0, "details": f"Error retrieving balance sheet data: {str(e)}"}
    if financials is None or balance_sheet is None or financials.empty or balance_sheet.empty or len(financials) < 3 or len(balance_sheet) < 3:
        return {"score": 0, "max_score": 0, "details": "Insufficient data for moat analysis"}

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []


    net_income = safe_get_row(financials, "Net Income")
    shareholders_equity = safe_get_row(balance_sheet, "Stockholders Equity")

    if len(net_income) >= 3 and len(shareholders_equity) >= 3:
        for i in range(len(net_income)):
            if net_income.iloc[i] is not None and not np.isnan(net_income.iloc[i]) and shareholders_equity.iloc[i] is not None and not np.isnan(shareholders_equity.iloc[i]):
                historical_roes.append(net_income.iloc[i] / shareholders_equity.iloc[i])

    
    revenue = safe_get_row(financials, "Total Revenue")
    operating_expenses = safe_get_row(financials, "Operating Expense")

    if len(revenue) >= 3 and len(operating_expenses) >= 3:
        for i in range(len(revenue)):
            if revenue.iloc[i] is not None and not np.isnan(revenue.iloc[i]) and operating_expenses.iloc[i] is not None and not np.isnan(operating_expenses.iloc[i]):
                historical_margins.append((revenue.iloc[i] - operating_expenses.iloc[i]) / revenue.iloc[i])

    # Check for stable or improving ROE
    if len(historical_roes) >= 3:
        stable_roe = all(r > 0.15 for r in historical_roes)
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasoning.append("ROE not consistently above 15%")

    # Check for stable or improving operating margin
    if len(historical_margins) >= 3:
        stable_margin = all(m > 0.15 for m in historical_margins)
        if stable_margin:
            moat_score += 1
            reasoning.append("Stable operating margins above 15% (moat indicator)")
        else:
            reasoning.append("Operating margin not consistently above 15%")

    # If both are stable/improving, add an extra point
    if moat_score == 2:
        moat_score += 1
        reasoning.append("Both ROE and margin stability indicate a solid moat")

    return {
        "score": moat_score,
        "max_score": 3,
        "details": "; ".join(reasoning),
    }
 
 

def analyse_management_quality(ticker: str) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    company_ticker = get_ticker(ticker)

    try:
        cash_flow = company_ticker.cash_flow
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}

    if cash_flow is None or cash_flow.empty:
        return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
    
    reasoning = []
    mgmt_score = 0

    # Check for issuance or purchase of equity shares
    net_common_stock_issuance = safe_get_row(cash_flow, "Net Common Stock Issuance")

    if net_common_stock_issuance.empty or net_common_stock_issuance is None:
        return {"score": 0, "max_score": 0, "details": "Net common stock issuance data not available"}
    
    if net_common_stock_issuance.iloc[0] < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
    elif net_common_stock_issuance.iloc[0] > 0:
        mgmt_score -= 1
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    

    # Check for any dividends
    cash_dividends_paid = safe_get_row(cash_flow, "Cash Dividends Paid")

    if not cash_dividends_paid.empty and cash_dividends_paid is not None and abs(cash_dividends_paid.iloc[0]) > 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")
    
    
    
    
    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }

    

def calculate_intrinsic_value(ticker, growth_rate, discount_rate, terminal_multiple, projection_years):
    """
    Calculate intrinsic value using DCF model with owner earnings.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Intrinsic value analysis with value and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving ticker: {str(e)}"]}
    
    try:
        info = company_ticker.info
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving company info: {str(e)}"]}
    
    try:
        financials = company_ticker.financials
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving financial data: {str(e)}"]}
    
    try:
        cash_flow = company_ticker.cash_flow
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving cash flow data: {str(e)}"]}
    
    try:
        balance_sheet = company_ticker.balance_sheet
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving balance sheet data: {str(e)}"]}
    
    # Calculate owner earnings
    net_income = safe_get_row(financials, "Net Income")
    depreciation = safe_get_row(cash_flow, "Depreciation And Amortization")
    capex = safe_get_row(cash_flow, "Capital Expenditure")

    latest_net_income = net_income.iloc[0]
    latest_depreciation = depreciation.iloc[0]
    latest_capex = capex.iloc[0]

    if latest_net_income is None or np.isnan(latest_net_income):
        return {"value": None, "details": ["Net income data not available"]}
    if latest_depreciation is None or np.isnan(latest_depreciation):
        return {"value": None, "details": ["Depreciation data not available"]}
    if latest_capex is None or np.isnan(latest_capex):
        return {"value": None, "details": ["Capital expenditure data not available"]}

    try:
        # Buffett's owner earnings formula: Net Income + Depreciation - Maintenance CapEx
        # We estimate maintenance CapEx as a percentage of total CapEx
        maintenance_capex = latest_capex * 0.75  # Assuming 75% of CapEx is for maintenance
        owner_earnings = latest_net_income + latest_depreciation - maintenance_capex
        
        if owner_earnings <= 0:
            return {"value": None, "details": [f"Calculated owner earnings are negative or zero: ${owner_earnings:,.2f}"]}
    except Exception as e:
        return {"value": None, "details": [f"Error calculating owner earnings: {str(e)}"]}

    # Get shares Outstanding
    shares_outstanding = info.get("sharesOutstanding", None)
    if shares_outstanding is None or np.isnan(shares_outstanding) or shares_outstanding <= 0:
        return {"value": None, "details": ["Shares outstanding data not available or invalid"]}

    try:
        # Calculate future value using current earnings
        future_value = 0
        for year in range(1, projection_years + 1):
            future_earnings = owner_earnings * (1 + growth_rate) ** year
            present_value = future_earnings / (1 + discount_rate) ** year
            future_value += present_value

        # Calculate terminal value
        terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) \
                        / (1 + discount_rate) ** projection_years

        intrinsic_value = future_value + terminal_value
        intrinsic_value_per_share = intrinsic_value / shares_outstanding

        market_cap = info.get("marketCap")




    except Exception as e:
        return {"value": None, "details": [f"Error in DCF calculation: {str(e)}"]}

    return {
        "market_cap": f"{market_cap:,}",
        "intrinsic_value": f"${intrinsic_value:,.2f}",
        "intrinsic_value_per_share": f"${intrinsic_value_per_share:,.2f}",
        "owner_earnings": f"${owner_earnings:,.2f}",
        "owner_earnings_per_share": f"${owner_earnings/shares_outstanding:,.2f}",
        "assumptions": {
            "growth_rate": f"{growth_rate:.1%}",
            "discount_rate": f"{discount_rate:.1%}",
            "terminal_multiple": f"{terminal_multiple}x",
            "projection_years": projection_years,
        },
        "details": [
            "Intrinsic value calculated using DCF model with owner earnings",
            f"Owner earnings: ${owner_earnings:,.2f}",
            f"Growth rate: {growth_rate:.1%}",
            f"Discount rate: {discount_rate:.1%}",
            f"Terminal multiple: {terminal_multiple}x",
            f"Projection years: {projection_years}"
        ],
    }


def calculate_buffett_analysis_data(ticker: str, growth_rate: float = 0.05, 
                                   discount_rate: float = 0.09, 
                                   terminal_multiple: float = 12, 
                                   projection_years: int = 10):
    """
    Analyzes stocks using Buffett's principles and returns a dictionary of the analysis data.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Complete Buffett analysis with signal, score, and detailed components
    """

    try:
        fundamental_analysis = analyse_fundamentals(ticker)
    except Exception as e:
        fundamental_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in fundamental analysis: {str(e)}"
        }

    try:
        consistency_analysis = analyse_consistency(ticker)
    except Exception as e:
        consistency_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in consistency analysis: {str(e)}"
        }

    try:
        moat_analysis = analyse_moat(ticker)
    except Exception as e:
        moat_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in moat analysis: {str(e)}"
        }

    try:
        management_quality = analyse_management_quality(ticker)
    except Exception as e:
        management_quality = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in management quality analysis: {str(e)}"
        }

    try:
        intrinsic_value_analysis = calculate_intrinsic_value(growth_rate, discount_rate, 
                                                           terminal_multiple, projection_years)
    except Exception as e:
        intrinsic_value_analysis = {
            "value": None, 
            "details": [f"Error in intrinsic value calculation: {str(e)}"]
        }

    fundamental_analysis_score = fundamental_analysis.get("score", 0)
    fundamental_analysis_max_score = fundamental_analysis.get("max_score", 0)

    consistency_analysis_score = consistency_analysis.get("score", 0)
    consistency_analysis_max_score = consistency_analysis.get("max_score", 0)

    moat_analysis_score = moat_analysis.get("score", 0)
    moat_analysis_max_score = moat_analysis.get("max_score", 0)

    management_quality_score = management_quality.get("score", 0)
    management_quality_max_score = management_quality.get("max_score", 0)   

    total_score = fundamental_analysis_score + consistency_analysis_score + moat_analysis_score + management_quality_score
    max_possible_score = fundamental_analysis_max_score + consistency_analysis_max_score + moat_analysis_max_score + management_quality_max_score

    margin_of_safety = None

    intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")

    market_cap = intrinsic_value_analysis.get("market_Cap")

    if intrinsic_value and market_cap and not np.isnan(market_cap) and market_cap > 0:
        try:
            # Extract numeric value from formatted string
            intrinsic_value_numeric = float(intrinsic_value.replace("$", "").replace(",", ""))
            margin_of_safety = (intrinsic_value_numeric - market_cap) / intrinsic_value_numeric

            # Add to score if there's a good margin of safety (>30%)
            if margin_of_safety > 0.3:
                total_score += 2
                max_possible_score += 2 # Increment max_possible_score
        except (ValueError, ZeroDivisionError, Exception) as e:
            if "details" in intrinsic_value_analysis:
                if isinstance(intrinsic_value_analysis["details"], list):
                    intrinsic_value_analysis["details"].append(f"Error calculating margin of safety: {str(e)}")
                else:
                    intrinsic_value_analysis["details"] = f"{intrinsic_value_analysis.get('details', '')}; Error calculating margin of safety: {str(e)}"

    # Generate trading signal
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    # Combine all analysis results
    buffett_analysis_data = {
        "ticker": ticker,
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "fundamental_analysis": fundamental_analysis,
        "consistency_analysis": consistency_analysis,
        "moat_analysis": moat_analysis,
        "management_quality": management_quality,
        "intrinsic_value_analysis": intrinsic_value_analysis,
        "market_cap": market_cap,
        "margin_of_safety": f"{margin_of_safety:.2%}" if margin_of_safety else None,
    }

    return buffett_analysis_data


if __name__ == "__main__":
    buffett_analysis_data = calculate_buffett_analysis_data("NVDA", 0.05, 0.09, 12, 10)
    print(buffett_analysis_data)








