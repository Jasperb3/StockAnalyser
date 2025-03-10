import yfinance as yf
from stock_analyser.tools.tool_utils.metrics import compute_financial_metrics
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


def analyse_fundamentals(metrics: dict):
    """
    Analyse the fundamentals of a company using the metrics dictionary.
    
    Args:
        metrics (dict): Financial metrics dictionary
        
    Returns:
        dict: Analysis results with score, details and metrics
    """
    if not metrics:
        return {"score": 0, "max_score": 0, "details": "Insufficient fundamental data", "metrics": {}}
    
    score = 0
    max_score = 0 # Initialize max_score
    reasoning = []

    # Check ROE (Return on Equity)
    roe = metrics.get('return_on_equity')
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
    debt_to_equity = metrics.get('debt_to_equity')
    if debt_to_equity is not None and not np.isnan(debt_to_equity):
        if debt_to_equity < 0.5:
            score += 2
            reasoning.append(f"Conservative debt levels (D/E ratio: {debt_to_equity:.2f})")
        else:
            reasoning.append(f"High debt to equity ratio of {debt_to_equity:.2f}")
    else:
        reasoning.append("Debt to equity data not available")

    max_score += 2 # Increment by the *highest* possible score for this section

    # Check Operating Margin
    operating_margin = metrics.get('operating_margin')
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
    current_ratio = metrics.get('current_ratio')
    if current_ratio is not None and not np.isnan(current_ratio):
        if current_ratio > 1.5:
            score += 1
            reasoning.append(f"Good liquidity position (current ratio: {current_ratio:.2f})")
        else:
            reasoning.append(f"Weak liquidity with current ratio of {current_ratio:.2f}")
    else:
        reasoning.append("Current ratio data not available")

    max_score += 1 # Increment by the *highest* possible score for this section

    # round all values in metrics to 2 decimal places
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, 2)

    return {"score": score, "max_score": max_score, "details": "; ".join(reasoning), "metrics": metrics}


def analyse_consistency(ticker: str, company_ticker=None, financials=None):
    """
    Analyse the consistency of a company's earnings over time.
    
    Args:
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        financials (pd.DataFrame, optional): Financial data to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    # Reuse financials if provided, otherwise get from ticker
    if financials is None:
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
    
    if n < 4:  # Need at least 4 periods for trend analysis
        return {"score": 0, "max_score": 0, "details": f"Insufficient historical data (have {n} periods, need at least 4)"}
    
    score = 0
    max_score = 0 # Initialize max_score
    reasoning = []
    
    # Check if earnings are consistently growing
    # Note: yfinance data is typically in reverse chronological order (newest first)
    try:
        # Check if each period's earnings are higher than the next (older) period
        earnings_growth = True
        for i in range(n - 1):
            if earnings_values[i] <= earnings_values[i+1]:
                earnings_growth = False
                break
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error analyzing earnings growth pattern: {str(e)}"}

    if earnings_growth:
        score += 3
        reasoning.append(f"Consistent earnings growth over {n} periods")
    else:
        reasoning.append(f"Inconsistent earnings growth pattern over {n} periods")

    max_score += 3 # Increment by the *highest* possible score for this section

    # Calculate growth rate from earliest to latest
    if n >= 2:
        try:
            latest = earnings_values[0]
            earliest = None

            # Find a valid (positive) earliest earnings value
            for i in range(len(earnings_values) - 1, -1, -1):
                if earnings_values[i] > 0:
                    earliest = earnings_values[i]
                    break
            
            if earliest is not None:
                growth_rate = (latest - earliest) / earliest
                reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {n} periods")
            elif len(earnings_values) > 1:
                reasoning.append("Cannot calculate growth rate (earliest earnings negative or zero)")
            else:
                reasoning.append("Cannot calculate growth rate (earliest earnings negative or zero)")
        except (ZeroDivisionError, IndexError, Exception) as e:
            reasoning.append(f"Error calculating growth rate: {str(e)}")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(reasoning),
    }
    

def calculate_intrinsic_value(metrics, growth_rate, discount_rate, terminal_multiple, projection_years):
    """
    Calculate intrinsic value using DCF model with owner earnings.
    
    Args:
        metrics (dict): Financial metrics dictionary
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Intrinsic value analysis with value and details
    """
    if not metrics:
        return {"value": None, "details": ["Insufficient data for valuation"]}
    
    # Calculate owner earnings
    net_income = metrics.get("net_income")
    depreciation = metrics.get("depreciation_and_amortization")
    capex = metrics.get("capex")

    if net_income is None or np.isnan(net_income):
        return {"value": None, "details": ["Net income data not available"]}
    if depreciation is None or np.isnan(depreciation):
        return {"value": None, "details": ["Depreciation data not available"]}
    if capex is None or np.isnan(capex):
        return {"value": None, "details": ["Capital expenditure data not available"]}

    try:
        # Buffett's owner earnings formula: Net Income + Depreciation - Maintenance CapEx
        # We estimate maintenance CapEx as a percentage of total CapEx
        maintenance_capex = capex * 0.75  # Assuming 75% of CapEx is for maintenance
        owner_earnings = net_income + depreciation - maintenance_capex
        
        if owner_earnings <= 0:
            return {"value": None, "details": [f"Calculated owner earnings are negative or zero: ${owner_earnings:,.2f}"]}
    except Exception as e:
        return {"value": None, "details": [f"Error calculating owner earnings: {str(e)}"]}

    # Get shares Outstanding
    shares_outstanding = metrics.get("shares_outstanding", None)
    if shares_outstanding is None or np.isnan(shares_outstanding) or shares_outstanding <= 0:
        return {"value": None, "details": ["Shares outstanding data not available or invalid"]}

    try:
        # Calculate future value using DCF
        future_value = 0
        for year in range(1, projection_years + 1):
            future_earnings = owner_earnings * (1 + growth_rate) ** year
            present_value = future_earnings / (1 + discount_rate) ** year
            future_value += present_value

        # Calculate terminal value
        terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / (1 + discount_rate) ** projection_years

        intrinsic_value = future_value + terminal_value
        intrinsic_value_per_share = intrinsic_value / shares_outstanding
    except Exception as e:
        return {"value": None, "details": [f"Error in DCF calculation: {str(e)}"]}

    return {
        "intrinsic_value": f"${intrinsic_value:,.2f}",
        "intrinsic_value_per_share": f"${intrinsic_value_per_share:,.2f}",
        "owner_earnings": f"${owner_earnings:,.2f}",
        "owner_earnings_per_share": f"${owner_earnings/shares_outstanding:,.2f}",
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
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
        # Get ticker data once and reuse
        company_ticker = get_ticker(ticker)
        financials = company_ticker.financials
        
        # Get metrics once
        metrics = compute_financial_metrics(ticker)
    except Exception as e:
        return {
            "ticker": ticker,
            "signal": "neutral",
            "score": 0,
            "max_score": 0, # Initialize max_score
            "error": f"Failed to compute financial metrics: {str(e)}"
        }

    try:
        fundamental_analysis = analyse_fundamentals(metrics)
    except Exception as e:
        fundamental_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in fundamental analysis: {str(e)}"
        }

    try:
        consistency_analysis = analyse_consistency(ticker, company_ticker, financials)
    except Exception as e:
        consistency_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in consistency analysis: {str(e)}"
        }

    try:
        intrinsic_value_analysis = calculate_intrinsic_value(metrics, growth_rate, discount_rate, 
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

    total_score = fundamental_analysis_score + consistency_analysis_score
    max_possible_score = fundamental_analysis_max_score + consistency_analysis_max_score

    margin_of_safety = None
    intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")
    market_cap = metrics.get("market_cap")

    if intrinsic_value and market_cap and not np.isnan(market_cap) and market_cap > 0:
        try:
            # Extract numeric value from formatted string
            intrinsic_value_numeric = float(intrinsic_value.replace("$", "").replace(",", ""))
            margin_of_safety = (intrinsic_value_numeric - market_cap) / market_cap

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
        "intrinsic_value_analysis": intrinsic_value_analysis,
        "market_cap": market_cap,
        "margin_of_safety": f"{margin_of_safety:.2%}" if margin_of_safety is not None else None,
    }

    return buffett_analysis_data


if __name__ == "__main__":
    buffett_analysis_data = calculate_buffett_analysis_data("MSFT", 0.05, 0.09, 12, 10)
    print(buffett_analysis_data)








