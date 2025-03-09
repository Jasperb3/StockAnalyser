import yfinance as yf
from stock_analyser.tools.tool_utils.metrics import compute_financial_metrics
import numpy as np
import pandas as pd


def get_ticker(ticker: str):
    try:
        return yf.Ticker(ticker.upper())
    except Exception as e:
        raise Exception(f"Failed to get ticker data for {ticker}: {str(e)}")


def analyse_fundamentals(metrics: dict):
    """
    Analyse the fundamentals of a company using the metrics dictionary.
    """

    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}
    
    score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    if metrics.get('return_on_equity') and not np.isnan(metrics['return_on_equity']) and metrics['return_on_equity'] > 0.15:  # 15% ROE threshold
        score += 2
        reasoning.append(f"Strong ROE of {metrics['return_on_equity']:.1%}")
    elif metrics.get('return_on_equity') and not np.isnan(metrics['return_on_equity']):
        reasoning.append(f"Weak ROE of {metrics['return_on_equity']:.1%}")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity
    if metrics.get('debt_to_equity') is not None and not np.isnan(metrics['debt_to_equity']) and metrics['debt_to_equity'] < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif metrics.get('debt_to_equity') is not None and not np.isnan(metrics['debt_to_equity']):
        reasoning.append(f"High debt to equity ratio of {metrics['debt_to_equity']:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin
    if metrics.get('operating_margin') and not np.isnan(metrics['operating_margin']) and metrics['operating_margin'] > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif metrics.get('operating_margin') and not np.isnan(metrics['operating_margin']):
        reasoning.append(f"Weak operating margin of {metrics['operating_margin']:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    if metrics.get('current_ratio') and not np.isnan(metrics['current_ratio']) and metrics['current_ratio'] > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif metrics.get('current_ratio') and not np.isnan(metrics['current_ratio']):
        reasoning.append(f"Weak liquidity with current ratio of {metrics['current_ratio']:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": metrics}


def analyse_consistency(ticker: str):
    """
    Analyse the consistency of a company using the metrics dictionary.
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}

    try:
        financials = company_ticker.financials
        if financials.empty:
            return {"score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving financial data: {str(e)}"}

    try:
        net_income_df = financials.loc["Net Income"]
    except (KeyError, Exception) as e:
        return {"score": 0, "details": f"Net Income data not available: {str(e)}"}

    if net_income_df.empty:
        return {"score": 0, "details": "Insufficient historical data"}

    earnings_values = [net_income for net_income in net_income_df if net_income is not None and not np.isnan(net_income)]

    n = len(earnings_values)
    
    if n < 4:  # Need at least 4 periods for trend analysis
        return {"score": 0, "details": "Insufficient historical data"}
    
    score = 0
    reasoning = []

    years_back = 0

    if n >= 4:
        while years_back < n:
            if earnings_values[years_back] is not None and not np.isnan(earnings_values[years_back]):
                years_back += 1
            else:
                break

        earnings_values = earnings_values[:years_back]
        try:
            earnings_growth = all(earnings_values[i] > earnings_values[i+1] and not np.isnan(earnings_values[i]) and not np.isnan(earnings_values[i+1]) for i in range(len(earnings_values) - 1))
        except IndexError:
            return {"score": 0, "details": "Error analyzing earnings growth pattern"}

    if earnings_growth:
        score += 3
        reasoning.append("Consistent earnings growth over past periods")
    else:
        reasoning.append("Inconsistent earnings growth pattern")

    # Calculate growth rate
    if len(earnings_values) >= 2:
        try:
            if not np.isnan(earnings_values[0]) and not np.isnan(earnings_values[-1]) and earnings_values[-1] != 0:
                growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
                reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
            else:
                reasoning.append("Cannot calculate growth rate due to invalid earnings values")
        except (ZeroDivisionError, IndexError, Exception) as e:
            reasoning.append(f"Error calculating growth rate: {str(e)}")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }
    

def calculate_intrinsic_value(metrics, growth_rate, discount_rate, terminal_multiple, projection_years):
    """Intrinsic value calculated using DCF model with owner earnings"""

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
        maintenance_capex = capex * 0.75
        owner_earnings = net_income + depreciation - maintenance_capex
    except Exception as e:
        return {"value": None, "details": [f"Error calculating owner earnings: {str(e)}"]}

    # Get shares Outstanding
    shares_outstanding = metrics.get("shares_outstanding", None)
    if shares_outstanding is None or np.isnan(shares_outstanding):
        return {"value": None, "details": ["Shares outstanding data not available"]}

    try:
        # Calculate future value
        future_value = 0
        for year in range(1, projection_years + 1):
            future_earnings = owner_earnings * (1 + growth_rate) ** year
            present_value = future_earnings / (1 + discount_rate) ** year
            future_value += present_value

        # Calculate terminal value
        terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / (1 + discount_rate) ** projection_years

        intrinsic_value = future_value + terminal_value
    except Exception as e:
        return {"value": None, "details": [f"Error in DCF calculation: {str(e)}"]}

    return {
        "intrinsic_value": f"${intrinsic_value:,.2f}",
        "owner_earnings": f"${owner_earnings:,.2f}",
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }


def calculate_buffett_analysis_data(ticker: str, growth_rate: float, discount_rate: float, terminal_multiple: float, projection_years: int):
    """Analyzes stocks using Buffett's principles and returns a dictionary of the analysis data"""

    try:
        metrics = compute_financial_metrics(ticker)
    except Exception as e:
        return {
            "ticker": ticker,
            "signal": "neutral",
            "score": 0,
            "max_score": 10,
            "error": f"Failed to compute financial metrics: {str(e)}"
        }

    try:
        fundamental_analysis = analyse_fundamentals(metrics)
    except Exception as e:
        fundamental_analysis = {
            "score": 0, 
            "details": f"Error in fundamental analysis: {str(e)}"
        }

    try:
        consistency_analysis = analyse_consistency(ticker)
    except Exception as e:
        consistency_analysis = {
            "score": 0, 
            "details": f"Error in consistency analysis: {str(e)}"
        }

    try:
        intrinsic_value_analysis = calculate_intrinsic_value(metrics, growth_rate, discount_rate, terminal_multiple, projection_years)
    except Exception as e:
        intrinsic_value_analysis = {
            "value": None, 
            "details": [f"Error in intrinsic value calculation: {str(e)}"]
        }

    fundamental_analysis_score = fundamental_analysis.get("score", 0)
    consistency_analysis_score = consistency_analysis.get("score", 0)

    total_score = fundamental_analysis_score + consistency_analysis_score
    max_possible_score = 10

    margin_of_safety = None
    intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")
    market_cap = metrics.get("market_cap")

    if intrinsic_value and market_cap and not np.isnan(market_cap):
        try:
            # Extract numeric value from formatted string
            intrinsic_value_numeric = float(intrinsic_value.replace("$", "").replace(",", ""))
            margin_of_safety = (intrinsic_value_numeric - market_cap) / market_cap

            # Add to score if there's a good margin of safety (>30%)
            if margin_of_safety is not None and margin_of_safety > 0.3:
                total_score += 2
                max_possible_score += 2
        except (ValueError, ZeroDivisionError, Exception) as e:
            intrinsic_value_analysis["details"].append(f"Error calculating margin of safety: {str(e)}")

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
        "margin_of_safety": margin_of_safety,
    }

    return buffett_analysis_data


if __name__ == "__main__":
    buffett_analysis_data = calculate_buffett_analysis_data("AAPL", 0.05, 0.09, 12, 10)
    print(buffett_analysis_data)








