from stock_analyser.tools.tool_utils.metrics import compute_financial_metrics
import yfinance as yf
import math
import numpy as np
import pandas as pd


def get_ticker(ticker: str):
    try:
        return yf.Ticker(ticker.upper())
    except Exception as e:
        raise Exception(f"Failed to get ticker data for {ticker}: {str(e)}")


def analyse_earnings_stability(metrics: dict, ticker: str):
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
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
    
    score = 0
    details = []

    if not metrics:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}
    
    try:
        eps_vals_df = financials.loc["Basic EPS"]
    except (KeyError, Exception) as e:
        return {"score": score, "details": f"EPS data not available: {str(e)}"}
    
    if eps_vals_df.empty:
        return {"score": score, "details": "No EPS data available"}
    
    eps_vals = [earnings_per_share for earnings_per_share in eps_vals_df if earnings_per_share is not None and not np.isnan(earnings_per_share)]

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}
    
    for years in range(len(eps_vals)):
        if eps_vals[years] > 0 and not np.isnan(eps_vals[years]):
            score += 1

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0 and not np.isnan(e))
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    
    # 2. EPS growth from earliest to latest
    try:
        if not np.isnan(eps_vals[-1]) and not np.isnan(eps_vals[0]) and eps_vals[-1] != 0:
            if eps_vals[-1] < eps_vals[0]:
                score += 1
                growth_percentage = (eps_vals[0] - eps_vals[-1]) / abs(eps_vals[-1])
                details.append(f"EPS grew from {eps_vals[-1]} to {eps_vals[0]} ({growth_percentage:.2%} growth).")
            else:
                details.append("EPS did not grow from earliest to latest period.")
        else:
            details.append("Cannot calculate EPS growth due to invalid values.")
    except (IndexError, ZeroDivisionError, Exception) as e:
        details.append(f"Error calculating EPS growth: {str(e)}")

    return {"score": score, "details": "; ".join(details)}


def analyse_financial_strength(metrics: dict, ticker: str):
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow.empty:
            return {"score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}

    score = 0
    details = []

    if not metrics:
        return {"score": score, "details": "Insufficient data for financial strength analysis"}
    
    total_assets = metrics.get("total_assets", 0)
    total_liabilities = metrics.get("total_liabilities", 0)
    current_assets = metrics.get("current_assets", 0)
    current_liabilities = metrics.get("current_liabilities", 0)

    # 1. Current ratio
    if current_liabilities is not None and current_liabilities > 0 and not np.isnan(current_liabilities) and current_assets is not None and not np.isnan(current_assets):
        try:
            current_ratio = current_assets / current_liabilities
            if current_ratio >= 2.0:
                score += 2
                details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
            elif current_ratio >= 1.5:
                score += 1
                details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
            else:
                details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating current ratio: {str(e)}")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    if total_assets is not None and total_assets > 0 and not np.isnan(total_assets) and total_liabilities is not None and not np.isnan(total_liabilities):
        try:
            debt_ratio = total_liabilities / total_assets
            if debt_ratio < 0.5:
                score += 2
                details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
            elif debt_ratio < 0.8:
                score += 1
                details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
            else:
                details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating debt ratio: {str(e)}")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    try:
        cash_dividends_paid_df = cash_flow.loc["Cash Dividends Paid"]
    except (KeyError, Exception) as e:
        details.append(f"Dividend data not available: {str(e)}")
        return {"score": score, "details": "; ".join(details)}

    if cash_dividends_paid_df.empty:
        details.append("No dividend data available to assess payout consistency.")
        return {"score": score, "details": "; ".join(details)}

    div_periods = [dividend_and_other_cash_distribution for dividend_and_other_cash_distribution in cash_dividends_paid_df if dividend_and_other_cash_distribution is not None and not np.isnan(dividend_and_other_cash_distribution)]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0 and not np.isnan(d))
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("Company paid dividends in the majority of the reported years.")
            else:
                details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "details": "; ".join(details)}


def analyse_valuation_graham(metrics: dict, ticker: str):
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}

    market_cap = metrics.get("market_cap")

    if not metrics or market_cap is None or market_cap <= 0 or np.isnan(market_cap):
        return {"score": 0, "details": "Insufficient data for valuation analysis"}
    
    current_assets = metrics.get("current_assets", 0)
    total_liabilities = metrics.get("total_liabilities", 0)
    book_value_ps = metrics.get("book_value_per_share", 0)
    eps = metrics.get("earnings_per_share", 0)
    shares_outstanding = metrics.get("shares_outstanding", 0)

    # Check for None or NaN values
    if current_assets is None or np.isnan(current_assets):
        current_assets = 0
    if total_liabilities is None or np.isnan(total_liabilities):
        total_liabilities = 0
    if book_value_ps is None or np.isnan(book_value_ps):
        book_value_ps = 0
    if eps is None or np.isnan(eps):
        eps = 0
    if shares_outstanding is None or np.isnan(shares_outstanding):
        shares_outstanding = 0

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    try:
        net_current_asset_value = current_assets - total_liabilities
        if net_current_asset_value > 0 and shares_outstanding > 0:
            net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
            price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

            details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
            details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
            details.append(f"Price Per Share = {price_per_share:,.2f}")

            if net_current_asset_value > market_cap:
                score += 4  # Very strong Graham signal
                details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
            else:
                # For partial net-net discount
                if net_current_asset_value_per_share >= (price_per_share * 0.67):
                    score += 2
                    details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
        else:
            details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")
    except (ZeroDivisionError, Exception) as e:
        details.append(f"Error calculating Net-Net values: {str(e)}")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        try:
            graham_number = math.sqrt(22.5 * eps * book_value_ps)
            details.append(f"Graham Number = {graham_number:.2f}")
        except (ValueError, Exception) as e:
            details.append(f"Error calculating Graham Number: {str(e)}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        try:
            current_price = market_cap / shares_outstanding
            if current_price > 0:
                margin_of_safety = (graham_number - current_price) / current_price
                details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
                if margin_of_safety > 0.5:
                    score += 3
                    details.append("Price is well below Graham Number (>=50% margin).")
                elif margin_of_safety > 0.2:
                    score += 1
                    details.append("Some margin of safety relative to Graham Number.")
                else:
                    details.append("Price close to or above Graham Number, low margin of safety.")
            else:
                details.append("Current price is zero or invalid; can't compute margin of safety.")
        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating margin of safety: {str(e)}")

    return {"score": score, "details": "; ".join(details)}
    

def calculate_graham_analysis_data(ticker: str):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    try:
        metrics = compute_financial_metrics(ticker)
    except Exception as e:
        return {
            "signal": "neutral",
            "score": 0,
            "max_score": 15,
            "error": f"Failed to compute financial metrics: {str(e)}"
        }

    try:
        earnings_analysis = analyse_earnings_stability(metrics, ticker)
    except Exception as e:
        earnings_analysis = {
            "score": 0, 
            "details": f"Error in earnings stability analysis: {str(e)}"
        }

    try:
        strength_analysis = analyse_financial_strength(metrics, ticker)
    except Exception as e:
        strength_analysis = {
            "score": 0, 
            "details": f"Error in financial strength analysis: {str(e)}"
        }

    try:
        valuation_analysis = analyse_valuation_graham(metrics, ticker)
    except Exception as e:
        valuation_analysis = {
            "score": 0, 
            "details": f"Error in valuation analysis: {str(e)}"
        }

    total_score = earnings_analysis.get("score", 0) + strength_analysis.get("score", 0) + valuation_analysis.get("score", 0)
    max_possible_score = 15

    # Map total_score to signal
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    
    graham_analysis_data = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "earnings_analysis": earnings_analysis,
        "strength_analysis": strength_analysis,
        "valuation_analysis": valuation_analysis,
    }

    return graham_analysis_data


if __name__ == "__main__":
    TICKER = "NVDA"
    graham_analysis_data = calculate_graham_analysis_data(TICKER)
    print(graham_analysis_data)










