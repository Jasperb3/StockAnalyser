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


def analyse_business_quality(ticker: str):
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages, and potential for long-term growth.
    """
    score = 0
    details = []

    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}

    try:
        financials = company_ticker.financials
        cash_flow = company_ticker.cashflow
        
        if financials.empty:
            return {"score": 0, "details": "No financial data available"}
        if cash_flow.empty:
            return {"score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving financial data: {str(e)}"}

    # 1. Multi-period revenue growth analysis
    try:
        revenue_df = financials.loc["Total Revenue"]
    except (KeyError, Exception) as e:
        details.append(f"Revenue data not available: {str(e)}")
        revenue_df = pd.Series()

    if revenue_df.empty:
        details.append("No revenue data available.")

    revenues = [revenue for revenue in revenue_df if revenue is not None and not np.isnan(revenue)]
    if len(revenues) >= 2:
        try:
            # Check if overall revenue grew from first to last
            initial, final = revenues[-1], revenues[0]
            if initial and final and final > initial and not np.isnan(initial) and not np.isnan(final):
                # Simple growth rate
                growth_rate = (final - initial) / abs(initial)
                if growth_rate > 0.5:  # e.g., 50% growth over the available time
                    score += 2
                    details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period.")
                else:
                    score += 1
                    details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
            else:
                details.append("Revenue did not grow significantly or data insufficient.")
        except (IndexError, ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating growth rate: {str(e)}")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    # We'll check if operating_margin or free_cash_flow are consistently positive/improving
    fcf_df = cash_flow.loc["Free Cash Flow"]

    if fcf_df.empty:
        details.append("No free cash flow data available.")
    
    fcf_vals = [free_cash_flow for free_cash_flow in fcf_df if free_cash_flow is not None and not np.isnan(free_cash_flow)]

    total_revenue_df = financials.loc["Total Revenue"]
    operating_income_df = financials.loc["Operating Income"]

    if total_revenue_df.empty or operating_income_df.empty:
        details.append("Insufficient data for margin consistency analysis")
        return {
            "score": 0,
            "details": "; ".join(details)
        }

    op_margin_vals = []

    revenues = [revenue for revenue in total_revenue_df if revenue is not None and not np.isnan(revenue)]
    op_income = [operating_income for operating_income in operating_income_df if operating_income is not None and not np.isnan(operating_income)]
    
    if revenues and op_income and len(revenues) == len(op_income):
        for i in range(len(revenues)):
            if revenues[i] != 0 and not np.isnan(revenues[i]) and not np.isnan(op_income[i]):
                op_margin_vals.append(op_income[i] / revenues[i])

    
    if op_margin_vals:
        # Check if the majority of operating margins are > 15%
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15%.")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        # Check if free cash flow is positive in most periods
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    # (If you want multi-period ROE, you'd need that in financial_line_items as well.)
    return_on_equity = company_ticker.info.get('returnOnEquity', None)
    if return_on_equity and return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {return_on_equity:.1%}, indicating potential moat.")
    elif return_on_equity:
        details.append(f"ROE of {return_on_equity:.1%} is not indicative of a strong moat.")
    else:
        details.append("ROE data not available in metrics.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyse_financial_discipline(ticker: str):
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    score = 0
    details = []

    company_ticker = get_ticker(ticker)

    financials = company_ticker.financials
    balance_sheet = company_ticker.balance_sheet
    cash_flow = company_ticker.cashflow

    # 1. Multi-period debt ratio or debt_to_equity
    # Check if the company's leverage is stable or improving

    total_liabilities_df = balance_sheet.loc["Total Liabilities Net Minority Interest"]
    shareholder_equity_df = balance_sheet.loc["Stockholders Equity"]

    if total_liabilities_df.empty or shareholder_equity_df.empty:
        details.append("Insufficient data for debt ratio analysis")
    
    total_liabilities = [liability for liability in total_liabilities_df if liability is not None and not np.isnan(liability)]
    shareholder_equity = [equity for equity in shareholder_equity_df if equity is not None and not np.isnan(equity)]

    debt_to_equity_vals = []
    if total_liabilities and shareholder_equity and len(total_liabilities) == len(shareholder_equity):
        for i in range(len(total_liabilities)):
            if total_liabilities[i] != 0 and not np.isnan(total_liabilities[i]) and not np.isnan(shareholder_equity[i]) and shareholder_equity[i] != 0:
                debt_to_equity_vals.append(total_liabilities[i] / shareholder_equity[i])
    
    # If we have multi-year data, see if D/E ratio has gone down or stayed <1 across most periods
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for the majority of periods.")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods.")
    else:
        # Fallback to total_liabilities/total_assets if D/E not available

        total_assets_df = balance_sheet.loc["Total Assets"]
        if total_assets_df.empty:
            details.append("No total assets data available.")

        total_assets = [asset for asset in total_assets_df if asset is not None and not np.isnan(asset)]

        liab_to_assets = []
        if total_liabilities and total_assets and len(total_liabilities) == len(total_assets):
            for i in range(len(total_liabilities)):
                if total_liabilities[i] != 0 and not np.isnan(total_liabilities[i]) and not np.isnan(total_assets[i]) and total_assets[i] != 0:
                    liab_to_assets.append(total_liabilities[i] / total_assets[i])
        
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data available.")
    
    # 2. Capital allocation approach (dividends + share counts)
    # If the company paid dividends or reduced share count over time, it may reflect discipline

    dividends_df = cash_flow.loc["Cash Dividends Paid"]
    dividends_list = [dividend for dividend in dividends_df if dividend is not None and not np.isnan(dividend)]
    if dividends_list:
        # Check if dividends were paid (i.e., negative outflows to shareholders) in most periods
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("Company has a history of returning capital to shareholders (dividends).")
        else:
            details.append("Dividends not consistently paid or no data.")
    else:
        details.append("No dividend data found across periods.")
    
    # Check for decreasing share count (simple approach):
    # We can compare first vs last if we have at least two data points
    shares_df = balance_sheet.loc["Share Issued"]

    if shares_df.empty:
        details.append("No outstanding shares data available.")

    shares = [share for share in shares_df if share is not None and not np.isnan(share)]
    if len(shares) >= 2:
        if shares[-1] < shares[0]:
            score += 1
            details.append("Outstanding shares have decreased over time (possible buybacks).")
        else:
            details.append("Outstanding shares have not decreased over the available periods.")
    else:
        details.append("No multi-period share count data to assess buybacks.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }

    
def analyse_valuation(ticker: str, growth_rate: float = 0.06, discount_rate: float = 0.10, terminal_multiple: float = 15, projection_years: int = 5):
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    We can do a simplified DCF or an FCF-based approach.
    This function currently uses the latest free cash flow only, 
    but you could expand it to use an average or multi-year FCF approach.
    """
    company_ticker = get_ticker(ticker)
    cash_flow = company_ticker.cashflow
    market_cap = company_ticker.info.get('marketCap', None)

    # Example: use the most recent item for FCF
    fcf = cash_flow.loc["Free Cash Flow"].iloc[0]

    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }
    
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                     / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    
    # Compare with market cap => margin of safety
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    score = 0
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1
    
    details = [
        f"Calculated intrinsic value: ~${intrinsic_value:,.2f}",
        f"Market cap: ~${market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]
    
    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": f"${intrinsic_value:,.2f}",
        "margin_of_safety": f"{margin_of_safety:.6%}"
    }


def calculate_bill_ackman_analysis_data(ticker: str, growth_rate: float = 0.06, discount_rate: float = 0.10, terminal_multiple: float = 15, projection_years: int = 5):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data so we can analyze long-term trends.
    """

    quality_analysis = analyse_business_quality(ticker)
    balance_sheet_analysis = analyse_financial_discipline(ticker)
    valuation_analysis = analyse_valuation(ticker, growth_rate, discount_rate, terminal_multiple, projection_years)

    # Combine partial scores or signals
    total_score = quality_analysis["score"] + balance_sheet_analysis["score"] + valuation_analysis["score"]
    max_possible_score = 15  # Adjust weighting as desired
    
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
    print(calculate_bill_ackman_analysis_data("AAPL", 0.06, 0.10, 15, 5))