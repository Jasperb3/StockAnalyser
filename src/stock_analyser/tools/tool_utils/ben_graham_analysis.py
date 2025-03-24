import yfinance as yf
import math
import numpy as np
import pandas as pd
from stock_analyser.utils.convert_currency import convert_currency


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


def analyse_earnings_stability(ticker: str):
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.

    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol

    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0
    details = []

    try:
        company_ticker = yf.Ticker(ticker)
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Failed to get ticker data: {str(e)}",
        }

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No financial data available",
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving financial data: {str(e)}",
        }

    # Get EPS data, trying alternative row names if needed
    eps_vals_df = safe_get_row(financials, "Diluted EPS", ["Basic EPS"])

    if eps_vals_df is None:
        return {
            "score": score,
            "max_score": max_score,
            "details": "EPS data not available in financial statements",
        }

    eps_vals = filter_valid_values(eps_vals_df)
    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data (need at least 2 periods).")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)

    # Add score for positive EPS years (but avoid double counting with the next section)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append(f"EPS was positive in all {total_eps_years} available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append(
            f"EPS was positive in {positive_eps_years} of {total_eps_years} periods ({positive_eps_years / total_eps_years:.0%})"
        )
    else:
        details.append(
            f"EPS was negative in {total_eps_years - positive_eps_years} of {total_eps_years} periods."
        )

    max_score += 3  # Increment max_score

    # 2. EPS growth from earliest to latest
    # Note: yfinance data is typically in reverse chronological order (newest first)
    # So eps_vals[0] is the latest (newest) and eps_vals[-1] is the earliest (oldest)
    try:
        latest_eps = eps_vals[0]
        earliest_eps = None  # Initialize

        # Find a valid (non-zero) earliest EPS
        for i in range(len(eps_vals) - 1, -1, -1):
            if eps_vals[i] != 0:
                earliest_eps = eps_vals[i]
                break

        if (
            earliest_eps is not None
            and not np.isnan(earliest_eps)
            and not np.isnan(latest_eps)
        ):
            growth_percentage = (latest_eps - earliest_eps) / abs(earliest_eps)

            if latest_eps > earliest_eps:
                score += 1
                details.append(
                    f"EPS grew from {earliest_eps:.2f} to {latest_eps:.2f} ({growth_percentage:.2%} growth)."
                )
            else:
                details.append(
                    f"EPS declined from {earliest_eps:.2f} to {latest_eps:.2f} ({growth_percentage:.2%} change)."
                )

        elif (
            len(eps_vals) > 1
        ):  # We did NOT find a valid earliest_eps, but there are at least two values
            details.append("Cannot calculate EPS growth due to zero or invalid values.")

        else:  # We did not find a valid earliest_eps AND there is only one value
            details.append("Cannot calculate EPS growth due to zero or invalid values.")

    except (IndexError, ZeroDivisionError, Exception) as e:
        details.append(f"Error calculating EPS growth: {str(e)}")

    max_score += 1

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_financial_strength(ticker: str):
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).

    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol

    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0
    details = []

    try:
        company_ticker = yf.Ticker(ticker)
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Failed to get ticker data for financial strength analysis: {str(e)}",
        }

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No cash flow data available",
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving cash flow data for financial strength analysis: {str(e)}",
        }

    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No balance sheet data available",
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving balance sheet data for financial strength analysis: {str(e)}",
        }

    # Extract metrics with safe handling of None/NaN values
    try:
        total_assets = balance_sheet.loc["Total Assets"].dropna().iloc[0]
        total_liabilities = (
            balance_sheet.loc["Total Liabilities Net Minority Interest"]
            .dropna()
            .iloc[0]
        )
        current_assets = balance_sheet.loc["Current Assets"].dropna().iloc[0]
        current_liabilities = balance_sheet.loc["Current Liabilities"].dropna().iloc[0]
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving balance sheet data: {str(e)}",
        }
          

    # 1. Current ratio
    if (
        current_assets is not None
        and current_liabilities is not None
        and not np.isnan(current_assets)
        and not np.isnan(current_liabilities)
        and current_liabilities > 0
    ):
        try:
            current_ratio = current_assets / current_liabilities
            if current_ratio >= 2.0:
                score += 2
                details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
            elif current_ratio >= 1.5:
                score += 1
                details.append(
                    f"Current ratio = {current_ratio:.2f} (moderately strong)."
                )
            else:
                details.append(
                    f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity)."
                )

            max_score += 2  # Increment max_score

        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating current ratio: {str(e)}")
    else:
        details.append(
            "Cannot compute current ratio (missing or invalid current assets/liabilities data)."
        )

    # 2. Debt vs. Assets
    if (
        total_assets is not None
        and total_liabilities is not None
        and not np.isnan(total_assets)
        and not np.isnan(total_liabilities)
        and total_assets > 0
    ):
        try:
            debt_ratio = total_liabilities / total_assets
            if debt_ratio < 0.5:
                score += 2
                details.append(
                    f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative)."
                )
            elif debt_ratio < 0.8:
                score += 1
                details.append(
                    f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable."
                )
            else:
                details.append(
                    f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards."
                )

            max_score += 2  # Increment max_score

        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating debt ratio: {str(e)}")
    else:
        details.append(
            "Cannot compute debt ratio (missing or invalid assets/liabilities data)."
        )

    # 3. Dividend track record
    cash_dividends_paid_df = safe_get_row(
        cash_flow, "Cash Dividends Paid", ["Dividends Paid", "Common Stock Dividend"]
    )

    if cash_dividends_paid_df is None:
        details.append("Dividend data not available in cash flow statement.")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    div_periods = filter_valid_values(cash_dividends_paid_df)
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        total_periods = len(div_periods)

        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (total_periods // 2 + 1):
                score += 1
                details.append(
                    f"Company paid dividends in {div_paid_years} of {total_periods} reported years."
                )
            else:
                details.append(
                    f"Company has some dividend payments ({div_paid_years} of {total_periods} years)."
                )
        else:
            details.append("Company did not pay dividends in these periods.")

        max_score += 1  # Increment max_score

    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_valuation_graham(ticker: str):
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Analysis results with score and details
    """

    try:
        company_ticker = yf.Ticker(ticker)
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Failed to get ticker data: {str(e)}",
        }

    try:
        info = company_ticker.info
        if info is None:
            return {"score": 0, "max_score": 0, "details": "No info data available"}
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving info data: {str(e)}",
        }

    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No balance sheet data available",
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving balance sheet data: {str(e)}",
        }

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No financial data available",
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving financial data: {str(e)}",
        }

    try:
        exchange_rate = convert_currency(ticker)
    except Exception as e:
        print(f"Error converting currency: {str(e)}")
        exchange_rate = 1

    market_cap = info.get("marketCap")

    if not info or market_cap is None or market_cap <= 0 or np.isnan(market_cap):
        return {
            "score": 0,
            "max_score": 0,
            "details": "Insufficient data for valuation analysis",
        }

    # Extract metrics with safe handling of None/NaN values
    try:
        current_assets = (
            balance_sheet.loc["Current Assets"].dropna().iloc[0] * exchange_rate
        )
        total_liabilities = (
            balance_sheet.loc["Total Liabilities Net Minority Interest"]
            .dropna()
            .iloc[0]
            * exchange_rate
        )
        stockholders_equity = (
            balance_sheet.loc["Stockholders Equity"].dropna().iloc[0] * exchange_rate
        )
        goodwill = balance_sheet.loc["Goodwill"].dropna().iloc[0] * exchange_rate
        other_intangible_assets = (
            balance_sheet.loc["Other Intangible Assets"].dropna().iloc[0]
            * exchange_rate
        )
        shares_outstanding = info.get("sharesOutstanding")
        total_intangible_assets = goodwill + other_intangible_assets
        tangible_book_value = stockholders_equity - total_intangible_assets
        tangible_book_value_per_share = tangible_book_value / shares_outstanding
        eps = financials.loc["Diluted EPS"].dropna().iloc[0]
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving balance sheet data for Graham valuation: {str(e)}",
        }

    details = []
    score = 0
    max_score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    try:
        net_current_asset_value = current_assets - total_liabilities
        if net_current_asset_value > 0 and shares_outstanding > 0:
            net_current_asset_value_per_share = (
                net_current_asset_value / shares_outstanding
            )
            price_per_share = (
                market_cap / shares_outstanding if shares_outstanding else 0
            )

            details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
            details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
            details.append(f"Price Per Share = {price_per_share:,.2f}")

            if net_current_asset_value > market_cap:
                score += 4  # Very strong Graham signal
                details.append("Net-Net: NCAV > Market Cap (classic Graham deep value)")
            else:
                # For partial net-net discount
                if net_current_asset_value_per_share >= (price_per_share * 0.67):
                    score += 2
                    details.append(
                        "NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount)"
                    )
        else:
            if net_current_asset_value <= 0:
                details.append(
                    "NCAV is negative or zero (current assets don't exceed total liabilities)"
                )
            elif shares_outstanding <= 0:
                details.append(
                    "Cannot calculate per-share values (shares outstanding data invalid)"
                )
            else:
                details.append(
                    "NCAV not exceeding market cap or insufficient data for net-net approach"
                )

        max_score += 4

    except (ZeroDivisionError, Exception) as e:
        details.append(f"Error calculating Net-Net values: {str(e)}")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and tangible_book_value_per_share > 0:
        try:
            graham_number = math.sqrt(22.5 * eps * tangible_book_value_per_share)
            details.append(f"Graham Number = {graham_number:.2f}")
        except (ValueError, Exception) as e:
            details.append(f"Error calculating Graham Number: {str(e)}")
    else:
        if eps <= 0:
            details.append("Unable to compute Graham Number (EPS is negative or zero)")
        elif tangible_book_value_per_share <= 0:
            details.append(
                "Unable to compute Graham Number (Book Value per Share is negative or zero)"
            )
        else:
            details.append(
                "Unable to compute Graham Number (EPS or Book Value missing)"
            )

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        try:
            current_price = market_cap / shares_outstanding
            if current_price > 0:
                margin_of_safety = (graham_number - current_price) / current_price
                details.append(
                    f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}"
                )
                if margin_of_safety > 0.5:
                    score += 3
                    details.append("Price is well below Graham Number (>=50% margin).")
                elif margin_of_safety > 0.2:
                    score += 1
                    details.append("Some margin of safety relative to Graham Number.")
                else:
                    details.append(
                        "Price close to or above Graham Number, low margin of safety."
                    )

                max_score += 3  # Increment max_score
            else:
                details.append(
                    "Current price is zero or invalid; can't compute margin of safety."
                )
        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating margin of safety: {str(e)}")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def calculate_graham_analysis_data(ticker: str):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Complete Graham analysis with signal, score, and detailed components
    """

    try:
        earnings_analysis = analyse_earnings_stability(ticker)
    except Exception as e:
        earnings_analysis = {
            "score": 0,
            "max_score": 0,
            "details": f"Error in earnings stability analysis: {str(e)}",
        }

    try:
        strength_analysis = analyse_financial_strength(ticker)
    except Exception as e:
        strength_analysis = {
            "score": 0,
            "max_score": 0,
            "details": f"Error in financial strength analysis: {str(e)}",
        }

    try:
        valuation_analysis = analyse_valuation_graham(ticker)
    except Exception as e:
        valuation_analysis = {
            "score": 0,
            "max_score": 0,
            "details": f"Error in valuation analysis: {str(e)}",
        }

    total_score = (
        earnings_analysis.get("score", 0)
        + strength_analysis.get("score", 0)
        + valuation_analysis.get("score", 0)
    )
    max_possible_score = (
        earnings_analysis.get("max_score", 0)
        + strength_analysis.get("max_score", 0)
        + valuation_analysis.get("max_score", 0)
    )

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
    TICKER = "PEP"
    graham_analysis_data = calculate_graham_analysis_data(TICKER)
    print(graham_analysis_data)
