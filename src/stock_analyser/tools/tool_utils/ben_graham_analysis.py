"""
Benjamin Graham value investing analysis implementation.

Analyzes stocks using Graham's classic value-investing principles:
1. Earnings stability over multiple years
2. Solid financial strength (low debt, adequate liquidity)
3. Discount to intrinsic value (Graham Number or net-net)
4. Adequate margin of safety
"""

from typing import Dict, Any
import math
import numpy as np
import pandas as pd

from stock_analyser.utils.convert_currency import convert_currency
from stock_analyser.tools.tool_utils.analysis_helpers import (
    get_ticker,
    safe_get_row,
    filter_valid_values,
    calculate_growth_rate,
    create_error_result,
    safe_divide,
)
from stock_analyser.tools.tool_utils.analysis_constants import (
    MIN_PERIODS_FOR_ANALYSIS,
    CURRENT_RATIO_EXCELLENT,
    CURRENT_RATIO_GOOD,
    DEBT_TO_ASSETS_CONSERVATIVE,
    CONSISTENCY_THRESHOLD_GOOD,
    CONSISTENCY_THRESHOLD_MAJORITY,
    GRAHAM_MARGIN_STRONG,
    GRAHAM_MARGIN_MODERATE,
    SIGNAL_BULLISH_THRESHOLD,
    SIGNAL_BEARISH_THRESHOLD,
)


def analyse_earnings_stability(ticker: str) -> Dict[str, Any]:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    Checks:
    1. Number of years with positive EPS
    2. Growth in EPS from first to last period

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    score = 0
    max_score = 0
    details = []

    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return create_error_result("No financial data available")
    except Exception as e:
        return create_error_result(f"Error retrieving financial data: {str(e)}")

    # Get EPS data, trying alternative row names if needed
    eps_vals_df = safe_get_row(financials, "Diluted EPS", ["Basic EPS"])

    if eps_vals_df is None:
        return {
            "score": score,
            "max_score": max_score,
            "details": "EPS data not available in financial statements",
        }

    eps_vals = filter_valid_values(eps_vals_df)
    if len(eps_vals) < MIN_PERIODS_FOR_ANALYSIS:
        details.append(f"Not enough multi-year EPS data (need at least {MIN_PERIODS_FOR_ANALYSIS} periods).")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)

    if positive_eps_years == total_eps_years:
        score += 3
        details.append(f"EPS was positive in all {total_eps_years} available periods.")
    elif positive_eps_years >= (total_eps_years * CONSISTENCY_THRESHOLD_GOOD):
        score += 2
        details.append(
            f"EPS was positive in {positive_eps_years} of {total_eps_years} periods ({positive_eps_years / total_eps_years:.0%})"
        )
    else:
        details.append(
            f"EPS was negative in {total_eps_years - positive_eps_years} of {total_eps_years} periods."
        )

    max_score += 3

    # 2. EPS growth from earliest to latest
    growth_rate = calculate_growth_rate(eps_vals, latest_first=True)

    if growth_rate is not None:
        latest_eps = eps_vals[0]
        # Find earliest non-zero value
        earliest_eps = None
        for i in range(len(eps_vals) - 1, -1, -1):
            if eps_vals[i] != 0:
                earliest_eps = eps_vals[i]
                break

        if earliest_eps is not None:
            if latest_eps > earliest_eps:
                score += 1
                details.append(
                    f"EPS grew from {earliest_eps:.2f} to {latest_eps:.2f} ({growth_rate:.2%} growth)."
                )
            else:
                details.append(
                    f"EPS declined from {earliest_eps:.2f} to {latest_eps:.2f} ({growth_rate:.2%} change)."
                )
        else:
            details.append("Cannot calculate EPS growth due to zero or invalid earliest value.")
    else:
        details.append("Cannot calculate EPS growth due to insufficient data.")

    max_score += 1

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_financial_strength(ticker: str) -> Dict[str, Any]:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    score = 0
    max_score = 0
    details = []

    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return create_error_result("No cash flow data available")
    except Exception as e:
        return create_error_result(f"Error retrieving cash flow data: {str(e)}")

    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return create_error_result("No balance sheet data available")
    except Exception as e:
        return create_error_result(f"Error retrieving balance sheet data: {str(e)}")

    # Extract metrics with safe handling of None/NaN values
    try:
        total_assets_row = safe_get_row(balance_sheet, "Total Assets")
        total_liabilities_row = safe_get_row(balance_sheet, "Total Liabilities Net Minority Interest")
        current_assets_row = safe_get_row(balance_sheet, "Current Assets")
        current_liabilities_row = safe_get_row(balance_sheet, "Current Liabilities")

        # Check if any rows are None (not using all() to avoid pandas Series ambiguity)
        if (total_assets_row is None or total_liabilities_row is None or
            current_assets_row is None or current_liabilities_row is None):
            return create_error_result("Missing required balance sheet data")

        total_assets_vals = filter_valid_values(total_assets_row)
        total_liabilities_vals = filter_valid_values(total_liabilities_row)
        current_assets_vals = filter_valid_values(current_assets_row)
        current_liabilities_vals = filter_valid_values(current_liabilities_row)

        # Check if any value lists are empty
        if (not total_assets_vals or not total_liabilities_vals or
            not current_assets_vals or not current_liabilities_vals):
            return create_error_result("No valid balance sheet values available")

        total_assets = total_assets_vals[0]
        total_liabilities = total_liabilities_vals[0]
        current_assets = current_assets_vals[0]
        current_liabilities = current_liabilities_vals[0]

    except Exception as e:
        return create_error_result(f"Error extracting balance sheet data: {str(e)}")

    # 1. Current ratio
    current_ratio = safe_divide(current_assets, current_liabilities)
    if current_ratio > 0:
        if current_ratio >= CURRENT_RATIO_EXCELLENT:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>={CURRENT_RATIO_EXCELLENT}: solid).")
        elif current_ratio >= CURRENT_RATIO_GOOD:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<{CURRENT_RATIO_GOOD}: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or invalid data).")

    max_score += 2

    # 2. Debt vs. Assets
    debt_ratio = safe_divide(total_liabilities, total_assets)
    if debt_ratio >= 0:  # Valid calculation
        if debt_ratio < DEBT_TO_ASSETS_CONSERVATIVE:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under {DEBT_TO_ASSETS_CONSERVATIVE} (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing or invalid data).")

    max_score += 2

    # 3. Dividend track record
    cash_dividends_paid_df = safe_get_row(
        cash_flow, "Cash Dividends Paid", ["Dividends Paid", "Common Stock Dividend"]
    )

    if cash_dividends_paid_df is None:
        details.append("Dividend data not available in cash flow statement.")
    else:
        div_periods = filter_valid_values(cash_dividends_paid_df)
        if div_periods:
            # In many data feeds, dividend outflow is shown as negative
            div_paid_years = sum(1 for d in div_periods if d < 0)
            total_periods = len(div_periods)

            if div_paid_years > 0:
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
        else:
            details.append("No valid dividend data available.")

    max_score += 1

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_valuation_graham(ticker: str) -> Dict[str, Any]:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    try:
        info = company_ticker.info
        if info is None:
            return create_error_result("No info data available")
    except Exception as e:
        return create_error_result(f"Error retrieving info data: {str(e)}")

    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return create_error_result("No balance sheet data available")
    except Exception as e:
        return create_error_result(f"Error retrieving balance sheet data: {str(e)}")

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return create_error_result("No financial data available")
    except Exception as e:
        return create_error_result(f"Error retrieving financial data: {str(e)}")

    try:
        exchange_rate = convert_currency(ticker)
    except Exception as e:
        print(f"Warning: Error converting currency: {str(e)}, using rate of 1")
        exchange_rate = 1

    market_cap = info.get("marketCap")

    if not market_cap or market_cap <= 0 or np.isnan(market_cap):
        return create_error_result("Insufficient or invalid market cap data for valuation analysis")

    # Extract metrics with validation
    try:
        current_assets = validate_balance_sheet_value(balance_sheet, "Current Assets", exchange_rate)
        total_liabilities = validate_balance_sheet_value(
            balance_sheet, "Total Liabilities Net Minority Interest", exchange_rate
        )
        stockholders_equity = validate_balance_sheet_value(balance_sheet, "Stockholders Equity", exchange_rate)
        goodwill = validate_balance_sheet_value(balance_sheet, "Goodwill", exchange_rate) or 0
        other_intangible = validate_balance_sheet_value(balance_sheet, "Other Intangible Assets", exchange_rate) or 0
        shares_outstanding = info.get("sharesOutstanding")

        # Check if any critical values are None or zero (avoiding all() with potentially None values)
        if (current_assets is None or total_liabilities is None or
            stockholders_equity is None or shares_outstanding is None or
            shares_outstanding <= 0):
            return create_error_result("Missing required balance sheet values for Graham valuation")

        total_intangible_assets = goodwill + other_intangible
        tangible_book_value = stockholders_equity - total_intangible_assets
        tangible_book_value_per_share = safe_divide(tangible_book_value, shares_outstanding)

        eps_row = safe_get_row(financials, "Diluted EPS")
        if eps_row is None:
            return create_error_result("EPS data not available")

        eps_vals = filter_valid_values(eps_row)
        if not eps_vals:
            return create_error_result("No valid EPS values")

        eps = eps_vals[0]

    except Exception as e:
        return create_error_result(f"Error extracting valuation data: {str(e)}")

    details = []
    score = 0
    max_score = 0

    # 1. Net-Net Check
    try:
        net_current_asset_value = current_assets - total_liabilities
        if net_current_asset_value > 0 and shares_outstanding > 0:
            net_current_asset_value_per_share = safe_divide(net_current_asset_value, shares_outstanding)
            price_per_share = safe_divide(market_cap, shares_outstanding)

            details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
            details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
            details.append(f"Price Per Share = {price_per_share:,.2f}")

            if net_current_asset_value > market_cap:
                score += 4  # Very strong Graham signal
                details.append("Net-Net: NCAV > Market Cap (classic Graham deep value)")
            elif net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append(
                    "NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount)"
                )
            else:
                details.append("No significant net-net discount present")
        else:
            if net_current_asset_value <= 0:
                details.append("NCAV is negative or zero (current assets don't exceed total liabilities)")
            else:
                details.append("NCAV not exceeding market cap")

        max_score += 4

    except Exception as e:
        details.append(f"Error calculating Net-Net values: {str(e)}")

    # 2. Graham Number
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
            details.append("Unable to compute Graham Number (Book Value per Share is negative or zero)")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        try:
            current_price = safe_divide(market_cap, shares_outstanding)
            if current_price > 0:
                margin_of_safety = safe_divide(graham_number - current_price, current_price)
                details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")

                if margin_of_safety > GRAHAM_MARGIN_STRONG:
                    score += 3
                    details.append(f"Price is well below Graham Number (>={GRAHAM_MARGIN_STRONG:.0%} margin).")
                elif margin_of_safety > GRAHAM_MARGIN_MODERATE:
                    score += 1
                    details.append("Some margin of safety relative to Graham Number.")
                else:
                    details.append("Price close to or above Graham Number, low margin of safety.")

                max_score += 3
            else:
                details.append("Current price is zero or invalid; can't compute margin of safety.")
        except Exception as e:
            details.append(f"Error calculating margin of safety: {str(e)}")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def validate_balance_sheet_value(
    balance_sheet: pd.DataFrame,
    row_name: str,
    exchange_rate: float,
    alternative_names: list = None
) -> float:
    """Helper to extract and convert balance sheet values."""
    row = safe_get_row(balance_sheet, row_name, alternative_names)
    if row is None:
        return None
    vals = filter_valid_values(row)
    if not vals:
        return None
    return vals[0] * exchange_rate


def calculate_graham_analysis_data(ticker: str) -> Dict[str, Any]:
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years
    2. Solid financial strength (low debt, adequate liquidity)
    3. Discount to intrinsic value (Graham Number or net-net)
    4. Adequate margin of safety

    Args:
        ticker: Stock ticker symbol

    Returns:
        Complete Graham analysis with signal, score, and detailed components
    """
    try:
        earnings_analysis = analyse_earnings_stability(ticker)
    except Exception as e:
        earnings_analysis = create_error_result(f"Error in earnings stability analysis: {str(e)}")

    try:
        strength_analysis = analyse_financial_strength(ticker)
    except Exception as e:
        strength_analysis = create_error_result(f"Error in financial strength analysis: {str(e)}")

    try:
        valuation_analysis = analyse_valuation_graham(ticker)
    except Exception as e:
        valuation_analysis = create_error_result(f"Error in valuation analysis: {str(e)}")

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
    if max_possible_score > 0:
        score_ratio = total_score / max_possible_score
        if score_ratio >= SIGNAL_BULLISH_THRESHOLD:
            signal = "bullish"
        elif score_ratio <= SIGNAL_BEARISH_THRESHOLD:
            signal = "bearish"
        else:
            signal = "neutral"
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
    TICKER = "MSFT"
    graham_analysis_data = calculate_graham_analysis_data(TICKER)
    print(graham_analysis_data)
