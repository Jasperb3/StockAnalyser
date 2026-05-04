"""
Warren Buffett value investing analysis implementation.

Analyzes stocks using Buffett's refined value investing principles:
1. Strong fundamentals (ROE, low debt, good margins)
2. Consistent earnings growth
3. Durable competitive advantage (moat)
4. Quality management
5. Owner earnings and intrinsic value
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from stock_analyser.utils.convert_currency import convert_currency
from stock_analyser.tools.tool_utils.analysis_helpers import (
    get_ticker,
    safe_get_row,
    filter_valid_values,
    create_error_result,
    safe_divide,
    calculate_growth_rate,
)
from stock_analyser.tools.tool_utils.analysis_constants import (
    ROE_STRONG,
    DEBT_TO_EQUITY_MODERATE,
    OPERATING_MARGIN_STRONG,
    CURRENT_RATIO_GOOD,
    DCFDefaults,
    MARGIN_OF_SAFETY_GOOD,
    SIGNAL_BULLISH_THRESHOLD,
    SIGNAL_BEARISH_THRESHOLD,
    PREFERRED_PERIODS_FOR_ANALYSIS,
)

this_year = datetime.now().year


def analyse_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Analyse the fundamentals of a company.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Error retrieving ticker: {str(e)}")

    try:
        info = company_ticker.info
    except Exception as e:
        return create_error_result(f"Error retrieving company info: {str(e)}")

    score = 0
    max_score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    roe = info.get("returnOnEquity")
    if roe is not None and not np.isnan(roe):
        if roe > ROE_STRONG:
            score += 2
            reasoning.append(f"Strong ROE of {roe:.1%}")
        else:
            reasoning.append(f"Weak ROE of {roe:.1%}")
    else:
        reasoning.append("ROE data not available")

    max_score += 2

    # Check Debt to Equity
    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is not None and not np.isnan(debt_to_equity):
        debt_to_equity /= 100
        if debt_to_equity < DEBT_TO_EQUITY_MODERATE:
            score += 2
            reasoning.append(
                f"Conservative debt levels (D/E ratio: {debt_to_equity:.2f})"
            )
        else:
            reasoning.append(f"High debt to equity ratio of {debt_to_equity:.2f}")
    else:
        reasoning.append("Debt to equity data not available")

    max_score += 2

    # Check Operating Margin
    operating_margin = info.get("operatingMargins")
    if operating_margin is not None and not np.isnan(operating_margin):
        if operating_margin > OPERATING_MARGIN_STRONG:
            score += 2
            reasoning.append(f"Strong operating margins of {operating_margin:.1%}")
        else:
            reasoning.append(f"Weak operating margin of {operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    max_score += 2

    # Check Current Ratio
    current_ratio = info.get("currentRatio")
    if current_ratio is not None and not np.isnan(current_ratio):
        if current_ratio > CURRENT_RATIO_GOOD:
            score += 1
            reasoning.append(
                f"Good liquidity position (current ratio: {current_ratio:.2f})"
            )
        else:
            reasoning.append(
                f"Weak liquidity with current ratio of {current_ratio:.2f}"
            )
    else:
        reasoning.append("Current ratio data not available")

    max_score += 1

    return {"score": score, "max_score": max_score, "details": "; ".join(reasoning)}


def _analyze_period_growth(earnings_values: List[float], n: int) -> Tuple[int, List[str]]:
    """
    Helper function to analyze period-over-period earnings growth.
    Returns score and details list.
    """
    score = 0
    details = []

    for i in range(n - 1):
        if earnings_values[i] > earnings_values[i + 1]:
            score += 1
            details.append(
                f"earnings grew from period {this_year - i - 2} to {this_year - i - 1}"
            )
        else:
            details.append(
                f"earnings did not grow from period {this_year - i - 2} to {this_year - i - 1}"
            )

    return score, details


def analyse_consistency(ticker: str) -> Dict[str, Any]:
    """
    Analyse the consistency of a company's earnings over time.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Error retrieving ticker: {str(e)}")

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return create_error_result("No financial data available")
    except Exception as e:
        return create_error_result(f"Error retrieving financial data: {str(e)}")

    # Get Net Income data, trying alternative row names if needed
    net_income_df = safe_get_row(
        financials, "Net Income", ["Net Income Common Stockholders", "Net Profit"]
    )

    if net_income_df is None:
        return create_error_result(
            "Net Income data not available in financial statements"
        )

    earnings_values = filter_valid_values(net_income_df)
    n = len(earnings_values)

    if n < PREFERRED_PERIODS_FOR_ANALYSIS:
        return create_error_result(
            f"Insufficient historical data (have {n} periods, need at least {PREFERRED_PERIODS_FOR_ANALYSIS})"
        )

    score = 0
    max_score = n - 1
    reasoning = []

    # Check if earnings are consistently growing
    try:
        consistent_earnings_growth = all(
            earnings_values[i] > earnings_values[i + 1] for i in range(n - 1)
        )
    except Exception as e:
        return create_error_result(f"Error analyzing earnings growth pattern: {str(e)}")

    if consistent_earnings_growth:
        score += n
        reasoning.append(f"Consistent earnings growth over {n} periods")
    else:
        reasoning.append(f"Inconsistent earnings growth pattern over {n} periods:")
        # Use helper function to reduce nesting
        period_score, period_details = _analyze_period_growth(earnings_values, n)
        score += period_score
        reasoning.extend(period_details)

    # Calculate growth rate from earliest to latest
    growth_rate = calculate_growth_rate(earnings_values, latest_first=True)

    if growth_rate is not None:
        reasoning.append(
            f"Total earnings growth of {growth_rate:.1%} over past {n} periods"
        )
    else:
        reasoning.append(
            "Cannot calculate growth rate (earliest earnings negative or zero)"
        )

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(reasoning),
    }


def analyse_moat(ticker: str) -> Dict[str, Any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Error retrieving ticker: {str(e)}")

    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return create_error_result("No financial data available")
    except Exception as e:
        return create_error_result(f"Error retrieving financial data: {str(e)}")

    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return create_error_result("No balance sheet data available")
    except Exception as e:
        return create_error_result(f"Error retrieving balance sheet data: {str(e)}")

    if (
        financials is None
        or balance_sheet is None
        or financials.empty
        or balance_sheet.empty
        or len(financials) < 3
        or len(balance_sheet) < 3
    ):
        return create_error_result("Insufficient data for moat analysis")

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []

    net_income = safe_get_row(financials, "Net Income")
    shareholders_equity = safe_get_row(balance_sheet, "Stockholders Equity")

    if net_income is not None and shareholders_equity is not None and len(net_income) >= 3 and len(shareholders_equity) >= 3:
        for i in range(len(net_income)):
            ni = net_income.iloc[i] if isinstance(net_income, pd.Series) else None
            se = shareholders_equity.iloc[i] if isinstance(shareholders_equity, pd.Series) else None

            if (
                ni is not None
                and not np.isnan(ni)
                and se is not None
                and not np.isnan(se)
                and se != 0
            ):
                historical_roes.append(safe_divide(ni, se))

    revenue = safe_get_row(financials, "Total Revenue")
    operating_expenses = safe_get_row(financials, "Operating Expense")

    if revenue is not None and operating_expenses is not None and len(revenue) >= 3 and len(operating_expenses) >= 3:
        for i in range(len(revenue)):
            rev = revenue.iloc[i] if isinstance(revenue, pd.Series) else None
            opex = operating_expenses.iloc[i] if isinstance(operating_expenses, pd.Series) else None

            if (
                rev is not None
                and not np.isnan(rev)
                and opex is not None
                and not np.isnan(opex)
                and rev != 0
            ):
                historical_margins.append(safe_divide(rev - opex, rev))

    # Check for stable or improving ROE
    if len(historical_roes) >= 3:
        stable_roe = all(r > ROE_STRONG for r in historical_roes)
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasoning.append("ROE not consistently above 15%")

    # Check for stable or improving operating margin
    if len(historical_margins) >= 3:
        stable_margin = all(m > OPERATING_MARGIN_STRONG for m in historical_margins)
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


def analyse_management_quality(ticker: str) -> Dict[str, Any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Error retrieving ticker: {str(e)}")

    try:
        cash_flow = company_ticker.cash_flow
    except Exception as e:
        return create_error_result(f"Error retrieving cash flow data: {str(e)}")

    if cash_flow is None or cash_flow.empty:
        return create_error_result("No cash flow data available")

    reasoning = []
    mgmt_score = 0

    # Check for issuance or purchase of equity shares
    net_common_stock_issuance = safe_get_row(cash_flow, "Net Common Stock Issuance")

    if net_common_stock_issuance is None or net_common_stock_issuance.empty:
        return create_error_result("Net common stock issuance data not available")

    issuance_values = filter_valid_values(net_common_stock_issuance)
    if not issuance_values:
        return create_error_result("No valid issuance data")

    if issuance_values[0] < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
    elif issuance_values[0] > 0:
        mgmt_score -= 1
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    # Check for any dividends
    cash_dividends_paid = safe_get_row(cash_flow, "Cash Dividends Paid")

    if cash_dividends_paid is not None and not cash_dividends_paid.empty:
        dividend_values = filter_valid_values(cash_dividends_paid)
        if dividend_values and abs(dividend_values[0]) > 0:
            mgmt_score += 1
            reasoning.append("Company has a track record of paying dividends")
        else:
            reasoning.append("No or minimal dividends paid")
    else:
        reasoning.append("No or minimal dividends paid")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_intrinsic_value(ticker: str) -> Dict[str, Any]:
    """
    Calculate intrinsic value using DCF model with owner earnings.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Intrinsic value analysis with value and details
    """
    try:
        company_ticker = get_ticker(ticker)
        exchange_rate = convert_currency(ticker)
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving ticker: {str(e)}"]}

    try:
        info = company_ticker.info
        financials = company_ticker.financials
        cash_flow = company_ticker.cash_flow
    except Exception as e:
        return {"value": None, "details": [f"Error retrieving financial data: {str(e)}"]}

    # Calculate owner earnings
    net_income = safe_get_row(financials, "Net Income")
    depreciation = safe_get_row(cash_flow, "Depreciation And Amortization")
    capex = safe_get_row(cash_flow, "Capital Expenditure")

    if net_income is None or depreciation is None or capex is None:
        return {"value": None, "details": ["Required financial data not available"]}

    net_income_vals = filter_valid_values(net_income)
    depreciation_vals = filter_valid_values(depreciation)
    capex_vals = filter_valid_values(capex)

    if not all([net_income_vals, depreciation_vals, capex_vals]):
        return {"value": None, "details": ["No valid financial values available"]}

    latest_net_income = net_income_vals[0] * exchange_rate
    latest_depreciation = depreciation_vals[0] * exchange_rate
    latest_capex = capex_vals[0] * exchange_rate

    try:
        # Buffett's owner earnings formula: Net Income + Depreciation - Maintenance CapEx
        # We estimate maintenance CapEx as a percentage of total CapEx
        maintenance_capex = (
            latest_capex * 0.75
        )  # Assuming 75% of CapEx is for maintenance
        owner_earnings = latest_net_income + latest_depreciation - maintenance_capex

        if owner_earnings <= 0:
            return {
                "value": None,
                "details": [
                    f"Calculated owner earnings are negative or zero: ${owner_earnings:,.2f}"
                ],
            }
    except Exception as e:
        return {
            "value": None,
            "details": [f"Error calculating owner earnings: {str(e)}"],
        }

    # Get shares Outstanding
    shares_outstanding = info.get("sharesOutstanding", None)
    market_cap = info.get("marketCap", None)

    if (
        shares_outstanding is None
        or np.isnan(shares_outstanding)
        or shares_outstanding <= 0
    ):
        return {
            "value": None,
            "details": ["Shares outstanding data not available or invalid"],
        }

    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        return {
            "value": None,
            "details": ["Market cap data not available or invalid"],
        }

    # Buffett's DCF assumptions (conservative approach)
    growth_rate = DCFDefaults.GROWTH_RATE
    discount_rate = DCFDefaults.DISCOUNT_RATE
    terminal_multiple = DCFDefaults.TERMINAL_MULTIPLE
    projection_years = DCFDefaults.PROJECTION_YEARS

    try:
        # Calculate future value using current earnings
        future_value = 0
        for year in range(1, projection_years + 1):
            future_earnings = owner_earnings * (1 + growth_rate) ** year
            present_value = safe_divide(future_earnings, (1 + discount_rate) ** year)
            future_value += present_value

        # Calculate terminal value
        terminal_value = safe_divide(
            owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple,
            (1 + discount_rate) ** projection_years
        )

        intrinsic_value = future_value + terminal_value
        intrinsic_value_per_share = safe_divide(intrinsic_value, shares_outstanding)

    except Exception as e:
        return {"value": None, "details": [f"Error in DCF calculation: {str(e)}"]}

    return {
        "market_cap": f"${market_cap:,}",
        "intrinsic_value": f"${intrinsic_value:,.2f}",
        "intrinsic_value_per_share": f"${intrinsic_value_per_share:,.2f}",
        "owner_earnings": f"${owner_earnings:,.2f}",
        "owner_earnings_per_share": f"${safe_divide(owner_earnings, shares_outstanding):,.2f}",
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
            f"Projection years: {projection_years}",
        ],
    }


def calculate_buffett_analysis_data(ticker: str) -> Dict[str, Any]:
    """
    Analyzes stocks using Buffett's principles and returns a dictionary of the analysis data.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Complete Buffett analysis with signal, score, and detailed components
    """
    try:
        fundamental_analysis = analyse_fundamentals(ticker)
    except Exception as e:
        fundamental_analysis = create_error_result(
            f"Error in fundamental analysis: {str(e)}"
        )

    try:
        consistency_analysis = analyse_consistency(ticker)
    except Exception as e:
        consistency_analysis = create_error_result(
            f"Error in consistency analysis: {str(e)}"
        )

    try:
        moat_analysis = analyse_moat(ticker)
    except Exception as e:
        moat_analysis = create_error_result(f"Error in moat analysis: {str(e)}")

    try:
        management_quality = analyse_management_quality(ticker)
    except Exception as e:
        management_quality = create_error_result(
            f"Error in management quality analysis: {str(e)}"
        )

    try:
        intrinsic_value_analysis = calculate_intrinsic_value(ticker)
    except Exception as e:
        intrinsic_value_analysis = {
            "value": None,
            "details": [f"Error in intrinsic value calculation: {str(e)}"],
        }

    fundamental_analysis_score = fundamental_analysis.get("score", 0)
    fundamental_analysis_max_score = fundamental_analysis.get("max_score", 0)

    consistency_analysis_score = consistency_analysis.get("score", 0)
    consistency_analysis_max_score = consistency_analysis.get("max_score", 0)

    moat_analysis_score = moat_analysis.get("score", 0)
    moat_analysis_max_score = moat_analysis.get("max_score", 0)

    management_quality_score = management_quality.get("score", 0)
    management_quality_max_score = management_quality.get("max_score", 0)

    total_score = (
        fundamental_analysis_score
        + consistency_analysis_score
        + moat_analysis_score
        + management_quality_score
    )
    max_possible_score = (
        fundamental_analysis_max_score
        + consistency_analysis_max_score
        + moat_analysis_max_score
        + management_quality_max_score
    )

    margin_of_safety = None

    intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")

    try:
        company_ticker = get_ticker(ticker)
        market_cap = company_ticker.info.get("marketCap")
    except:
        market_cap = None

    if intrinsic_value and market_cap and not np.isnan(market_cap) and market_cap > 0:
        try:
            # Extract numeric value from formatted string
            intrinsic_value_numeric = float(
                intrinsic_value.replace("$", "").replace(",", "")
            )
            margin_of_safety = safe_divide(
                intrinsic_value_numeric - market_cap,
                intrinsic_value_numeric
            )

            # Add to score if there's a good margin of safety (>30%)
            if margin_of_safety > MARGIN_OF_SAFETY_GOOD:
                total_score += 2
                max_possible_score += 2
        except (ValueError, Exception) as e:
            if "details" in intrinsic_value_analysis:
                if isinstance(intrinsic_value_analysis["details"], list):
                    intrinsic_value_analysis["details"].append(
                        f"Error calculating margin of safety: {str(e)}"
                    )
                else:
                    intrinsic_value_analysis["details"] = (
                        f"{intrinsic_value_analysis.get('details', '')}; Error calculating margin of safety: {str(e)}"
                    )

    # Generate trading signal
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
        "market_cap": f"${market_cap:,}" if market_cap else None,
        "margin_of_safety": f"{margin_of_safety:.2%}" if margin_of_safety else None,
    }

    return buffett_analysis_data


if __name__ == "__main__":
    buffett_analysis_data = calculate_buffett_analysis_data("NVDA")
    print(buffett_analysis_data)
