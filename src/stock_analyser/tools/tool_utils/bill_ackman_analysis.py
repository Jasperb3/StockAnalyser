"""
Bill Ackman investment analysis module.

Analyzes stocks using Bill Ackman's activist value investing approach:
- Business quality (revenue growth, margins, FCF, ROE, brand value)
- Financial discipline (debt management, capital returns)
- Activism potential (operational improvement opportunities)
- Intrinsic value via DCF with margin of safety
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from stock_analyser.utils.convert_currency import convert_currency
from stock_analyser.tools.tool_utils.analysis_helpers import (
    get_ticker,
    safe_get_row,
    filter_valid_values,
    safe_divide,
    create_error_result,
)
from stock_analyser.tools.tool_utils.analysis_constants import (
    REVENUE_GROWTH_EXCEPTIONAL,
    REVENUE_GROWTH_MODERATE,
    OPERATING_MARGIN_EXCELLENT,
    ROE_EXCELLENT,
    DEBT_TO_EQUITY_MODERATE,
    MARGIN_OF_SAFETY_LARGE,
    MARGIN_OF_SAFETY_MODERATE,
    MIN_PERIODS_FOR_ANALYSIS,
    AckmanDCFDefaults,
)


def analyse_business_quality(ticker: str) -> Dict[str, Any]:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages, and potential for long-term growth.

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

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return create_error_result("No cash flow data available")
    except Exception as e:
        return create_error_result(f"Error retrieving cash flow data: {str(e)}")

    # 1. Multi-period revenue growth analysis
    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])

    if revenue_df is not None and not revenue_df.empty:
        revenues = filter_valid_values(revenue_df)

        if len(revenues) >= MIN_PERIODS_FOR_ANALYSIS:
            latest = revenues[0]
            earliest = None

            # Find valid earliest revenue (non-zero)
            for i in range(len(revenues) - 1, -1, -1):
                if revenues[i] != 0:
                    earliest = revenues[i]
                    break

            if earliest is not None and earliest != 0:
                growth_rate = safe_divide(latest - earliest, abs(earliest))

                if growth_rate > REVENUE_GROWTH_EXCEPTIONAL:
                    score += 2
                    details.append(
                        f"Revenue grew by {(growth_rate * 100):.1f}% over the full period (strong growth)."
                    )
                else:
                    score += 1
                    details.append(
                        f"Revenue growth is positive but under {REVENUE_GROWTH_EXCEPTIONAL:.0%} cumulatively ({(growth_rate * 100):.1f}%)."
                    )
            else:
                details.append("Cannot calculate revenue growth (zero or negative base value)")
        else:
            details.append(
                f"Not enough revenue data for multi-period trend (need at least {MIN_PERIODS_FOR_ANALYSIS} periods)."
            )
    else:
        details.append("Revenue data not available in financial statements.")

    max_score += 2

    # 2. Operating margin and free cash flow consistency
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    fcf_vals = filter_valid_values(fcf_df) if fcf_df is not None else []

    total_revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    operating_income_df = safe_get_row(financials, "Operating Income", ["Operating Profit"])

    if (
        total_revenue_df is not None
        and not total_revenue_df.empty
        and operating_income_df is not None
        and not operating_income_df.empty
    ):
        revenues = filter_valid_values(total_revenue_df)
        op_income = filter_valid_values(operating_income_df)

        op_margin_vals = []

        if revenues and op_income and len(revenues) == len(op_income):
            for i in range(len(revenues)):
                op_margin_vals.append(safe_divide(op_income[i], revenues[i]))

        if op_margin_vals:
            above_threshold_count = sum(1 for m in op_margin_vals if m > OPERATING_MARGIN_EXCELLENT)
            if above_threshold_count >= (len(op_margin_vals) // 2 + 1):
                score += 2
                details.append(
                    f"Operating margins exceeded {OPERATING_MARGIN_EXCELLENT:.0%} in {above_threshold_count} of {len(op_margin_vals)} periods (indicates good profitability)."
                )
            else:
                details.append(
                    f"Operating margin exceeded {OPERATING_MARGIN_EXCELLENT:.0%} in only {above_threshold_count} of {len(op_margin_vals)} periods."
                )
        else:
            details.append("Could not calculate operating margins (insufficient data).")
    else:
        details.append("Revenue or operating income data missing for margin analysis.")

    max_score += 2

    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append(
                f"Positive free cash flow in {positive_fcf_count} of {len(fcf_vals)} periods."
            )
        else:
            details.append(
                f"Free cash flow positive in only {positive_fcf_count} of {len(fcf_vals)} periods."
            )
    else:
        details.append("No free cash flow data available for analysis.")

    max_score += 1

    # 3. Return on Equity (ROE) check
    return_on_equity = company_ticker.info.get("returnOnEquity", None)
    if return_on_equity is not None and not np.isnan(return_on_equity):
        if return_on_equity > ROE_EXCELLENT:
            score += 2
            details.append(
                f"High ROE of {return_on_equity:.1%}, indicating a competitive advantage."
            )
        else:
            details.append(
                f"ROE of {return_on_equity:.1%} is moderate and below {ROE_EXCELLENT:.0%} threshold for strong moat."
            )
    else:
        details.append("ROE data not available in company information.")

    max_score += 2

    # 4. Brand/Intangible Assets
    intangible_df = safe_get_row(financials, "Intangible Assets")
    intangible_vals = filter_valid_values(intangible_df)
    if intangible_vals and sum(intangible_vals) > 0:
        details.append("Significant intangible assets may indicate brand value or proprietary tech.")
        score += 1

    max_score += 1

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_financial_discipline(ticker: str) -> Dict[str, Any]:
    """
    Evaluate the company's balance sheet:
    - Debt ratio trends
    - Capital returns to shareholders (dividends, buybacks)

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
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return create_error_result("No balance sheet data available")
    except Exception as e:
        return create_error_result(f"Error retrieving balance sheet data: {str(e)}")

    try:
        cash_flow = company_ticker.cashflow
    except Exception as e:
        cash_flow = pd.DataFrame()
        details.append(f"Error retrieving cash flow data: {str(e)}")

    # 1. Multi-period debt ratio analysis
    total_liabilities_df = safe_get_row(
        balance_sheet,
        "Total Liabilities Net Minority Interest",
        ["Total Liabilities", "Total Debt"],
    )
    shareholder_equity_df = safe_get_row(
        balance_sheet, "Stockholders Equity", ["Total Equity", "Shareholders Equity"]
    )

    if (
        total_liabilities_df is not None
        and not total_liabilities_df.empty
        and shareholder_equity_df is not None
        and not shareholder_equity_df.empty
    ):
        total_liabilities = filter_valid_values(total_liabilities_df)
        shareholder_equity = filter_valid_values(shareholder_equity_df)

        debt_to_equity_vals = []
        if (
            total_liabilities
            and shareholder_equity
            and len(total_liabilities) == len(shareholder_equity)
        ):
            for i in range(len(total_liabilities)):
                debt_to_equity_vals.append(
                    safe_divide(total_liabilities[i], shareholder_equity[i])
                )

        if debt_to_equity_vals:
            below_threshold_count = sum(1 for d in debt_to_equity_vals if d < DEBT_TO_EQUITY_MODERATE)
            if below_threshold_count >= (len(debt_to_equity_vals) // 2 + 1):
                score += 2
                details.append(
                    f"Debt-to-equity < {DEBT_TO_EQUITY_MODERATE} for {below_threshold_count} of {len(debt_to_equity_vals)} periods (reasonable leverage)."
                )
            else:
                details.append(
                    f"Debt-to-equity >= {DEBT_TO_EQUITY_MODERATE} in {len(debt_to_equity_vals) - below_threshold_count} of {len(debt_to_equity_vals)} periods (could be high leverage)."
                )
        else:
            # Fallback to liabilities/assets ratio
            total_assets_df = safe_get_row(balance_sheet, "Total Assets")

            if total_assets_df is not None and not total_assets_df.empty:
                total_assets = filter_valid_values(total_assets_df)

                liab_to_assets = []
                if (
                    total_liabilities
                    and total_assets
                    and len(total_liabilities) == len(total_assets)
                ):
                    for i in range(len(total_liabilities)):
                        liab_to_assets.append(
                            safe_divide(total_liabilities[i], total_assets[i])
                        )

                if liab_to_assets:
                    below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
                    if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                        score += 2
                        details.append(
                            f"Liabilities-to-assets < 50% for {below_50pct_count} of {len(liab_to_assets)} periods."
                        )
                    else:
                        details.append(
                            f"Liabilities-to-assets >= 50% in {len(liab_to_assets) - below_50pct_count} of {len(liab_to_assets)} periods."
                        )
                else:
                    details.append("Could not calculate leverage ratios (insufficient data).")
            else:
                details.append("Total assets data not available for leverage analysis.")
    else:
        details.append("Liabilities or equity data missing for debt ratio analysis.")

    max_score += 2

    # 2. Dividend payments
    dividends_df = safe_get_row(
        cash_flow, "Cash Dividends Paid", ["Dividends Paid", "Common Stock Dividend"]
    )

    if dividends_df is not None and not dividends_df.empty:
        dividends_list = filter_valid_values(dividends_df)
        if dividends_list:
            paying_dividends_count = sum(1 for d in dividends_list if d < 0)
            if paying_dividends_count >= (len(dividends_list) // 2 + 1):
                score += 1
                details.append(
                    f"Company paid dividends in {paying_dividends_count} of {len(dividends_list)} periods."
                )
            else:
                details.append(
                    f"Dividends paid in only {paying_dividends_count} of {len(dividends_list)} periods."
                )
        else:
            details.append("No valid dividend data found across periods.")
    else:
        details.append("Dividend data not available in cash flow statement.")

    max_score += 1

    # 3. Share buybacks (decreasing share count)
    shares_df = safe_get_row(
        balance_sheet,
        "Share Issued",
        ["Common Stock Shares Outstanding", "Ordinary Shares Number"],
    )

    if shares_df is not None and not shares_df.empty:
        shares = filter_valid_values(shares_df)
        if len(shares) >= MIN_PERIODS_FOR_ANALYSIS:
            latest = shares[0]
            earliest = None

            for i in range(len(shares) - 1, -1, -1):
                if shares[i] != 0:
                    earliest = shares[i]
                    break

            if earliest is not None:
                if latest < earliest:
                    score += 1
                    details.append(
                        f"Outstanding shares decreased from {earliest:,.0f} to {latest:,.0f} (possible buybacks)."
                    )
                else:
                    details.append(
                        f"Outstanding shares increased from {earliest:,.0f} to {latest:,.0f}."
                    )
            else:
                details.append("Cannot calculate share change due to zero or invalid values.")
        else:
            details.append(
                f"Insufficient share count data to assess buybacks (need at least {MIN_PERIODS_FOR_ANALYSIS} periods)."
            )
    else:
        details.append("Outstanding shares data not available in balance sheet.")

    max_score += 1

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_activism_potential(ticker: str) -> Dict[str, Any]:
    """
    Bill Ackman often engages in activism if a company has a decent brand or moat
    but is underperforming operationally.

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
        income_statement = company_ticker.income_stmt
        if income_statement is None or income_statement.empty:
            return create_error_result("No income statement data available")
    except Exception as e:
        return create_error_result(f"Error retrieving income statement data: {str(e)}")

    # Check revenue growth vs. operating margin
    total_revenues = safe_get_row(income_statement, "Total Revenue")
    if total_revenues is None or total_revenues.empty:
        return create_error_result("No total revenue data available")

    total_revenues = filter_valid_values(total_revenues)

    op_income = safe_get_row(income_statement, "Operating Income")
    if op_income is None or op_income.empty:
        return create_error_result("No operating income data available")

    op_income = filter_valid_values(op_income)

    # Use safe_divide for operating margins calculation
    op_margins = [
        safe_divide(i, r) * 100 for i, r in zip(op_income, total_revenues)
    ]

    if len(total_revenues) < MIN_PERIODS_FOR_ANALYSIS or not op_margins:
        return create_error_result(
            "Not enough data to assess activism potential (need multi-year revenue + margins)."
        )

    initial, final = total_revenues[-1], total_revenues[0]
    revenue_growth = safe_divide(final - initial, abs(initial)) if initial != 0 else 0
    avg_margin = safe_divide(sum(op_margins), len(op_margins))

    # If there's decent revenue growth but margins are low, Ackman might see activism potential
    if revenue_growth > REVENUE_GROWTH_MODERATE and avg_margin < 10.0:
        score += 2
        details.append(
            f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low (avg {avg_margin:.1f}%). "
            "Activism could unlock margin improvements."
        )
    else:
        details.append(
            "No clear sign of activism opportunity (either margins are already decent or growth is weak)."
        )

    max_score += 2

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_valuation(
    ticker: str,
    growth_rate: float = AckmanDCFDefaults.GROWTH_RATE,
    discount_rate: float = AckmanDCFDefaults.DISCOUNT_RATE,
    terminal_multiple: float = AckmanDCFDefaults.TERMINAL_MULTIPLE,
    projection_years: int = AckmanDCFDefaults.PROJECTION_YEARS,
) -> Dict[str, Any]:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.

    Args:
        ticker: Stock ticker symbol
        growth_rate: Annual growth rate for projections
        discount_rate: Discount rate for present value calculations
        terminal_multiple: Multiple for terminal value calculation
        projection_years: Number of years to project cash flows

    Returns:
        Analysis results with score, details and valuation metrics
    """
    try:
        company_ticker = get_ticker(ticker)
        exchange_rate = convert_currency(ticker)
    except Exception as e:
        return create_error_result(
            f"Failed to get ticker data: {str(e)}", intrinsic_value=None
        )

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return create_error_result("No cash flow data available", intrinsic_value=None)

        # Only apply exchange rate to numeric columns
        cash_flow = cash_flow.apply(
            lambda x: x * exchange_rate if pd.api.types.is_numeric_dtype(x) else x
        )
    except Exception as e:
        return create_error_result(
            f"Error retrieving cash flow data: {str(e)}", intrinsic_value=None
        )

    market_cap = company_ticker.info.get("marketCap", None)
    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        return create_error_result(
            "Market cap data not available or invalid", intrinsic_value=None
        )

    # Get the most recent FCF value
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    if fcf_df is None or fcf_df.empty:
        return create_error_result(
            "Free cash flow data not available in cash flow statement", intrinsic_value=None
        )

    fcf_vals = filter_valid_values(fcf_df)
    if not fcf_vals:
        return create_error_result(
            "No valid free cash flow values available", intrinsic_value=None
        )

    fcf = fcf_vals[0]

    if fcf <= 0:
        return create_error_result(
            f"Most recent FCF is negative or zero (${fcf:,.0f}), cannot perform valuation",
            intrinsic_value=None,
        )

    try:
        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = safe_divide(future_fcf, (1 + discount_rate) ** year)
            present_value += pv

        # Terminal Value
        terminal_fcf = fcf * (1 + growth_rate) ** projection_years
        terminal_value = safe_divide(
            terminal_fcf * terminal_multiple, (1 + discount_rate) ** projection_years
        )

        intrinsic_value = present_value + terminal_value

        # Margin of safety
        margin_of_safety = safe_divide(intrinsic_value - market_cap, intrinsic_value)

        score = 0
        if margin_of_safety > MARGIN_OF_SAFETY_LARGE:
            score += 3
            safety_description = f"Significant margin of safety (>{MARGIN_OF_SAFETY_LARGE:.0%})"
        elif margin_of_safety > MARGIN_OF_SAFETY_MODERATE:
            score += 1
            safety_description = f"Modest margin of safety ({MARGIN_OF_SAFETY_MODERATE:.0%}-{MARGIN_OF_SAFETY_LARGE:.0%})"
        else:
            safety_description = f"Limited or no margin of safety (<{MARGIN_OF_SAFETY_MODERATE:.0%})"

        details = [
            f"Calculated intrinsic value: ~${intrinsic_value:,.2f}",
            f"Market cap: ~${market_cap:,.2f}",
            f"Margin of safety: {margin_of_safety:.2%}",
            safety_description,
        ]

        max_score = 3

        return {
            "score": score,
            "max_score": max_score,
            "details": "; ".join(details),
            "intrinsic_value": f"${intrinsic_value:,.2f}",
            "margin_of_safety": f"{margin_of_safety:.2%}",
        }
    except Exception as e:
        return create_error_result(
            f"Error in DCF calculation: {str(e)}", intrinsic_value=None
        )


def calculate_bill_ackman_analysis_data(
    ticker: str,
    growth_rate: float = AckmanDCFDefaults.GROWTH_RATE,
    discount_rate: float = AckmanDCFDefaults.DISCOUNT_RATE,
    terminal_multiple: float = AckmanDCFDefaults.TERMINAL_MULTIPLE,
    projection_years: int = AckmanDCFDefaults.PROJECTION_YEARS,
) -> Dict[str, Any]:
    """
    Analyzes stocks using Bill Ackman's investing principles.

    Args:
        ticker: Stock ticker symbol
        growth_rate: Annual growth rate for projections
        discount_rate: Discount rate for present value calculations
        terminal_multiple: Multiple for terminal value calculation
        projection_years: Number of years to project cash flows

    Returns:
        Complete Ackman analysis with signal, score, and detailed components
    """
    try:
        quality_analysis = analyse_business_quality(ticker)
    except Exception as e:
        quality_analysis = create_error_result(f"Error in business quality analysis: {str(e)}")

    try:
        balance_sheet_analysis = analyse_financial_discipline(ticker)
    except Exception as e:
        balance_sheet_analysis = create_error_result(
            f"Error in financial discipline analysis: {str(e)}"
        )

    try:
        activism_analysis = analyse_activism_potential(ticker)
    except Exception as e:
        activism_analysis = create_error_result(
            f"Error in activism potential analysis: {str(e)}"
        )

    try:
        valuation_analysis = analyse_valuation(
            ticker, growth_rate, discount_rate, terminal_multiple, projection_years
        )
    except Exception as e:
        valuation_analysis = create_error_result(
            f"Error in valuation analysis: {str(e)}", intrinsic_value=None
        )

    # Combine scores
    total_score = (
        quality_analysis.get("score", 0)
        + balance_sheet_analysis.get("score", 0)
        + activism_analysis.get("score", 0)
        + valuation_analysis.get("score", 0)
    )
    max_possible_score = (
        quality_analysis.get("max_score", 0)
        + balance_sheet_analysis.get("max_score", 0)
        + activism_analysis.get("max_score", 0)
        + valuation_analysis.get("max_score", 0)
    )

    # Generate signal
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    analysis_data = {
        "ticker": ticker,
        "signal": signal,
        "score": round(total_score, 2),
        "max_score": round(max_possible_score, 2),
        "quality_analysis": quality_analysis,
        "balance_sheet_analysis": balance_sheet_analysis,
        "activism_analysis": activism_analysis,
        "valuation_analysis": valuation_analysis,
    }

    return analysis_data


if __name__ == "__main__":
    print(calculate_bill_ackman_analysis_data("BRK-B", 0.06, 0.10, 15, 5))
