"""
Charlie Munger investment analysis module.

Analyzes stocks using Charlie Munger's investing principles focusing on:
- Competitive moat strength (ROIC, pricing power, low capital needs)
- Management quality (capital allocation, insider ownership)
- Business predictability (consistent revenue, earnings, margins)
- Valuation based on owner earnings (FCF)
"""

from typing import Dict, Any
from stock_analyser.tools.tool_utils.news_sentiment_util import (
    get_news_sentiment_scores,
)
from stock_analyser.tools.tool_utils.analysis_helpers import (
    get_ticker,
    safe_get_row,
    filter_valid_values,
    safe_divide,
    create_error_result,
)
from stock_analyser.tools.tool_utils.analysis_constants import (
    ROIC_EXCELLENT,
    ROIC_GOOD,
    CONSISTENCY_THRESHOLD_GOOD,
    CONSISTENCY_THRESHOLD_MAJORITY,
    GROSS_MARGIN_EXCELLENT,
    CAPEX_TO_REVENUE_LOW,
    CAPEX_TO_REVENUE_MODERATE,
    FCF_TO_NI_EXCELLENT,
    FCF_TO_NI_GOOD,
    FCF_TO_NI_MODERATE,
    DEBT_TO_EQUITY_CONSERVATIVE,
    DEBT_TO_EQUITY_MODERATE,
    DEBT_TO_EQUITY_HIGH,
    CASH_TO_REVENUE_MIN_IDEAL,
    CASH_TO_REVENUE_MAX_IDEAL,
    REVENUE_GROWTH_GOOD,
    FCF_YIELD_EXCELLENT,
    FCF_YIELD_GOOD,
    FCF_YIELD_FAIR,
    MARGIN_OF_SAFETY_LARGE,
    MARGIN_OF_SAFETY_MODERATE,
    MIN_PERIODS_FOR_ANALYSIS,
)


def analyse_moat_strength(ticker: str) -> Dict[str, Any]:
    """
    Analyze the business's competitive advantage using Munger's approach:
    - Consistent high returns on capital (ROIC)
    - Pricing power (stable/improving gross margins)
    - Low capital requirements
    - Network effects and intangible assets (R&D investments, goodwill)

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
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return create_error_result("No balance sheet data available")
    except Exception as e:
        return create_error_result(f"Error retrieving balance sheet data: {str(e)}")

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return create_error_result("No cash flow data available")
    except Exception as e:
        return create_error_result(f"Error retrieving cash flow data: {str(e)}")

    try:
        income_statement = company_ticker.income_stmt
        if income_statement is None or income_statement.empty:
            return create_error_result("No income statement data available")
    except Exception as e:
        return create_error_result(f"Error retrieving income statement data: {str(e)}")

    # 1. Return on Invested Capital (ROIC) - Munger's favorite metric
    # ROIC = (net income – dividends) / (debt + equity)
    net_income = safe_get_row(income_statement, "Net Income")
    dividends = safe_get_row(cash_flow, "Cash Dividends Paid")
    debt = safe_get_row(balance_sheet, "Total Debt")
    equity = safe_get_row(balance_sheet, "Stockholders Equity")

    if (
        net_income is not None
        and not net_income.empty
        and dividends is not None
        and not dividends.empty
        and debt is not None
        and not debt.empty
        and equity is not None
        and not equity.empty
    ):
        return_on_invested_capital = (net_income - dividends) / (debt + equity)
        roic_values = filter_valid_values(return_on_invested_capital)

        if roic_values and len(roic_values) > 0:
            # Check if ROIC consistently above 15% (Munger's threshold)
            high_roic_count = sum(1 for r in roic_values if r > ROIC_EXCELLENT)
            roic_ratio = safe_divide(high_roic_count, len(roic_values))

            if roic_ratio >= CONSISTENCY_THRESHOLD_GOOD:
                score += 3
                details.append(
                    f"Excellent ROIC: >{ROIC_EXCELLENT:.0%} in {high_roic_count}/{len(roic_values)} periods"
                )
            elif roic_ratio >= CONSISTENCY_THRESHOLD_MAJORITY:
                score += 2
                details.append(
                    f"Good ROIC: >{ROIC_EXCELLENT:.0%} in {high_roic_count}/{len(roic_values)} periods"
                )
            elif high_roic_count > 0:
                score += 1
                details.append(
                    f"Mixed ROIC: >{ROIC_EXCELLENT:.0%} in only {high_roic_count}/{len(roic_values)} periods"
                )
            else:
                details.append(f"Poor ROIC: Never exceeds {ROIC_EXCELLENT:.0%} threshold")
    else:
        details.append("Insufficient ROIC data available")

    max_score += 3

    # 2. Pricing power - check gross margin stability and trends
    gross_profit = safe_get_row(financials, "Gross Profit")
    total_revenue = safe_get_row(financials, "Total Revenue")

    if (
        gross_profit is not None
        and not gross_profit.empty
        and total_revenue is not None
        and not total_revenue.empty
    ):
        gross_margins = gross_profit / total_revenue
        gross_margin_values = filter_valid_values(gross_margins)
        gross_margin_count = len(gross_margin_values)

        if gross_margin_values and gross_margin_count > MIN_PERIODS_FOR_ANALYSIS:
            # Munger likes stable or improving gross margins
            margin_trend = sum(
                1
                for i in range(1, len(gross_margins))
                if gross_margins.iloc[i - 1] >= gross_margins.iloc[i]
            )
            margin_trend_ratio = safe_divide(margin_trend, len(gross_margins))

            if margin_trend_ratio >= 0.7:
                score += 2
                details.append(
                    "Strong pricing power: Gross margins consistently improving"
                )
            elif safe_divide(sum(gross_margins), len(gross_margins)) > GROSS_MARGIN_EXCELLENT:
                score += 1
                avg_margin = safe_divide(sum(gross_margins), len(gross_margins))
                details.append(
                    f"Good pricing power: Average gross margin {avg_margin:.1%}"
                )
            else:
                details.append("Limited pricing power: Low or declining gross margins")
    else:
        details.append("Insufficient gross margin data")

    max_score += 2

    # 3. Capital intensity - Munger prefers low capex businesses
    capex_to_revenue = []
    capex = safe_get_row(cash_flow, "Capital Expenditure")
    capex_values = filter_valid_values(capex)
    revenue_values = filter_valid_values(total_revenue)

    if len(capex_values) >= MIN_PERIODS_FOR_ANALYSIS and len(revenue_values) >= MIN_PERIODS_FOR_ANALYSIS:
        for i in range(min(len(capex_values), len(revenue_values))):
            if revenue_values[i] > 0:
                capex_to_revenue.append(safe_divide(abs(capex_values[i]), revenue_values[i]))

        if capex_to_revenue:
            avg_capex_ratio = safe_divide(sum(capex_to_revenue), len(capex_to_revenue))
            if avg_capex_ratio < CAPEX_TO_REVENUE_LOW:
                score += 2
                details.append(
                    f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue"
                )
            elif avg_capex_ratio < CAPEX_TO_REVENUE_MODERATE:
                score += 1
                details.append(
                    f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue"
                )
            else:
                details.append(
                    f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue"
                )
        else:
            details.append("No capital expenditure data available")
    else:
        details.append("Insufficient data for capital intensity analysis")

    max_score += 2

    # 4. Intangible assets - Munger values R&D and intellectual property
    goodwill_and_intangible_assets = safe_get_row(
        balance_sheet, "Goodwill And Intangible Assets"
    )
    research_and_development = safe_get_row(financials, "Research And Development")

    g_and_i_values = filter_valid_values(goodwill_and_intangible_assets)
    r_and_d_values = filter_valid_values(research_and_development)

    if g_and_i_values and len(g_and_i_values) > 0:
        if sum(g_and_i_values) > 0:
            score += 1
            details.append(
                "Significant goodwill/intangible assets, suggesting brand value or IP"
            )
        else:
            details.append("No goodwill/intangible assets")

    if r_and_d_values and len(r_and_d_values) > 0:
        if sum(r_and_d_values) > 0:
            score += 1
            details.append("Invests in R&D, building intellectual property")
        else:
            details.append("No R&D investment")

    max_score += 2

    final_score = round(min(10, safe_divide(score * 10, max_score, default=0)), 1)

    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details),
    }


def analyse_management_quality(ticker: str) -> Dict[str, Any]:
    """
    Evaluate management quality using Munger's criteria:
    - Capital allocation wisdom
    - Insider ownership and transactions
    - Cash management efficiency
    - Candor and transparency
    - Long-term focus

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

    try:
        income_statement = company_ticker.income_stmt
        if income_statement is None or income_statement.empty:
            return create_error_result("No income statement data available")
    except Exception as e:
        return create_error_result(f"Error retrieving income statement data: {str(e)}")

    # 1. Capital allocation - Check FCF to net income ratio
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    net_income_df = safe_get_row(financials, "Net Income")

    fcf_values = filter_valid_values(fcf_df)
    net_income_values = filter_valid_values(net_income_df)

    if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
        fcf_to_ni_ratios = []
        for i in range(len(fcf_values)):
            if net_income_values[i] and net_income_values[i] > 0:
                fcf_to_ni_ratios.append(safe_divide(fcf_values[i], net_income_values[i]))

        if fcf_to_ni_ratios:
            avg_ratio = safe_divide(sum(fcf_to_ni_ratios), len(fcf_to_ni_ratios))
            if avg_ratio > FCF_TO_NI_EXCELLENT:
                score += 3
                details.append(
                    f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}"
                )
            elif avg_ratio > FCF_TO_NI_GOOD:
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > FCF_TO_NI_MODERATE:
                score += 1
                details.append(
                    f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}"
                )
            else:
                details.append(
                    f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}"
                )
        else:
            details.append("Could not calculate FCF to Net Income ratios")
    else:
        details.append("Missing FCF or Net Income data")

    max_score += 3

    # 2. Debt Management - Munger prefers conservative debt levels
    total_debt = safe_get_row(balance_sheet, "Total Debt")
    shareholders_equity = safe_get_row(balance_sheet, "Stockholders Equity")

    total_debt_values = filter_valid_values(total_debt)
    shareholders_equity_values = filter_valid_values(shareholders_equity)

    if (
        total_debt_values
        and shareholders_equity_values
        and len(total_debt_values) > 0
        and len(shareholders_equity_values) > 0
    ):
        recent_de_ratio = safe_divide(
            total_debt_values[0],
            shareholders_equity_values[0],
            default=float("inf"),
        )

        if recent_de_ratio < DEBT_TO_EQUITY_CONSERVATIVE:
            score += 3
            details.append(
                f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}"
            )
        elif recent_de_ratio < DEBT_TO_EQUITY_MODERATE:
            score += 2
            details.append(
                f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}"
            )
        elif recent_de_ratio < DEBT_TO_EQUITY_HIGH:
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        details.append("Missing debt or equity data")

    max_score += 3

    # 3. Cash management efficiency - Munger values appropriate cash levels
    cash_and_equivalents = safe_get_row(balance_sheet, "Cash And Cash Equivalents")
    total_revenue = safe_get_row(financials, "Total Revenue")

    cash_values = filter_valid_values(cash_and_equivalents)
    revenue_values = filter_valid_values(total_revenue)

    if cash_values and revenue_values and len(cash_values) > 0 and len(revenue_values) > 0:
        cash_to_revenue = safe_divide(cash_values[0], revenue_values[0])

        if CASH_TO_REVENUE_MIN_IDEAL <= cash_to_revenue <= CASH_TO_REVENUE_MAX_IDEAL:
            score += 2
            details.append(
                f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}"
            )
        elif 0.05 <= cash_to_revenue < CASH_TO_REVENUE_MIN_IDEAL or CASH_TO_REVENUE_MAX_IDEAL < cash_to_revenue <= 0.4:
            score += 1
            details.append(
                f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}"
            )
        elif cash_to_revenue > 0.4:
            details.append(
                f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}"
            )
        else:
            details.append(
                f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}"
            )
    else:
        details.append("Insufficient cash or revenue data")

    max_score += 2

    # 4. Insider activity - Munger values skin in the game
    try:
        insider_trades = company_ticker.insider_purchases

        if insider_trades is None or insider_trades.empty:
            details.append("No insider trades data available; defaulting to neutral.")
        else:
            if len(insider_trades) >= 2:
                buys = insider_trades["Trans"].iloc[0]
                sells = insider_trades["Trans"].iloc[1]
                total_trades = buys + sells

                if total_trades > 0:
                    buy_ratio = safe_divide(buys, total_trades)
                    if buy_ratio > 0.7:
                        score += 2
                        details.append(
                            f"Strong insider buying: {buys}/{total_trades} transactions are purchases"
                        )
                    elif buy_ratio > 0.4:
                        score += 1
                        details.append(
                            f"Balanced insider trading: {buys}/{total_trades} transactions are purchases"
                        )
                    elif buy_ratio < 0.1 and sells > 5:
                        score -= 1
                        details.append(
                            f"Concerning insider selling: {sells}/{total_trades} transactions are sales"
                        )
                    else:
                        details.append(
                            f"Mixed insider activity: {buys}/{total_trades} transactions are purchases"
                        )
                else:
                    details.append("No insider trading data available")
            else:
                details.append("Insufficient insider trades data")
    except Exception as e:
        details.append(f"Error analyzing insider trades: {str(e)}")

    max_score += 2

    # 5. Consistency in share count - Munger prefers stable/decreasing shares
    outstanding_shares = safe_get_row(income_statement, "Basic Average Shares")
    share_counts = []

    if outstanding_shares is not None and not outstanding_shares.empty:
        share_counts = filter_valid_values(outstanding_shares)

    if share_counts and len(share_counts) >= MIN_PERIODS_FOR_ANALYSIS:
        if share_counts[0] < share_counts[-1] * 0.95:
            score += 2
            details.append("Shareholder-friendly: Reducing share count over time")
        elif share_counts[0] < share_counts[-1] * 1.05:
            score += 1
            details.append("Stable share count: Limited dilution")
        elif share_counts[0] > share_counts[-1] * 1.2:
            score -= 1
            details.append("Concerning dilution: Share count increased significantly")
        else:
            details.append("Moderate share count increase over time")
    else:
        details.append("Insufficient share count data")

    max_score += 2

    final_score = round(max(0, min(10, safe_divide(score * 10, max_score, default=0))), 2)

    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details),
    }


def analyse_predictability(ticker: str) -> Dict[str, Any]:
    """
    Assess the predictability of the business - Munger strongly prefers businesses
    whose future operations and cashflows are relatively easy to predict.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Analysis results with score and details
    """
    years_back = 4
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

    # 1. Revenue Consistency - Munger values predictable revenue streams
    total_revenue = safe_get_row(financials, "Total Revenue")

    if total_revenue is not None and not total_revenue.empty:
        revenues = filter_valid_values(total_revenue)
        if revenues and len(revenues) >= years_back:
            growth_rates = [
                safe_divide(revenues[i], revenues[i + 1]) - 1
                for i in range(len(revenues) - 1)
                if revenues[i + 1] != 0
            ]
            avg_growth = safe_divide(sum(growth_rates), len(growth_rates))
            growth_volatility = safe_divide(
                sum(abs(r - avg_growth) for r in growth_rates),
                len(growth_rates)
            )

            if avg_growth > REVENUE_GROWTH_GOOD and growth_volatility < 0.1:
                score += 3
                details.append(
                    f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility"
                )
            elif avg_growth > 0 and growth_volatility < 0.2:
                score += 2
                details.append(
                    f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility"
                )
            elif avg_growth > 0:
                score += 1
                details.append(
                    f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility"
                )
            else:
                details.append(
                    f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth"
                )
    else:
        details.append("Insufficient revenue history for predictability analysis")

    max_score += 3

    # 2. Operating income Consistency - Munger values predictable earnings
    operating_income = safe_get_row(financials, "Operating Income")

    if operating_income is not None and not operating_income.empty:
        op_income = filter_valid_values(operating_income)
        if op_income and len(op_income) >= years_back:
            positive_periods = sum(1 for income in op_income if income > 0)
            positive_ratio = safe_divide(positive_periods, len(op_income))

            if positive_periods == len(op_income):
                score += 3
                details.append(
                    "Highly predictable operations: Operating income positive in all periods"
                )
            elif positive_ratio >= CONSISTENCY_THRESHOLD_GOOD:
                score += 2
                details.append(
                    f"Predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods"
                )
            elif positive_ratio >= 0.6:
                score += 1
                details.append(
                    f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods"
                )
            else:
                details.append(
                    f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(op_income)} periods"
                )
    else:
        details.append("Insufficient operating income history")

    max_score += 3

    # 3. Margin consistency - Munger values stable margins
    operating_income = safe_get_row(financials, "Operating Income")
    total_revenue = safe_get_row(financials, "Total Revenue")

    if (
        operating_income is not None
        and not operating_income.empty
        and total_revenue is not None
        and not total_revenue.empty
    ):
        operating_margins = [
            safe_divide(operating_income.iloc[i], total_revenue.iloc[i])
            for i in range(min(len(operating_income), len(total_revenue)))
        ]

        op_margins = filter_valid_values(operating_margins)

        if op_margins and len(op_margins) >= years_back:
            avg_margin = safe_divide(sum(op_margins), len(op_margins))
            margin_volatility = safe_divide(
                sum(abs(m - avg_margin) for m in op_margins),
                len(op_margins)
            )

            if margin_volatility < 0.03:
                score += 2
                details.append(
                    f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility"
                )
            elif margin_volatility < 0.07:
                score += 1
                details.append(
                    f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility"
                )
            else:
                details.append(
                    f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})"
                )
    else:
        details.append("Insufficient margin history")

    max_score += 2

    # 4. Cash generation reliability
    free_cash_flow = safe_get_row(cash_flow, "Free Cash Flow")

    if free_cash_flow is not None and not free_cash_flow.empty:
        fcf_values = filter_valid_values(free_cash_flow)
        if fcf_values and len(fcf_values) >= years_back:
            positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
            fcf_ratio = safe_divide(positive_fcf_periods, len(fcf_values))

            if positive_fcf_periods == len(fcf_values):
                score += 2
                details.append(
                    "Highly predictable cash generation: Positive FCF in all periods"
                )
            elif fcf_ratio >= CONSISTENCY_THRESHOLD_GOOD:
                score += 1
                details.append(
                    f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods"
                )
            else:
                details.append(
                    f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods"
                )
    else:
        details.append("Insufficient free cash flow history")

    max_score += 2

    final_score = round(max(0, min(10, safe_divide(score * 10, max_score, default=0))), 2)

    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details),
    }


def calculate_munger_valuation(ticker: str) -> Dict[str, Any]:
    """
    Calculate intrinsic value using Munger's approach:
    - Focus on owner earnings (approximated by FCF)
    - Simple multiple on normalized earnings
    - Prefer paying a fair price for a wonderful business

    Args:
        ticker: Stock ticker symbol

    Returns:
        Valuation analysis with score and details
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

    market_cap = company_ticker.info.get("marketCap", None)
    if market_cap is None:
        return create_error_result("Market cap data not available")

    free_cash_flow = safe_get_row(cash_flow, "Free Cash Flow")

    if free_cash_flow is None or free_cash_flow.empty:
        return create_error_result("No free cash flow data available")

    fcf_values = filter_valid_values(free_cash_flow)
    if not fcf_values or len(fcf_values) < MIN_PERIODS_FOR_ANALYSIS:
        return create_error_result("Insufficient free cash flow data for valuation")

    # 1. Normalize earnings by taking average of last 3-5 years
    normalized_fcf = safe_divide(
        sum(fcf_values[: min(5, len(fcf_values))]),
        min(5, len(fcf_values))
    )

    if normalized_fcf <= 0:
        return create_error_result(
            f"Negative or zero normalized FCF ({normalized_fcf}), cannot value",
            intrinsic_value=None,
        )

    # 2. Calculate FCF yield (inverse of P/FCF multiple)
    fcf_yield = safe_divide(normalized_fcf, market_cap)

    # 3. Apply Munger's FCF multiple based on business quality
    if fcf_yield > FCF_YIELD_EXCELLENT:
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > FCF_YIELD_GOOD:
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > FCF_YIELD_FAIR:
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

    max_score += 4

    # 4. Calculate simple intrinsic value range
    conservative_value = normalized_fcf * 10
    reasonable_value = normalized_fcf * 15
    optimistic_value = normalized_fcf * 20

    # 5. Calculate margins of safety
    current_to_reasonable = safe_divide(reasonable_value - market_cap, market_cap)

    if current_to_reasonable > MARGIN_OF_SAFETY_LARGE:
        score += 3
        details.append(
            f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value"
        )
    elif current_to_reasonable > MARGIN_OF_SAFETY_MODERATE:
        score += 2
        details.append(
            f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value"
        )
    elif current_to_reasonable > -0.1:
        score += 1
        details.append(
            f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})"
        )
    else:
        details.append(
            f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value"
        )

    max_score += 3

    # 6. Check earnings trajectory for additional context
    if len(fcf_values) >= MIN_PERIODS_FOR_ANALYSIS:
        recent_avg = safe_divide(sum(fcf_values[:3]), 3)
        older_avg = (
            safe_divide(sum(fcf_values[-3:]), 3) if len(fcf_values) >= 6 else fcf_values[-1]
        )

        if recent_avg > older_avg * 1.2:
            score += 3
            details.append("Growing FCF trend adds to intrinsic value")
        elif recent_avg > older_avg:
            score += 2
            details.append("Stable to growing FCF supports valuation")
        else:
            details.append("Declining FCF trend is concerning")

    max_score += 3

    final_score = round(min(10, safe_divide(score * 10, max_score, default=0)), 2)

    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details),
        "intrinsic_value_range": {
            "conservative": f"${conservative_value:,.2f}",
            "reasonable": f"${reasonable_value:,.2f}",
            "optimistic": f"${optimistic_value:,.2f}",
        },
        "fcf_yield": f"{fcf_yield:.2%}",
        "normalized_fcf": f"${normalized_fcf:,.2f}",
    }


def analyse_news_sentiment(ticker: str, number_of_articles: int = 10) -> Dict[str, Any]:
    """
    Analyze recent news sentiment for the company.

    Args:
        ticker: Stock ticker symbol
        number_of_articles: Number of articles to analyze

    Returns:
        Sentiment analysis result
    """
    try:
        sentiment_score = get_news_sentiment_scores(ticker, number_of_articles)

        return {
            "score": sentiment_score,
            "normalized_max_score": 10,
            "details": f"News sentiment score: {sentiment_score} (range: -10 to 10)",
        }
    except Exception as e:
        return create_error_result(f"Could not analyze news sentiment: {str(e)}")


def analyse_charlie_munger_valuation(ticker: str) -> Dict[str, Any]:
    """
    Analyzes stocks using Charlie Munger's investing principles and mental models.
    Focuses on moat strength, management quality, predictability, and valuation.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Complete Munger analysis with signal, score, and detailed components
    """
    try:
        moat_analysis = analyse_moat_strength(ticker)
    except Exception as e:
        moat_analysis = create_error_result(f"Error in moat strength analysis: {str(e)}")

    try:
        management_analysis = analyse_management_quality(ticker)
    except Exception as e:
        management_analysis = create_error_result(
            f"Error in management quality analysis: {str(e)}"
        )

    try:
        predictability_analysis = analyse_predictability(ticker)
    except Exception as e:
        predictability_analysis = create_error_result(
            f"Error in predictability analysis: {str(e)}"
        )

    try:
        valuation_analysis = calculate_munger_valuation(ticker)
    except Exception as e:
        valuation_analysis = create_error_result(
            f"Error in valuation analysis: {str(e)}"
        )

    try:
        news_sentiment = analyse_news_sentiment(ticker)
    except Exception as e:
        news_sentiment = create_error_result(
            f"Error in news sentiment analysis: {str(e)}"
        )

    # Calculate total score
    total_score = (
        moat_analysis.get("score", 0)
        + management_analysis.get("score", 0)
        + predictability_analysis.get("score", 0)
        + valuation_analysis.get("score", 0)
        + news_sentiment.get("score", 0)
    )

    max_possible_score = (
        moat_analysis.get("normalized_max_score", 10)
        + management_analysis.get("normalized_max_score", 10)
        + predictability_analysis.get("normalized_max_score", 10)
        + valuation_analysis.get("normalized_max_score", 10)
        + news_sentiment.get("normalized_max_score", 10)
    )

    # Generate signal based on score
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    munger_analysis = {
        "ticker": ticker,
        "signal": signal,
        "score": round(total_score, 2),
        "max_score": round(max_possible_score, 2),
        "moat_analysis": moat_analysis,
        "management_analysis": management_analysis,
        "predictability_analysis": predictability_analysis,
        "valuation_analysis": valuation_analysis,
        "news_sentiment": news_sentiment,
    }

    return munger_analysis


if __name__ == "__main__":
    ticker = "HMC"
    print(analyse_charlie_munger_valuation(ticker))
