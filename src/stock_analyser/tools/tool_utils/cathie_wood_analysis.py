import yfinance as yf
from pprint import pprint
import numpy as np
import pandas as pd
from stock_analyser.utils.convert_currency import convert_currency


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


def calculate_cathie_wood_analysis_data(
    ticker: str,
    growth_rate: float = 0.20,
    discount_rate: float = 0.15,
    terminal_multiple: float = 25,
    projection_years: int = 5,
):
    """
    Analyzes stocks using Cathie Wood's innovation-focused investing principles.

    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections (default 20%)
        discount_rate (float): Discount rate for present value calculations (default 15%)
        terminal_multiple (float): Multiple for terminal value calculation (default 25x)
        projection_years (int): Number of years to project cash flows (default 5)

    Returns:
        dict: Complete analysis with signal, score, and detailed components
    """

    try:
        disruptive_analysis = analyse_disruptive_potential(ticker)
    except Exception as e:
        disruptive_analysis = {
            "score": 0,
            "max_score": 0,
            "details": f"Error in disruptive potential analysis: {str(e)}",
        }

    try:
        innovation_analysis = analyse_innovation_growth(ticker)
    except Exception as e:
        innovation_analysis = {
            "score": 0,
            "max_score": 0,
            "details": f"Error in innovation growth analysis: {str(e)}",
        }

    try:
        valuation_analysis = analyse_cathie_wood_valuation(
            ticker, growth_rate, discount_rate, terminal_multiple, projection_years
        )
    except Exception as e:
        valuation_analysis = {
            "score": 0,
            "max_score": 0,
            "details": f"Error in valuation analysis: {str(e)}",
        }

    # Combine partial scores and max_scores
    total_score = (
        disruptive_analysis.get("score", 0)
        + innovation_analysis.get("score", 0)
        + valuation_analysis.get("score", 0)
    )
    max_possible_score = (
        disruptive_analysis.get("normalized_max_score", 0)
        + innovation_analysis.get("normalized_max_score", 0)
        + valuation_analysis.get("normalized_max_score", 0)
    )

    # Generate signal based on score
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    analysis_data = {
        "signal": signal,
        "score": round(total_score, 2),
        "max_score": round(max_possible_score, 2),
        "disruptive_analysis": disruptive_analysis,
        "innovation_analysis": innovation_analysis,
        "valuation_analysis": valuation_analysis,
    }

    return analysis_data


def analyse_disruptive_potential(ticker: str):
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Sector analysis - identifies if the company is in a disruptive sector
    2. Revenue Growth Acceleration - indicates market adoption
    3. R&D Intensity - shows innovation investment
    4. Gross Margin Trends - suggests pricing power and scalability
    5. Operating Leverage - demonstrates business model efficiency
    6. Market Share Dynamics - indicates competitive position

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0
    details = []

    # Get company info and sector
    try:
        company_ticker = get_ticker(ticker)
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
            "details": f"Error retrieving cash flow data: {str(e)}",
        }

    try:
        income_statement = company_ticker.income_stmt
        if income_statement is None or income_statement.empty:
            return {
                "score": 0,
                "max_score": 0,
                "details": "No income statement data available",
            }
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Error retrieving income statement data: {str(e)}",
        }

    # Get company info and sector
    info = company_ticker.info
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")

    # 1. Check if company is in a disruptive/innovative sector
    disruptive_sectors = [
        "technology",
        "healthcare",
        "communication-services",
        "consumer-cyclical",
        "energy",
        "industrials",
    ]

    disruptive_industries = [
        # Genomics
        "biotechnology",
        "diagnostics-research",
        "health-information-services",
        "medical-devices",
        # Robotics
        "semiconductors",
        "software-application",
        "software-infrastructure",
        "electronic-components",
        "information-technology-services",
        "communication-equipment",
        "computer-hardware",
        "aerospace-defense",
        "specialty-industrial-machinery",
        # Energy Storage
        "solar",
        "utilities-renewable",
        "oil-gas-e-p",
        "electronic-components",
        # Artificial Intelligence (AI)
        "semiconductors",
        "software-application",
        "software-infrastructure",
        "information-technology-services",
        # Blockchain Technology
        "software-application",
        "software-infrastructure",
        "information-technology-services",
        "financial-data-stock-exchanges",
        # Space Exploration
        "aerospace-defense",
        # Next Generation Internet (Web3, Metaverse)
        "internet-content-information",
        "software-application",
        "software-infrastructure",
        "electronic-gaming-multimedia",
        "communication-equipment",
    ]

    if any(
        disruptive_term.lower() in sector.lower()
        for disruptive_term in disruptive_sectors
    ):
        score += 2
        details.append(f"Company operates in disruptive sector: {sector}")
    elif any(
        disruptive_term.lower() in industry.lower()
        for disruptive_term in disruptive_industries
    ):
        score += 2
        details.append(f"Company operates in innovative industry: {industry}")
    else:
        details.append(
            f"Company sector ({sector}) and industry ({industry}) not identified as highly disruptive"
        )

    max_score += 2

    # 2. Revenue Growth Analysis - Check for accelerating growth

    revenue = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    if revenue is not None and not revenue.empty:
        revenue_values = filter_valid_values(revenue)
        if len(revenue_values) >= 3:
            growth_rates = []
            for i in range(len(revenue_values) - 1):
                if revenue_values[i] and revenue_values[i + 1]:
                    growth_rate = (
                        (revenue_values[i] - revenue_values[i + 1])
                        / abs(revenue_values[i + 1])
                        if revenue_values[i + 1] != 0
                        else 0
                    )
                    growth_rates.append(growth_rate)

            if len(growth_rates) >= 2:
                latest_growth = growth_rates[0]
                earliest_growth = growth_rates[-1]

                if latest_growth > earliest_growth:
                    score += 2
                    details.append(
                        f"Revenue growth is accelerating: {(latest_growth * 100):.1f}% vs {(earliest_growth * 100):.1f}%"
                    )

                # Check absolute growth rate
                if latest_growth > 1.0:
                    score += 3
                    details.append(
                        f"Exceptional revenue growth: {(latest_growth * 100):.1f}%"
                    )
                elif latest_growth > 0.5:
                    score += 2
                    details.append(
                        f"Strong revenue growth: {(latest_growth * 100):.1f}%"
                    )
                elif latest_growth > 0.2:
                    score += 1
                    details.append(
                        f"Moderate revenue growth: {(latest_growth * 100):.1f}%"
                    )
            else:
                details.append("Insufficient revenue data for growth analysis")

        else:
            details.append("Insufficient revenue data for growth analysis")
    else:
        details.append("Insufficient revenue data for growth analysis")

    max_score += 5

    # 3. Gross Margin - High margins suggest pricing power and innovation
    try:
        gross_profit = safe_get_row(financials, "Gross Profit")
    except Exception as e:
        print(f"Error getting gross profit: {str(e)}")
        gross_profit = None

    try:
        total_revenue = safe_get_row(financials, "Total Revenue")
    except Exception as e:
        print(f"Error getting total revenue: {str(e)}")
        total_revenue = None

    if (
        gross_profit is not None
        and total_revenue is not None
        and not gross_profit.empty
        and not total_revenue.empty
    ):
        gross_margins = gross_profit / total_revenue
        gross_margin_values = [
            gm for gm in gross_margins if gm is not None and not np.isnan(gm)
        ]

        if len(gross_margin_values) >= 2:
            margin_trend = gross_margin_values[0] - gross_margin_values[-1]
            if margin_trend > 0.05:  # 5% improvement
                score += 2
            details.append(f"Expanding gross margins: +{(margin_trend * 100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(
                f"Slightly improving gross margins: +{(margin_trend * 100):.1f}%"
            )

        # Check absolute margin level
        if gross_margin_values[0] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margin_values[0] * 100):.1f}%")
        else:
            details.append("Insufficient gross margin data")
    else:
        details.append("Insufficient gross margin data")

    max_score += 4

    # 4. Operating Leverage Analysis

    operating_expense = safe_get_row(financials, "Operating Expense")
    if operating_expense is not None and not operating_expense.empty:
        operating_expense_values = filter_valid_values(operating_expense)

    if len(revenue_values) >= 2 and len(operating_expense_values) >= 2:
        rev_growth = (revenue_values[0] - revenue_values[-1]) / abs(revenue_values[-1])
        opex_growth = (
            operating_expense_values[0] - operating_expense_values[-1]
        ) / abs(operating_expense_values[-1])

        if rev_growth > opex_growth:
            score += 2
            details.append(
                "Positive operating leverage: Revenue growing faster than expenses"
            )
        else:
            details.append(
                "Negative operating leverage: Expenses growing faster than revenue"
            )
    else:
        details.append("Insufficient data for operating leverage analysis")

    max_score += 2

    # 5. R&D Investment Analysis

    research_and_development = safe_get_row(financials, "Research And Development")

    r_and_d_values = filter_valid_values(research_and_development)

    if r_and_d_values and revenue_values:
        rd_intensity = r_and_d_values[0] / revenue_values[0]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(
                f"High R&D investment: {(rd_intensity * 100):.1f}% of revenue"
            )
        elif rd_intensity > 0.08:
            score += 2
            details.append(
                f"Moderate R&D investment: {(rd_intensity * 100):.1f}% of revenue"
            )
        elif rd_intensity > 0.05:
            score += 1
            details.append(
                f"Some R&D investment: {(rd_intensity * 100):.1f}% of revenue"
            )
    else:
        details.append("No R&D data available")

    max_score += 3

    # Normalize score to be out of 5
    normalized_score = round((score / max_score) * 5, 2)

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "normalized_max_score": 5,
    }


def analyse_innovation_growth(ticker: str):
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0
    details = []

    try:
        company_ticker = get_ticker(ticker)
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
            "details": f"Error retrieving cash flow data: {str(e)}",
        }

    # 1. R&D Spending Growth and Intensity
    # Try to find R&D expenses in the financials
    research_and_development = safe_get_row(financials, "Research And Development")
    revenue = safe_get_row(financials, "Total Revenue")

    r_and_d_values = filter_valid_values(research_and_development)
    # print(f"r_and_d_values: {r_and_d_values}")
    revenue_values = filter_valid_values(revenue)
    # print(f"revenue_values: {revenue_values}")

    if r_and_d_values and revenue_values and len(r_and_d_values) >= 2:
        rd_growth = (
            (r_and_d_values[0] - r_and_d_values[-1]) / abs(r_and_d_values[-1])
            if r_and_d_values[-1] != 0
            else 0
        )
        # print(f"rd_growth: {rd_growth}")
        if rd_growth > 0.5:  # 50% growth in R&D
            score += 3
            details.append(f"Strong R&D investment growth: +{(rd_growth * 100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{(rd_growth * 100):.1f}%")

        # Check R&D intensity trend
        rd_intensity_start = (
            r_and_d_values[-1] / revenue_values[-1]
            if r_and_d_values
            and revenue_values
            and len(r_and_d_values) > 0
            and len(revenue_values) > 0
            and revenue_values[-1] != 0
            else 0
        )
        # print(f"rd_intensity_start: {rd_intensity_start}")
        rd_intensity_end = (
            r_and_d_values[0] / revenue_values[0]
            if r_and_d_values
            and revenue_values
            and len(r_and_d_values) > 0
            and len(revenue_values) > 0
            and revenue_values[0] != 0
            else 0
        )
        # print(f"rd_intensity_end: {rd_intensity_end}")

        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(
                f"Increasing R&D intensity: {(rd_intensity_end * 100):.1f}% vs {(rd_intensity_start * 100):.1f}%"
            )
    else:
        details.append("Insufficient R&D data for trend analysis")

    max_score += 3

    # 2. Free Cash Flow Analysis
    free_cash_flow = safe_get_row(cash_flow, "Free Cash Flow")
    fcf_values = filter_valid_values(free_cash_flow)
    # print(f"fcf_values: {fcf_values}")

    if fcf_values and len(fcf_values) >= 2:
        # Check FCF growth and consistency
        fcf_growth = (
            (fcf_values[0] - fcf_values[-1]) / abs(fcf_values[-1])
            if fcf_values[-1] != 0
            else 0
        )
        # print(f"fcf_growth: {fcf_growth}")
        positive_fcf_count = sum(1 for f in fcf_values if f > 0)
        # print(f"positive_fcf_count: {positive_fcf_count}")

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_values):
            score += 3
            details.append(
                "Strong and consistent FCF growth, excellent innovation funding capacity"
            )
        elif positive_fcf_count >= len(fcf_values) * 0.75:
            score += 2
            details.append("Consistent positive FCF, good innovation funding capacity")
        elif positive_fcf_count > len(fcf_values) * 0.5:
            score += 1
            details.append(
                "Moderately consistent FCF, adequate innovation funding capacity"
            )
    else:
        details.append("Insufficient FCF data for analysis")

    max_score += 3

    # 3. Operating Efficiency Analysis

    operating_income = safe_get_row(financials, "Operating Income")
    # print(f"operating_income: {operating_income}")
    total_revenue = safe_get_row(financials, "Total Revenue")
    # print(f"total_revenue: {total_revenue}")

    if not operating_income.empty and not total_revenue.empty:
        operating_margins = [
            operating_income.iloc[i] / total_revenue.iloc[i]
            for i in range(len(operating_income))
            if total_revenue.iloc[i] != 0
        ]
        # print(f"operating_margins: {operating_margins}")

        op_margins = filter_valid_values(operating_margins)
        # print(f"op_margins: {op_margins}")

        if op_margins and len(op_margins) >= 2:
            # Check margin improvement
            margin_trend = op_margins[0] - op_margins[-1]
            # print(f"margin_trend: {margin_trend}")

            if op_margins[0] > 0.15 and margin_trend > 0:
                score += 3
                details.append(
                    f"Strong and improving operating margin: {(op_margins[0] * 100):.1f}%"
                )
            elif op_margins[0] > 0.10:
                score += 2
                details.append(
                    f"Healthy operating margin: {(op_margins[0] * 100):.1f}%"
                )
            elif margin_trend > 0:
                score += 1
                details.append("Improving operating efficiency")
            else:
                details.append("Operating margin not improving")
        else:
            details.append("Insufficient operating margin data")
    else:
        details.append("Insufficient operating income or revenue data")

    max_score += 2

    # 4. Capital Allocation Analysis

    capital_expenditure = safe_get_row(cash_flow, "Capital Expenditure")
    capex = filter_valid_values(capital_expenditure)
    # print(f"capex: {capex}")

    if capex and len(capex) >= 2 and revenue_values and len(revenue_values) > 0:
        capex_intensity = (
            abs(capex[0]) / revenue_values[0] if revenue_values[0] != 0 else 0
        )
        # print(f"capex_intensity: {capex_intensity}")
        capex_growth = (
            (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0
        )
        # print(f"capex_growth: {capex_growth}")

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    max_score += 2

    # 5. Growth Reinvestment Analysis

    cash_dividends = safe_get_row(cash_flow, "Cash Dividends Paid")
    # print(f"cash_dividends: {cash_dividends}")
    dividends = filter_valid_values(cash_dividends)
    # print(f"dividends: {dividends}")

    if dividends and fcf_values and len(fcf_values) > 0:
        # Check if company prioritizes reinvestment over dividends
        latest_payout_ratio = dividends[0] / fcf_values[0] if fcf_values[0] != 0 else 1
        # print(f"latest_payout_ratio: {latest_payout_ratio}")
        if (
            latest_payout_ratio < 0.2
        ):  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    max_score += 2

    # Normalize score to be out of 5
    normalized_score = round((score / max_score) * 5, 2)

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "normalized_max_score": 5,
    }


def analyse_cathie_wood_valuation(
    ticker: str,
    growth_rate: float = 0.20,
    discount_rate: float = 0.15,
    terminal_multiple: float = 25,
    projection_years: int = 5,
):
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.

    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections (default 20%)
        discount_rate (float): Discount rate for present value calculations (default 15%)
        terminal_multiple (float): Multiple for terminal value calculation (default 25x)
        projection_years (int): Number of years to project cash flows (default 5)

    Returns:
        dict: Analysis results with score and details
    """

    try:
        company_ticker = get_ticker(ticker)
        exchange_rate = convert_currency(ticker)
    except Exception as e:
        return {
            "score": 0,
            "max_score": 0,
            "details": f"Failed to get ticker data: {str(e)}",
        }

    score = 0
    details = []

    # Get key metrics for valuation
    market_cap = company_ticker.info.get("marketCap", None)

    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        return {
            "score": 0,
            "max_score": 0,
            "details": "Market cap data not available or invalid",
        }

    free_cash_flow = safe_get_row(company_ticker.cashflow, "Free Cash Flow")
    fcf = filter_valid_values(free_cash_flow)[0] * exchange_rate

    if fcf <= 0 or np.isnan(fcf):
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = ${fcf:,.2f}",
            "intrinsic_value": "Not possible to accurately estimate with this methodology",
        }

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (
        fcf * (1 + growth_rate) ** projection_years * terminal_multiple
    ) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [
        f"Calculated intrinsic value: ~${intrinsic_value:,.2f}",
        f"Market cap: ~${market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}",
    ]

    return {"score": score, "normalized_max_score": 3, "details": "; ".join(details)}


if __name__ == "__main__":
    analysis = calculate_cathie_wood_analysis_data("MANU")
    pprint(analysis)
