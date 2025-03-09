from stock_analyser.tools.tool_utils.metrics import compute_financial_metrics
import yfinance as yf
import numpy as np
import pandas as pd

def get_ticker(ticker: str):
    return yf.Ticker(ticker.upper())


def calculate_cathie_wood_analysis_data(ticker: str, growth_rate: float = 0.20, discount_rate: float = 0.15, terminal_multiple: float = 25, projection_years: int = 5):
    """
    analyses stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
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
        disruptive_analysis = analyse_disruptive_potential(metrics, ticker)
    except Exception as e:
        disruptive_analysis = {
            "score": 0, 
            "details": f"Error in disruptive potential analysis: {str(e)}"
        }

    try:
        innovation_analysis = analyse_innovation_growth(metrics, ticker)
    except Exception as e:
        innovation_analysis = {
            "score": 0, 
            "details": f"Error in innovation growth analysis: {str(e)}"
        }

    try:
        valuation_analysis = analyse_cathie_wood_valuation(metrics, ticker, growth_rate, discount_rate, terminal_multiple, projection_years)
    except Exception as e:
        valuation_analysis = {
            "score": 0, 
            "details": f"Error in valuation analysis: {str(e)}"
        }
    
    # Combine partial scores or signals
    total_score = disruptive_analysis.get("score", 0) + innovation_analysis.get("score", 0) + valuation_analysis.get("score", 0)
    max_possible_score = 15  # Adjust weighting as desired

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
        "disruptive_analysis": disruptive_analysis,
        "innovation_analysis": innovation_analysis,
        "valuation_analysis": valuation_analysis
    }
    
    return analysis_data


def analyse_disruptive_potential(metrics: dict, ticker: str):
    """
    analyse whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """

    company_ticker = get_ticker(ticker)

    try:
        financials = company_ticker.financials
        if financials.empty:
            return {"score": 0, "details": "No financial data available for analysis"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving financial data: {str(e)}"}

    score = 0
    details = []

    if not metrics:
        return {"score": 0, "details": "Insufficient data to analyse disruptive potential"}
    
    # 1. Revenue Growth Analysis - Check for accelerating growth
    try:
        revenue_df = financials.loc["Total Revenue"]
        if revenue_df.empty:
            details.append("Insufficient revenue data for growth analysis")
    except (KeyError, Exception) as e:
        details.append(f"Revenue data not available: {str(e)}")
        revenue_df = pd.Series()

    revenues = [revenue for revenue in revenue_df if revenue is not None and not np.isnan(revenue)]
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        growth_rates = []
        for i in range(len(revenues)-1):
            if revenues[i] and revenues[i+1] and revenues[i] != 0 and not np.isnan(revenues[i]) and not np.isnan(revenues[i+1]):
                growth_rate = (revenues[i+1] - revenues[i]) / abs(revenues[i])
                growth_rates.append(growth_rate)
        
        # Check if growth is accelerating
        if len(growth_rates) >= 2:
            if growth_rates[0] > growth_rates[-1]:
                score += 2
                details.append(f"Revenue growth is accelerating: {(growth_rates[0]*100):.1f}% vs {(growth_rates[-1]*100):.1f}%")
            
            # Check absolute growth rate
            latest_growth = growth_rates[0] if growth_rates else 0
            if latest_growth > 1.0:
                score += 3
                details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
            elif latest_growth > 0.5:
                score += 2
                details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
            elif latest_growth > 0.2:
                score += 1
                details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
        else:
            details.append("Insufficient growth rate data for analysis")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis - Check for expanding margins
    try:
        cost_of_revenue_df = financials.loc["Cost Of Revenue"]
        if cost_of_revenue_df.empty:
            details.append("Insufficient cost of revenue data for gross margin analysis")
    except (KeyError, Exception) as e:
        details.append(f"Cost of revenue data not available: {str(e)}")
        cost_of_revenue_df = pd.Series()
    
    revenues = [revenue for revenue in revenue_df if revenue is not None and not np.isnan(revenue)]
    cost_of_revenues = [cost for cost in cost_of_revenue_df if cost is not None and not np.isnan(cost)]

    gross_margins = []

    if len(revenues) == len(cost_of_revenues) and len(revenues) > 0:
        for i in range(len(revenues)):
            if revenues[i] != 0 and not np.isnan(revenues[i]) and not np.isnan(cost_of_revenues[i]):
                gross_margin = (revenues[i] - cost_of_revenues[i]) / revenues[i]
                gross_margins.append(gross_margin)

    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level
        if gross_margins[0] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margins[0]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    try:
        operating_expenses_df = financials.loc["Operating Expense"]
        if operating_expenses_df.empty:
            details.append("Insufficient operating expense data for operating leverage analysis")
    except (KeyError, Exception) as e:
        details.append(f"Operating expense data not available: {str(e)}")
        operating_expenses_df = pd.Series()

    revenues = [revenue for revenue in revenue_df if revenue is not None and not np.isnan(revenue)]
    operating_expenses = [operating_expense for operating_expense in operating_expenses_df if operating_expense is not None and not np.isnan(operating_expense)]
    
    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        try:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])
            
            if rev_growth > opex_growth:
                score += 2
                details.append("Positive operating leverage: Revenue growing faster than expenses")
        except (ZeroDivisionError, Exception) as e:
            details.append(f"Error calculating operating leverage: {str(e)}")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    try:
        rd_expenses_df = financials.loc["Research And Development"]
        if rd_expenses_df.empty:
            details.append("Insufficient research and development expense data for R&D investment analysis")
    except (KeyError, Exception) as e:
        details.append(f"R&D expense data not available: {str(e)}")
        rd_expenses_df = pd.Series()

    rd_expenses = [research_and_development for research_and_development in rd_expenses_df if research_and_development is not None and not np.isnan(research_and_development)]
    
    if rd_expenses and revenues:
        try:
            rd_intensity = rd_expenses[0] / revenues[0]
            if rd_intensity > 0.15:  # High R&D intensity
                score += 3
                details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
            elif rd_intensity > 0.08:
                score += 2
                details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
            elif rd_intensity > 0.05:
                score += 1
                details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
        except (ZeroDivisionError, IndexError, Exception) as e:
            details.append(f"Error calculating R&D intensity: {str(e)}")
    else:
        details.append("No R&D data available")

    # Normalize score to be out of 5
    max_possible_score = 12  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "max_score": max_possible_score
    }


def analyse_innovation_growth(metrics: dict, ticker: str):
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    analyses multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """

    company_ticker = get_ticker(ticker)

    try:
        financials = company_ticker.financials
        cashflow = company_ticker.cashflow
        
        if financials.empty:
            return {"score": 0, "details": "No financial data available for analysis"}
        if cashflow.empty:
            return {"score": 0, "details": "No cash flow data available for analysis"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving financial data: {str(e)}"}

    score = 0
    details = []

    if not metrics:
        return {
            "score": 0,
            "details": "Insufficient data to analyse innovation-driven growth"
        }

    # 1. R&D Investment Trends
    try:
        rd_expenses_df = financials.loc["Research And Development"]
        revenue_df = financials.loc["Total Revenue"]
    except (KeyError, Exception) as e:
        details.append(f"R&D or revenue data not available: {str(e)}")
        rd_expenses_df = pd.Series()
        revenue_df = pd.Series()

    if rd_expenses_df.empty or revenue_df.empty:
        details.append("Insufficient research and development expense data for R&D investment analysis")

    rd_expenses = [research_and_development for research_and_development in rd_expenses_df if research_and_development is not None and not np.isnan(research_and_development)]
    revenues = [revenue for revenue in revenue_df if revenue is not None and not np.isnan(revenue)]
    
    if rd_expenses and revenues and len(rd_expenses) >= 2:
        try:
            # Check R&D growth rate
            rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1])
            if rd_growth > 0.5:  # 50% growth in R&D
                score += 3
                details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
            elif rd_growth > 0.2:
                score += 2
                details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")
            
            # Check R&D intensity trend
            if revenues[-1] != 0 and revenues[0] != 0:
                rd_intensity_start = rd_expenses[-1] / revenues[-1]
                rd_intensity_end = rd_expenses[0] / revenues[0]
                if rd_intensity_end > rd_intensity_start:
                    score += 2
                    details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
            else:
                details.append("Cannot calculate R&D intensity due to zero revenue")
        except (ZeroDivisionError, IndexError, Exception) as e:
            details.append(f"Error calculating R&D trends: {str(e)}")
    else:
        details.append("Insufficient R&D data for trend analysis")

    # 2. Free Cash Flow Analysis
    try:
        fcf_df = cashflow.loc["Free Cash Flow"]
    except (KeyError, Exception) as e:
        details.append(f"Free cash flow data not available: {str(e)}")
        fcf_df = pd.Series()
        
    if fcf_df.empty:
        details.append("Insufficient free cash flow data for analysis")

    fcf_vals = [free_cash_flow for free_cash_flow in fcf_df if free_cash_flow is not None and not np.isnan(free_cash_flow)]
    if fcf_vals and len(fcf_vals) >= 2:
        try:
            # Check FCF growth and consistency
            if fcf_vals[-1] != 0:
                fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
            else:
                fcf_growth = 0
                details.append("Cannot calculate FCF growth due to zero base value")
                
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
            
            if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
                score += 3
                details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
            elif positive_fcf_count >= len(fcf_vals) * 0.75:
                score += 2
                details.append("Consistent positive FCF, good innovation funding capacity")
            elif positive_fcf_count > len(fcf_vals) * 0.5:
                score += 1
                details.append("Moderately consistent FCF, adequate innovation funding capacity")
        except (ZeroDivisionError, IndexError, Exception) as e:
            details.append(f"Error calculating FCF trends: {str(e)}")
    else:
        details.append("Insufficient FCF data for analysis")

    # 3. Operating Efficiency Analysis
    # Calculate Operating Margin:  Operating Income / Total Revenue
    try:
        revenue_df = financials.loc["Total Revenue"]
        operating_income_df = financials.loc["Operating Income"]
    except (KeyError, Exception) as e:
        details.append(f"Revenue or operating income data not available: {str(e)}")
        revenue_df = pd.Series()
        operating_income_df = pd.Series()

    if revenue_df.empty or operating_income_df.empty:
        details.append("Insufficient revenue or operating income data for operating efficiency analysis")
    
    revenues = [revenue for revenue in revenue_df if revenue is not None and not np.isnan(revenue)]
    operating_incomes = [op_income for op_income in operating_income_df if op_income is not None and not np.isnan(op_income)]

    op_margin_vals = []

    if len(revenues) == len(operating_incomes) and len(revenues) > 0:
        for i in range(len(revenues)):
            if revenues[i] != 0 and not np.isnan(revenues[i]) and not np.isnan(operating_incomes[i]):
                op_margin = operating_incomes[i] / revenues[i]
                op_margin_vals.append(op_margin)

    if op_margin_vals and len(op_margin_vals) >= 2:
        # Check margin improvement
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]
        
        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    # 4. Capital Allocation Analysis
    try:
        capex_df = cashflow.loc["Capital Expenditure"]
    except (KeyError, Exception) as e:
        details.append(f"Capital expenditure data not available: {str(e)}")
        capex_df = pd.Series()
        
    if capex_df.empty:
        details.append("Insufficient capital expenditure data for analysis")

    capex = [capital_expenditure for capital_expenditure in capex_df if capital_expenditure is not None and not np.isnan(capital_expenditure)]
    if capex and revenues and len(capex) >= 2:
        try:
            capex_intensity = abs(capex[0]) / revenues[0]
            if capex[-1] != 0:
                capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1])
            else:
                capex_growth = 0
                details.append("Cannot calculate CAPEX growth due to zero base value")
            
            if capex_intensity > 0.10 and capex_growth > 0.2:
                score += 2
                details.append("Strong investment in growth infrastructure")
            elif capex_intensity > 0.05:
                score += 1
                details.append("Moderate investment in growth infrastructure")
        except (ZeroDivisionError, IndexError, Exception) as e:
            details.append(f"Error calculating capital allocation metrics: {str(e)}")
    else:
        details.append("Insufficient CAPEX data")

    # 5. Growth Reinvestment Analysis
    try:
        dividends_df = cashflow.loc["Cash Dividends Paid"]
    except (KeyError, Exception) as e:
        details.append(f"Dividend data not available: {str(e)}")
        dividends_df = pd.Series()

    if dividends_df.empty:
        details.append("Insufficient dividend data for analysis")

    dividends = [dividend for dividend in dividends_df if dividend is not None and not np.isnan(dividend)]

    if dividends and fcf_vals:
        try:
            # Check if company prioritizes reinvestment over dividends
            if fcf_vals[0] != 0 and not np.isnan(dividends[0]) and not np.isnan(fcf_vals[0]):
                latest_payout_ratio = dividends[0] / fcf_vals[0]
                if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
                    score += 2
                    details.append("Strong focus on reinvestment over dividends")
                elif latest_payout_ratio < 0.4:
                    score += 1
                    details.append("Moderate focus on reinvestment over dividends")
            else:
                details.append("Cannot calculate payout ratio due to zero FCF")
        except (ZeroDivisionError, IndexError, Exception) as e:
            details.append(f"Error calculating reinvestment metrics: {str(e)}")
    else:
        details.append("Insufficient dividend data")

    # Normalize score to be out of 5
    max_possible_score = 15  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "max_score": max_possible_score
    }


def analyse_cathie_wood_valuation(metrics: dict, ticker: str, growth_rate: float = 0.20, discount_rate: float = 0.15, terminal_multiple: float = 25, projection_years: int = 5):
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.

    company_ticker = get_ticker(ticker)

    try:
        cashflow = company_ticker.cashflow
        if cashflow.empty:
            return {"score": 0, "details": "No cash flow data available for valuation"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}

    market_cap = metrics.get("market_cap")
    if not metrics or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data for valuation"
        }

    try:
        latest = cashflow.loc["Free Cash Flow"].iloc[0]
        fcf = latest if latest and not np.isnan(latest) else 0
    except (KeyError, IndexError, Exception) as e:
        return {
            "score": 0,
            "details": f"Error retrieving FCF data: {str(e)}",
            "intrinsic_value": None
        }

    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }

    try:
        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = future_fcf / ((1 + discount_rate) ** year)
            present_value += pv

        # Terminal Value
        terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                        / ((1 + discount_rate) ** projection_years)
        
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
            f"Margin of safety: {margin_of_safety:.2%}"
        ]

        return {
            "score": score,
            "details": "; ".join(details),
            "intrinsic_value": intrinsic_value,
            "margin_of_safety": margin_of_safety
        }
    except Exception as e:
        return {
            "score": 0,
            "details": f"Error in valuation calculation: {str(e)}",
            "intrinsic_value": None
        }


if __name__ == "__main__":
    TICKER = "NVDA"
    cathie_wood_analysis_data = calculate_cathie_wood_analysis_data(TICKER)
    print(cathie_wood_analysis_data)    
