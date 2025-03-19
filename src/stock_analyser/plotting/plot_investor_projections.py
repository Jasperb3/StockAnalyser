import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional
import os

from stock_analyser.tools.tool_utils.bill_ackman_analysis import (
    get_ticker, safe_get_row, filter_valid_values
)


def get_buffett_data(ticker: str) -> Tuple[float, float, float, float, float]:
    """
    Get data needed for Warren Buffett's intrinsic value calculation.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: (market_cap, net_income, depreciation, capex, shares_outstanding)
    """
    company_ticker = get_ticker(ticker)
    
    # Get market cap
    market_cap = company_ticker.info.get('marketCap', None)
    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        raise ValueError(f"Market cap data not available or invalid for {ticker}")
    
    # Get financial data
    financials = company_ticker.financials
    if financials is None or financials.empty:
        raise ValueError(f"No financial data available for {ticker}")
    
    # Get cash flow data
    cash_flow = company_ticker.cashflow
    if cash_flow is None or cash_flow.empty:
        raise ValueError(f"No cash flow data available for {ticker}")
    
    # Get balance sheet data
    balance_sheet = company_ticker.balance_sheet
    if balance_sheet is None or balance_sheet.empty:
        raise ValueError(f"No balance sheet data available for {ticker}")
    
    # Get net income
    net_income_df = safe_get_row(financials, "Net Income", 
                                ["Net Income Common Stockholders", "Net Profit"])
    if net_income_df is None:
        raise ValueError(f"Net Income data not available for {ticker}")
    
    net_income_vals = filter_valid_values(net_income_df)
    if not net_income_vals:
        raise ValueError(f"No valid net income values available for {ticker}")
    
    net_income = net_income_vals[0]  # Most recent value
    
    # Get depreciation
    depreciation_df = safe_get_row(cash_flow, "Depreciation And Amortization", 
                                  ["Depreciation", "Depreciation & Amortization"])
    if depreciation_df is None:
        raise ValueError(f"Depreciation data not available for {ticker}")
    
    depreciation_vals = filter_valid_values(depreciation_df)
    if not depreciation_vals:
        raise ValueError(f"No valid depreciation values available for {ticker}")
    
    depreciation = depreciation_vals[0]  # Most recent value
    
    # Get capex
    capex_df = safe_get_row(cash_flow, "Capital Expenditure", 
                           ["Capital Expenditures", "Purchase Of Plant Property And Equipment"])
    if capex_df is None:
        raise ValueError(f"Capital expenditure data not available for {ticker}")
    
    capex_vals = filter_valid_values(capex_df)
    if not capex_vals:
        raise ValueError(f"No valid capital expenditure values available for {ticker}")
    
    capex = abs(capex_vals[0])  # Most recent value, ensure positive
    
    # Get shares outstanding
    shares_df = safe_get_row(balance_sheet, "Share Issued", 
                            ["Common Stock Shares Outstanding", "Ordinary Shares Number"])
    if shares_df is None:
        raise ValueError(f"Outstanding shares data not available for {ticker}")
    
    shares_vals = filter_valid_values(shares_df)
    if not shares_vals:
        raise ValueError(f"No valid outstanding shares values available for {ticker}")
    
    shares_outstanding = shares_vals[0]  # Most recent value
    
    return market_cap, net_income, depreciation, capex, shares_outstanding


def get_ackman_data(ticker: str) -> Tuple[float, float]:
    """
    Get data needed for Bill Ackman's intrinsic value calculation.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: (market_cap, fcf)
    """
    company_ticker = get_ticker(ticker)
    
    # Get market cap
    market_cap = company_ticker.info.get('marketCap', None)
    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        raise ValueError(f"Market cap data not available or invalid for {ticker}")
    
    # Get cash flow data
    cash_flow = company_ticker.cashflow
    if cash_flow is None or cash_flow.empty:
        raise ValueError(f"No cash flow data available for {ticker}")
    
    # Get FCF
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    if fcf_df is None:
        raise ValueError(f"Free cash flow data not available for {ticker}")
    
    fcf_vals = filter_valid_values(fcf_df)
    if not fcf_vals:
        raise ValueError(f"No valid free cash flow values available for {ticker}")
    
    fcf = fcf_vals[0]  # Most recent value
    
    if fcf <= 0:
        raise ValueError(f"Most recent FCF is negative or zero (${fcf:,.0f}), cannot perform valuation")
    
    return market_cap, fcf


def calculate_buffett_intrinsic_value_by_year(
    net_income: float,
    depreciation: float,
    capex: float,
    shares_outstanding: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.09,
    terminal_multiple: float = 12,
    projection_years: int = 10
) -> Dict[int, Dict[str, float]]:
    """
    Calculate Warren Buffett's intrinsic value and margin of safety for each projected year.
    
    Args:
        net_income (float): Net income
        depreciation (float): Depreciation and amortization
        capex (float): Capital expenditures
        shares_outstanding (float): Number of outstanding shares
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Dictionary with year as key and intrinsic value data as value
    """
    # Calculate owner earnings (Buffett's preferred cash flow metric)
    maintenance_capex = capex * 0.75  # Assuming 75% of CapEx is for maintenance
    owner_earnings = net_income + depreciation - maintenance_capex
    
    if owner_earnings <= 0:
        raise ValueError(f"Calculated owner earnings are negative or zero: ${owner_earnings:,.2f}")
    
    results = {}
    
    for year in range(1, projection_years + 1):
        # Calculate present value of projected owner earnings up to this year
        present_value = 0
        for y in range(1, year + 1):
            future_earnings = owner_earnings * (1 + growth_rate) ** y
            pv = future_earnings / (1 + discount_rate) ** y
            present_value += pv
        
        # Calculate terminal value at the end of this year
        terminal_value = (owner_earnings * (1 + growth_rate) ** year * terminal_multiple) \
                        / (1 + discount_rate) ** year
        
        # Total intrinsic value
        intrinsic_value = present_value + terminal_value
        
        # Intrinsic value per share
        intrinsic_value_per_share = intrinsic_value / shares_outstanding
        
        results[year] = {
            "intrinsic_value": intrinsic_value,
            "intrinsic_value_per_share": intrinsic_value_per_share,
            "owner_earnings": owner_earnings * (1 + growth_rate) ** year,
            "owner_earnings_per_share": (owner_earnings * (1 + growth_rate) ** year) / shares_outstanding
        }
    
    return results


def calculate_ackman_intrinsic_value_by_year(
    fcf: float,
    growth_rate: float = 0.06,
    discount_rate: float = 0.10,
    terminal_multiple: float = 15,
    projection_years: int = 5
) -> Dict[int, Dict[str, float]]:
    """
    Calculate Bill Ackman's intrinsic value and margin of safety for each projected year.
    
    Args:
        fcf (float): Free cash flow
        growth_rate (float): Annual growth rate for projections
        discount_rate (float): Discount rate for present value calculations
        terminal_multiple (float): Multiple for terminal value calculation
        projection_years (int): Number of years to project cash flows
        
    Returns:
        dict: Dictionary with year as key and intrinsic value data as value
    """
    results = {}
    
    for year in range(1, projection_years + 1):
        # Calculate present value of projected FCF up to this year
        present_value = 0
        for y in range(1, year + 1):
            future_fcf = fcf * (1 + growth_rate) ** y
            pv = future_fcf / (1 + discount_rate) ** y
            present_value += pv
        
        # Calculate terminal value at the end of this year
        terminal_value = (fcf * (1 + growth_rate) ** year * terminal_multiple) \
                        / (1 + discount_rate) ** year
        
        # Total intrinsic value
        intrinsic_value = present_value + terminal_value
        
        results[year] = {
            "intrinsic_value": intrinsic_value,
            "fcf": fcf * (1 + growth_rate) ** year
        }
    
    return results


def calculate_margin_of_safety(intrinsic_value: float, market_cap: float) -> float:
    """
    Calculate margin of safety.
    
    Args:
        intrinsic_value (float): Intrinsic value
        market_cap (float): Market capitalization
        
    Returns:
        float: Margin of safety as a decimal
    """
    return (intrinsic_value - market_cap) / intrinsic_value


def plot_intrinsic_value_projection(
    ticker: str,
    buffett_growth_rate: float = 0.05,
    buffett_discount_rate: float = 0.09,
    buffett_terminal_multiple: float = 12,
    buffett_projection_years: int = 10,
    ackman_growth_rate: float = 0.06,
    ackman_discount_rate: float = 0.10,
    ackman_terminal_multiple: float = 15,
    ackman_projection_years: int = 5,
    save_plot: bool = False,
    output_dir: str = None,
    timestamp: str = None
) -> plt.Figure:
    """
    Plot projected intrinsic value and margin of safety for both Buffett and Ackman methodologies.
    
    Args:
        ticker (str): Stock ticker symbol
        buffett_growth_rate (float): Growth rate for Buffett's method
        buffett_discount_rate (float): Discount rate for Buffett's method
        buffett_terminal_multiple (float): Terminal multiple for Buffett's method
        buffett_projection_years (int): Projection years for Buffett's method
        ackman_growth_rate (float): Growth rate for Ackman's method
        ackman_discount_rate (float): Discount rate for Ackman's method
        ackman_terminal_multiple (float): Terminal multiple for Ackman's method
        ackman_projection_years (int): Projection years for Ackman's method
        save_plot (bool): Whether to save the plot
        output_dir (str): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get data for both methodologies
    try:
        market_cap, net_income, depreciation, capex, shares_outstanding = get_buffett_data(ticker)
        _, fcf = get_ackman_data(ticker)
    except Exception as e:
        raise ValueError(f"Error retrieving data for {ticker}: {str(e)}")
    
    # Calculate intrinsic values for each year
    try:
        buffett_values = calculate_buffett_intrinsic_value_by_year(
            net_income, depreciation, capex, shares_outstanding,
            buffett_growth_rate, buffett_discount_rate,
            buffett_terminal_multiple, buffett_projection_years
        )
        
        ackman_values = calculate_ackman_intrinsic_value_by_year(
            fcf, ackman_growth_rate, ackman_discount_rate,
            ackman_terminal_multiple, ackman_projection_years
        )
    except Exception as e:
        raise ValueError(f"Error calculating intrinsic values: {str(e)}")
    
    # Calculate margin of safety for each year
    buffett_mos = {year: calculate_margin_of_safety(data["intrinsic_value"], market_cap) 
                  for year, data in buffett_values.items()}
    
    ackman_mos = {year: calculate_margin_of_safety(data["intrinsic_value"], market_cap) 
                 for year, data in ackman_values.items()}
    
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Create a second y-axis for margin of safety
    ax2 = ax1.twinx()
    
    # Plot intrinsic values
    buffett_years = list(buffett_values.keys())
    ackman_years = list(ackman_values.keys())
    
    buffett_iv = [data["intrinsic_value"] for data in buffett_values.values()]
    ackman_iv = [data["intrinsic_value"] for data in ackman_values.values()]
    
    buffett_mos_values = [buffett_mos[year] * 100 for year in buffett_years]
    ackman_mos_values = [ackman_mos[year] * 100 for year in ackman_years]
    
    # Plot intrinsic values on the first y-axis
    ax1.plot(buffett_years, buffett_iv, 'b-', linewidth=2, label="Buffett Intrinsic Value")
    ax1.plot(ackman_years, ackman_iv, 'g-', linewidth=2, label="Ackman Intrinsic Value")
    
    # Plot horizontal line for market cap
    ax1.axhline(y=market_cap, color='r', linestyle='-', linewidth=2, label=f"Market Cap")
    
    # Plot margin of safety on the second y-axis
    ax2.plot(buffett_years, buffett_mos_values, 'b--', linewidth=1.5, label="Buffett Margin of Safety")
    ax2.plot(ackman_years, ackman_mos_values, 'g--', linewidth=1.5, label="Ackman Margin of Safety")
    
    # Add a horizontal line at 0% margin of safety
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add a horizontal line at 30% margin of safety (significant)
    ax2.axhline(y=30, color='g', linestyle='--', linewidth=1, alpha=0.5, label="30% Margin of Safety")
    
    # Set labels and title
    ax1.set_xlabel("Projection Year", fontsize=12)
    ax1.set_ylabel("Intrinsic Value ($)", fontsize=12)
    ax2.set_ylabel("Margin of Safety (%)", fontsize=12)
    
    plt.title(f"Projected Intrinsic Value and Margin of Safety for {ticker.upper()}", fontsize=16)
    
    # Format y-axis labels with commas for thousands and Billions
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e9:,.0f}B"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    
    # Set x-axis to show integer years
    ax1.set_xticks(range(1, max(max(buffett_years), max(ackman_years)) + 1))
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Move the legend up
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Add annotations for key values
    # Final intrinsic values
    final_buffett_iv = buffett_iv[-1]
    final_ackman_iv = ackman_iv[-1]
    
    # Final margins of safety
    final_buffett_mos = buffett_mos_values[-1]
    final_ackman_mos = ackman_mos_values[-1]
    
    # Add text box with key metrics
    textstr = '\n'.join((
        f"Buffett Final IV: ${final_buffett_iv/1e9:,.0f}B",
        f"Buffett Final MoS: {final_buffett_mos:.1f}%",
        f"Ackman Final IV: ${final_ackman_iv/1e9:,.0f}B",
        f"Ackman Final MoS: {final_ackman_mos:.1f}%",
        f"Market Cap: ${market_cap/1e9:,.0f}B"
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.85, textstr, transform=ax1.transAxes, fontsize=10, # Moved up
            verticalalignment='top', bbox=props)
    
    # Add assumptions text box
    assumptions = '\n'.join((
        f"Buffett: g={buffett_growth_rate:.1%}, r={buffett_discount_rate:.1%}, m={buffett_terminal_multiple}x",
        f"Ackman: g={ackman_growth_rate:.1%}, r={ackman_discount_rate:.1%}, m={ackman_terminal_multiple}x"
    ))
    # Move assumptions box up
    ax1.text(0.05, 0.20, assumptions, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_path = f'{output_dir}/{ticker.upper()}_projection_{timestamp}.png'
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    ticker = "AMZN"
    
    # Default parameters
    buffett_growth_rate = 0.05
    buffett_discount_rate = 0.09
    buffett_terminal_multiple = 12
    buffett_projection_years = 10
    
    ackman_growth_rate = 0.06
    ackman_discount_rate = 0.10
    ackman_terminal_multiple = 15
    ackman_projection_years = 5
    
    fig = plot_intrinsic_value_projection(
        ticker,
        buffett_growth_rate, buffett_discount_rate, buffett_terminal_multiple, buffett_projection_years,
        ackman_growth_rate, ackman_discount_rate, ackman_terminal_multiple, ackman_projection_years,
        save_plot=True, output_dir="plots/intrinsic_value", timestamp="2021-09-01"
    )