import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
from stock_analyser.tools.tool_utils.value_calcs import (
    calculate_intrinsic_value_earnings,
    calculate_margin_of_safety_earnings,
    calculate_intrinsic_value_fcf,
    calculate_margin_of_safety_fcf
)


def get_ticker_data(ticker: str):
    """
    Get ticker data from yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: (ticker_obj, current_price, eps, net_income, depreciation, capex)
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        current_price = ticker_obj.info.get('currentPrice')
        
        if current_price is None:
            # Fallback to regularMarketPrice if currentPrice is not available
            current_price = ticker_obj.info.get('regularMarketPrice')
            
        if current_price is None or np.isnan(current_price):
            raise ValueError(f"Could not retrieve current price for {ticker}")
        
        # Get EPS data
        eps = ticker_obj.info.get('trailingEps')
        if eps is None or np.isnan(eps):
            raise ValueError(f"Could not retrieve EPS data for {ticker}")
        
        # Get financial data for FCF calculation
        financials = ticker_obj.financials
        cash_flow = ticker_obj.cashflow
        
        if financials is None or financials.empty or cash_flow is None or cash_flow.empty:
            raise ValueError(f"Could not retrieve financial data for {ticker}")
        
        # Get net income (try different possible row names)
        try:
            net_income = financials.loc["Net Income"].iloc[0]
        except (KeyError, IndexError):
            try:
                net_income = financials.loc["Net Income Common Stockholders"].iloc[0]
            except (KeyError, IndexError):
                raise ValueError(f"Could not retrieve net income data for {ticker}")
        
        # Get depreciation (try different possible row names)
        try:
            depreciation = cash_flow.loc["Depreciation And Amortization"].iloc[0]
        except (KeyError, IndexError):
            try:
                depreciation = cash_flow.loc["Depreciation"].iloc[0]
            except (KeyError, IndexError):
                raise ValueError(f"Could not retrieve depreciation data for {ticker}")
        
        # Get capex (try different possible row names)
        try:
            capex = abs(cash_flow.loc["Capital Expenditure"].iloc[0])
        except (KeyError, IndexError):
            try:
                capex = abs(cash_flow.loc["Capital Expenditures"].iloc[0])
            except (KeyError, IndexError):
                raise ValueError(f"Could not retrieve capex data for {ticker}")
        
        return ticker_obj, current_price, eps, net_income, depreciation, capex
    
    except Exception as e:
        raise Exception(f"Error retrieving data for {ticker}: {str(e)}")


def calculate_intrinsic_values(ticker: str, growth_rates: float = None, discount_rates: float = None, 
                              terminal_multiples: int = None, projection_years: int = None):
    """
    Calculate intrinsic values and margins of safety using both EPS and FCF methods
    across different parameter combinations.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rates (list): List of growth rates to use
        discount_rates (list): List of discount rates to use
        terminal_multiples (list): List of terminal multiples to use
        projection_years (list): List of projection years to use
        
    Returns:
        dict: Dictionary containing the calculation results
    """
    # Default parameter ranges if not provided
    if growth_rates is None:
        growth_rates = [0.05, 0.10, 0.15]
    if discount_rates is None:
        discount_rates = [0.08, 0.10, 0.12]
    if terminal_multiples is None:
        terminal_multiples = [12, 15, 18]
    if projection_years is None:
        projection_years = [5, 10]
    
    # Get ticker data
    _, current_price, eps, net_income, depreciation, capex = get_ticker_data(ticker)
    
    results = {
        "ticker": ticker,
        "current_price": current_price,
        "eps_values": [],
        "fcf_values": [],
        "parameters": []
    }
    
    # Calculate intrinsic values for all parameter combinations
    for growth_rate in growth_rates:
        for discount_rate in discount_rates:
            for terminal_multiple in terminal_multiples:
                for projection_year in projection_years:
                    # Calculate EPS-based intrinsic value
                    eps_intrinsic_value = calculate_intrinsic_value_earnings(
                        eps, growth_rate, discount_rate, terminal_multiple, projection_year
                    )
                    eps_mos = calculate_margin_of_safety_earnings(
                        eps, growth_rate, discount_rate, terminal_multiple, 
                        projection_year, current_price
                    )
                    
                    # Calculate FCF-based intrinsic value
                    fcf_intrinsic_value = calculate_intrinsic_value_fcf(
                        net_income, depreciation, capex, growth_rate, discount_rate, 
                        terminal_multiple, projection_year
                    )
                    fcf_mos = calculate_margin_of_safety_fcf(
                        net_income, depreciation, capex, growth_rate, discount_rate, 
                        terminal_multiple, projection_year, current_price
                    )
                    
                    # Store results
                    results["eps_values"].append({
                        "intrinsic_value": eps_intrinsic_value,
                        "margin_of_safety": eps_mos
                    })
                    
                    results["fcf_values"].append({
                        "intrinsic_value": fcf_intrinsic_value,
                        "margin_of_safety": fcf_mos
                    })
                    
                    results["parameters"].append({
                        "growth_rate": growth_rate,
                        "discount_rate": discount_rate,
                        "terminal_multiple": terminal_multiple,
                        "projection_years": projection_year
                    })
    
    return results


def plot_intrinsic_values(results):
    """
    Plot intrinsic values and margins of safety.
    
    Args:
        results (dict): Results from calculate_intrinsic_values
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract data
    ticker = results["ticker"]
    current_price = results["current_price"]
    eps_values = [item["intrinsic_value"] for item in results["eps_values"]]
    fcf_values = [item["intrinsic_value"] for item in results["fcf_values"]]
    eps_mos = [item["margin_of_safety"] * 100 for item in results["eps_values"]]
    fcf_mos = [item["margin_of_safety"] * 100 for item in results["fcf_values"]]
    
    # Calculate statistics
    eps_mean = np.mean(eps_values)
    fcf_mean = np.mean(fcf_values)
    eps_median = np.median(eps_values)
    fcf_median = np.median(fcf_values)
    
    # Plot intrinsic values
    ax1.scatter(range(len(eps_values)), eps_values, color='blue', alpha=0.5, label='EPS-based')
    ax1.scatter(range(len(fcf_values)), fcf_values, color='green', alpha=0.5, label='FCF-based')
    
    # Add horizontal line for current price
    ax1.axhline(y=current_price, color='red', linestyle='-', label=f'Current Price (${current_price:.2f})')
    
    # Add horizontal lines for mean values
    ax1.axhline(y=eps_mean, color='blue', linestyle='--', label=f'EPS Mean (${eps_mean:.2f})')
    ax1.axhline(y=fcf_mean, color='green', linestyle='--', label=f'FCF Mean (${fcf_mean:.2f})')
    
    # Set labels and title for first subplot
    ax1.set_xlabel('Calculation Scenario')
    ax1.set_ylabel('Intrinsic Value ($)')
    ax1.set_title(f'Intrinsic Value Estimates for {ticker}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot margins of safety
    ax2.scatter(range(len(eps_mos)), eps_mos, color='blue', alpha=0.5, label='EPS-based')
    ax2.scatter(range(len(fcf_mos)), fcf_mos, color='green', alpha=0.5, label='FCF-based')
    
    # Add horizontal line at 0% margin of safety
    ax2.axhline(y=0, color='red', linestyle='-', label='No Margin of Safety')
    
    # Add horizontal lines for mean values
    ax2.axhline(y=np.mean(eps_mos), color='blue', linestyle='--', 
                label=f'EPS Mean MoS ({np.mean(eps_mos):.2f}%)')
    ax2.axhline(y=np.mean(fcf_mos), color='green', linestyle='--', 
                label=f'FCF Mean MoS ({np.mean(fcf_mos):.2f}%)')
    
    # Set labels and title for second subplot
    ax2.set_xlabel('Calculation Scenario')
    ax2.set_ylabel('Margin of Safety (%)')
    ax2.set_title(f'Margin of Safety Estimates for {ticker}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Intrinsic Value Analysis for {ticker}', fontsize=16)
    plt.tight_layout()
    
    return fig


def analyze_intrinsic_value(ticker: str, growth_rates: float = None, discount_rates: float = None, 
                           terminal_multiples: int = None, projection_years: int = None, 
                           save_plot: bool = True, plot_path: str = None):
    """
    Main function to analyze intrinsic value and margin of safety for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rates (list, optional): List of growth rates to use
        discount_rates (list, optional): List of discount rates to use
        terminal_multiples (list, optional): List of terminal multiples to use
        projection_years (list, optional): List of projection years to use
        save_plot (bool, optional): Whether to save the plot to a file
        plot_path (str, optional): Path to save the plot to.  If None and
            save_plot is True, saves to 'plots/intrinsic_value/{ticker}.png'
        
    Returns:
        tuple: (results, figure, file_path) where results is a dictionary of calculation results,
               figure is the matplotlib figure, and file_path is the path to the saved plot
    """
    try:
        # Calculate intrinsic values
        results = calculate_intrinsic_values(
            ticker, growth_rates, discount_rates, terminal_multiples, projection_years
        )
        
        # Plot results
        fig = plot_intrinsic_values(results)
        
        file_path = None
        # Save plot if requested
        if save_plot:
            # Create directory if it doesn't exist
            if not plot_path:
                save_dir = "plots/intrinsic_value"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"{ticker}.png")
            else:
                file_path = plot_path

            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {file_path}")
        
        # Add summary to results
        eps_values = [item["intrinsic_value"] for item in results["eps_values"]]
        fcf_values = [item["intrinsic_value"] for item in results["fcf_values"]]
        eps_mos = [item["margin_of_safety"] * 100 for item in results["eps_values"]]
        fcf_mos = [item["margin_of_safety"] * 100 for item in results["fcf_values"]]
        
        results["summary"] = {
            "eps_mean_value": np.mean(eps_values),
            "fcf_mean_value": np.mean(fcf_values),
            "eps_median_value": np.median(eps_values),
            "fcf_median_value": np.median(fcf_values),
            "eps_mean_mos": np.mean(eps_mos),
            "fcf_mean_mos": np.mean(fcf_mos),
            "current_price": results["current_price"]
        }
        
        return results, fig, file_path
    
    except Exception as e:
        print(f"Error analyzing intrinsic value for {ticker}: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    # Example usage
    ticker = "MSFT"
    results, fig, file_path = analyze_intrinsic_value(
        ticker,
        growth_rates=[0.05, 0.10, 0.15],
        discount_rates=[0.08, 0.10, 0.12],
        terminal_multiples=[12, 15, 18],
        projection_years=[5, 10],
        save_plot=True
    )
    
    if results:
        print(f"\nIntrinsic Value Analysis for {ticker}:")
        print(f"Current Price: ${results['current_price']:.2f}")
        print(f"EPS-based Mean Intrinsic Value: ${results['summary']['eps_mean_value']:.2f}")
        print(f"FCF-based Mean Intrinsic Value: ${results['summary']['fcf_mean_value']:.2f}")
        print(f"EPS-based Mean Margin of Safety: {results['summary']['eps_mean_mos']:.2f}%")
        print(f"FCF-based Mean Margin of Safety: {results['summary']['fcf_mean_mos']:.2f}%")
        print(f"Plot saved at: {file_path}")