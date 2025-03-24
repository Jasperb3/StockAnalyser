# import yfinance as yf
# import matplotlib.pyplot as plt


# Margin of Safety – Earnings-Based Approach


def calculate_intrinsic_value_earnings(
    diluted_eps: float,
    growth_rate: float,
    discount_rate: float,
    terminal_multiple: int,
    projection_years: int,
) -> float:
    """
    Calculate the intrinsic value using a simplified discounted earnings model.

    Parameters:
    - diluted_eps: Current diluted earnings per share.
    - growth_rate: Annual EPS growth rate (as a decimal, e.g., 0.10 for 10%).
    - discount_rate: Annual discount rate (as a decimal, e.g., 0.11 for 11%).
    - terminal_multiple: Multiple applied to the final year's projected EPS.
    - projection_years: Number of years to project EPS.

    Returns:
    - intrinsic_value: Estimated intrinsic value based on future earnings.
    """
    intrinsic_value = 0.0

    # Sum the discounted projected EPS for each year
    for year in range(1, projection_years + 1):
        projected_eps = diluted_eps * ((1 + growth_rate) ** year)
        discounted_eps = projected_eps / ((1 + discount_rate) ** year)
        intrinsic_value += discounted_eps

    # Calculate and discount the terminal value using the final year's EPS
    final_year_eps = diluted_eps * ((1 + growth_rate) ** projection_years)
    terminal_value = (final_year_eps * terminal_multiple) / (
        (1 + discount_rate) ** projection_years
    )
    intrinsic_value += terminal_value

    return intrinsic_value


def calculate_margin_of_safety_earnings(
    diluted_eps: float,
    growth_rate: float,
    discount_rate: float,
    terminal_multiple: int,
    projection_years: int,
    current_price: float,
) -> float:
    """
    Calculate the margin of safety based on an earnings-based intrinsic value estimate.

    Parameters:
    - diluted_eps: Current diluted earnings per share.
    - growth_rate: Annual EPS growth rate (as a decimal).
    - discount_rate: Annual discount rate (as a decimal).
    - terminal_multiple: Terminal multiple for final year EPS.
    - projection_years: Number of years to project EPS.
    - current_price: The current market price of the stock.

    Returns:
    - margin_of_safety: Calculated margin of safety as a decimal.
    """
    intrinsic_value = calculate_intrinsic_value_earnings(
        diluted_eps, growth_rate, discount_rate, terminal_multiple, projection_years
    )
    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    return margin_of_safety


# Margin of Safety – FCF-Based Approach


def calculate_intrinsic_value_fcf(
    net_income: int,
    depreciation: float,
    capex: float,
    growth_rate: float,
    discount_rate: float,
    terminal_multiple: int,
    projection_years: int,
) -> float:
    """
    Calculate the intrinsic value based on free cash flow (FCF) using a discounted cash flow (DCF) model.

    Parameters:
    - net_income: Company's net income.
    - depreciation: Depreciation expense.
    - capex: Capital expenditures.
    - growth_rate: Annual growth rate for FCF (as a decimal).
    - discount_rate: Annual discount rate (as a decimal).
    - terminal_multiple: Terminal multiple for the final year's FCF.
    - projection_years: Number of years to project FCF.

    Returns:
    - intrinsic_value: Estimated intrinsic value based on FCF.
    """
    # Calculate initial free cash flow
    initial_fcf = net_income + depreciation - capex
    intrinsic_value = 0.0

    # Sum the discounted projected FCF for each year
    for year in range(1, projection_years + 1):
        projected_fcf = initial_fcf * ((1 + growth_rate) ** year)
        discounted_fcf = projected_fcf / ((1 + discount_rate) ** year)
        intrinsic_value += discounted_fcf

    # Calculate and discount the terminal value using the final year's FCF
    final_year_fcf = initial_fcf * ((1 + growth_rate) ** projection_years)
    terminal_value = (final_year_fcf * terminal_multiple) / (
        (1 + discount_rate) ** projection_years
    )
    intrinsic_value += terminal_value

    return intrinsic_value


def calculate_margin_of_safety_fcf(
    net_income: int,
    depreciation: float,
    capex: float,
    growth_rate: float,
    discount_rate: float,
    terminal_multiple: int,
    projection_years: int,
    current_price: float,
) -> float:
    """
    Calculate the margin of safety based on an FCF-based intrinsic value estimate.

    Parameters:
    - net_income: Company's net income.
    - depreciation: Depreciation expense.
    - capex: Capital expenditures.
    - growth_rate: Annual growth rate for FCF (as a decimal).
    - discount_rate: Annual discount rate (as a decimal).
    - terminal_multiple: Terminal multiple for final year's FCF.
    - projection_years: Number of years to project FCF.
    - current_price: The current market price of the stock.

    Returns:
    - margin_of_safety: Calculated margin of safety as a decimal.
    """
    intrinsic_value = calculate_intrinsic_value_fcf(
        net_income,
        depreciation,
        capex,
        growth_rate,
        discount_rate,
        terminal_multiple,
        projection_years,
    )
    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    return margin_of_safety


# Example test case for earnings-based margin of safety:
if __name__ == "__main__":
    # Earnings-based input parameters
    diluted_eps = 12.511  # Example EPS
    growth_rate = 0.192  # 19.2% growth rate
    discount_rate = 0.11  # 11% discount rate
    terminal_multiple = 15  # Example terminal multiple (can be thought of as a proxy for price/earnings)
    projection_years = 10  # Projection period of 10 years
    current_price = 388.56  # Current market price

    intrinsic_value_earnings = calculate_intrinsic_value_earnings(
        diluted_eps, growth_rate, discount_rate, terminal_multiple, projection_years
    )
    mos_earnings = calculate_margin_of_safety_earnings(
        diluted_eps,
        growth_rate,
        discount_rate,
        terminal_multiple,
        projection_years,
        current_price,
    )
    print(f"Earnings-Based Intrinsic Value: {intrinsic_value_earnings:.2f}")
    print(f"Earnings-Based Margin of Safety: {mos_earnings * 100:.2f}%")

    # Example test case for FCF-based margin of safety:

    # FCF-based input parameters
    net_income = 1000000  # Example net income
    depreciation = 50000.0  # Example depreciation expense
    capex = 30000.0  # Example capital expenditures
    growth_rate = 0.10  # 10% growth rate for FCF
    discount_rate = 0.11  # 11% discount rate
    terminal_multiple = 15  # Terminal multiple for FCF
    projection_years = 10  # Projection period of 10 years
    current_price = 388.56  # Current market price

    intrinsic_value_fcf = calculate_intrinsic_value_fcf(
        net_income,
        depreciation,
        capex,
        growth_rate,
        discount_rate,
        terminal_multiple,
        projection_years,
    )
    mos_fcf = calculate_margin_of_safety_fcf(
        net_income,
        depreciation,
        capex,
        growth_rate,
        discount_rate,
        terminal_multiple,
        projection_years,
        current_price,
    )
    print(f"FCF-Based Intrinsic Value: {intrinsic_value_fcf:,.2f}")
    print(f"FCF-Based Margin of Safety: {mos_fcf * 100:.2f}%")
