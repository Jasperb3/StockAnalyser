import yfinance as yf


def get_data(symbol):
    stock = yf.Ticker(symbol)
    variables = {
        "latest_revenue": stock.financials.loc["Total Revenue"].dropna().iloc[0],
        "total_shares": stock.info.get("sharesOutstanding"),
    }

    return variables


def calculate_intrinsic_value_dcf(
    latest_revenue,
    rev_growth_rates,
    fcf_margins,
    discount_rate,
    terminal_growth_rate,
    total_shares,
):
    """
    Calculates the intrinsic value of a company using the Discounted Cash Flow (DCF) model.

    Args:
        latest_revenue (float): The company's most recent annual revenue.
        rev_growth_rates (list): A list of annual revenue growth rates (as decimals) for the forecast period.
        fcf_margins (list): A list of free cash flow margins (as decimals) for the forecast period.
        discount_rate (float): The discount rate (WACC) as a decimal.
        terminal_growth_rate (float): The terminal growth rate as a decimal.
        total_shares (int): The total number of outstanding shares.

    Returns:
        float: The intrinsic value per share.
    """

    n_years = len(rev_growth_rates)

    # Project FCF
    proj_fcf = []
    rev_inc = latest_revenue
    for i in range(n_years):
        rev_inc = rev_inc * (1 + rev_growth_rates[i])
        proj_fcf.append(rev_inc * fcf_margins[i])

    # Get discount factors
    discount_factors = [(1 + discount_rate) ** (i + 1) for i in range(n_years)]

    # Total discounted fcf for n_years
    discounted_fcf = sum(proj_fcf[i] / discount_factors[i] for i in range(n_years))

    # Terminal value (discounted)
    terminal_value = (proj_fcf[-1] * (1 + terminal_growth_rate)) / (
        discount_rate - terminal_growth_rate
    )
    terminal_value /= discount_factors[-1]

    # Total value for all shares
    todays_value = discounted_fcf + terminal_value

    # Fair value per share
    fair_value_per_share = todays_value / total_shares

    return fair_value_per_share


if __name__ == "__main__":
    # Example Usage
    latest_revenue = 1000  # Example: $1000 million
    rev_growth_rates = [0.1, 0.12, 0.08, 0.05, 0.03]  # Example: 10%, 12%, 8%, 5%, 3%
    fcf_margins = [0.2, 0.22, 0.25, 0.25, 0.25]  # Example: 20%, 22%, 25%, 25%, 25%
    discount_rate = 0.09  # Example: 9%
    terminal_growth_rate = 0.025  # Example: 2.5%
    total_shares = 100  # Example: 100 million shares

    inputs = get_data("AAPL")

    intrinsic_value = calculate_intrinsic_value_dcf(**inputs)

    print(f"Intrinsic Value per Share: ${round(intrinsic_value, 2):.2f}")
