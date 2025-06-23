import yfinance as yf
import pandas as pd
import numpy as np
from stock_analyser.utils.convert_currency import convert_currency


def interpolate(initial_value, terminal_value, nyears):
    return np.linspace(initial_value, terminal_value, nyears)


def calculate_present_value(cash_flows, discount_rate):
    # Calculate the present value using the formula: PV = CF / (1 + r)^t + TV/(1 + r)^T
    present_values_cf = [
        cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, start=1)
    ]
    return present_values_cf


def format_value(val):
    if not pd.isna(val) or val != "nan":
        return f"{val / 1e6:.2f}"


def get_cost_of_equity(ticker: str):
    stock = yf.Ticker(ticker)

    beta = stock.info.get("beta", 0)
    # print(f"Beta: {beta:.2f}")

    # risk-free rate
    risk_free_rate = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 100
    # print(f"Risk-free Rate: {risk_free_rate:.2%}")

    # market_return = 0.07
    market_return = yf.Ticker("VTI").info.get("threeYearAverageReturn")
    # print(f"Market Return: {market_return:.2%}")

    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    # print(f"Cost of Equity: {cost_of_equity:.2%}")

    return cost_of_equity


def calculate_cost_of_debt(ticker: str):
    stock = yf.Ticker(ticker)
    exchange_rate = convert_currency(ticker)
    debt = stock.info.get("totalDebt", 0) * exchange_rate
    # print(f"Total Debt: ${debt:,.2f}")
    interest_expense = (
        stock.financials.loc["Interest Expense"].dropna().iloc[0] * exchange_rate
    )
    # print(f"Interest Expense: ${interest_expense:,.2f}")
    tax_rate = stock.financials.loc["Tax Rate For Calcs"].dropna().iloc[0]
    # print(f"Tax Rate: {tax_rate:.2%}")

    cost_of_debt = interest_expense * (1 - tax_rate) / debt
    # print(f"Cost of Debt: {cost_of_debt:.2%}")

    return cost_of_debt


def get_WACC(ticker: str, tax_perc_T: float):
    exchange_rate = convert_currency(ticker)
    stock = yf.Ticker(ticker)
    cost_of_equity = get_cost_of_equity(ticker)
    cost_of_debt = calculate_cost_of_debt(ticker)
    market_cap = stock.info.get("marketCap", 0)
    # print(f"Market Cap: ${market_cap:,.2f}")
    debt = stock.info.get("totalDebt", 0) * exchange_rate
    # print(f"Debt (from balance sheet): ${debt:,.2f}")
    total = market_cap + debt
    # print(f"Total (Market Cap + Debt): ${total:,.2f}")
    AfterTaxCostOfDebt = cost_of_debt * (1 - tax_perc_T)
    # print(f"After Tax Cost of Debt: {AfterTaxCostOfDebt:.2%}")
    wacc = (AfterTaxCostOfDebt * debt / total) + (cost_of_equity * market_cap / total)
    # print(f"WACC: {wacc:.2%}")

    return wacc


def get_data(ticker: str):
    stock = yf.Ticker(ticker)
    exchange_rate = convert_currency(ticker)
    # download BS, IS, CFS from EOD historical data
    balance_sheet = stock.balance_sheet.apply(lambda x: x * exchange_rate)
    income_statment = stock.financials.apply(lambda x: x * exchange_rate)
    cash_flow = stock.cashflow.apply(lambda x: x * exchange_rate)

    # transpose and concatenate
    df_bal = pd.DataFrame(balance_sheet).T
    df_inc = pd.DataFrame(income_statment).T
    df_cfs = pd.DataFrame(cash_flow).T
    df_all = pd.concat([df_bal, df_inc, df_cfs], axis=1).sort_index()
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]  # remove duplicated columns

    # convert all None values to np.NaN datatypes
    df = df_all.map(lambda x: float(x) if x is not None else np.NaN)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.year

    # compute % of sales variables
    df["rev_growth"] = df["Total Revenue"].pct_change()
    df["delta_nwc"] = df["Working Capital"].diff()
    df["ebit_of_sales"] = df["EBIT"] / df["Total Revenue"]
    df["dna_of_sales"] = df["Depreciation And Amortization"] / df["Total Revenue"]
    df["capex_of_sales"] = df["Capital Expenditure"] / df["Total Revenue"]
    df["nwc_of_sales"] = df["Change In Working Capital"] / df["Total Revenue"]
    df["tax_of_ebit"] = df["Tax Provision"] / df["EBIT"]
    df["ebiat"] = df["EBIT"] - df["Tax Provision"]

    return df


def calculate_implied_share_price(
    ticker,
    n,
    OutShares,
    revenue_growth_T,
    ebit_perc_T,
    tax_perc_T,
    dna_perc_T,
    capex_perc_T,
    nwc_perc_T,
    WACC,
    TGR,
):
    stock = yf.Ticker(ticker)
    df = get_data(ticker)

    last_year = df.iloc[-1, :]

    years = range(df.index[-1] + 1, df.index[-1] + n + 1)
    df_proj = pd.DataFrame(index=years, columns=df.columns)

    df_proj["rev_growth"] = interpolate(last_year["rev_growth"], revenue_growth_T, n)
    df_proj["ebit_of_sales"] = interpolate(last_year["ebit_of_sales"], ebit_perc_T, n)
    df_proj["dna_of_sales"] = interpolate(last_year["dna_of_sales"], dna_perc_T, n)
    df_proj["capex_of_sales"] = interpolate(
        last_year["capex_of_sales"], capex_perc_T, n
    )
    df_proj["tax_of_ebit"] = interpolate(last_year["tax_of_ebit"], tax_perc_T, n)
    df_proj["nwc_of_sales"] = interpolate(last_year["nwc_of_sales"], nwc_perc_T, n)

    df_proj["totalRevenue"] = (
        last_year["Total Revenue"] * (1 + df_proj["rev_growth"]).cumprod()
    )
    df_proj["ebit"] = df_proj["totalRevenue"] * df_proj["ebit_of_sales"]
    df_proj["capitalExpenditures"] = (
        df_proj["totalRevenue"] * df_proj["capex_of_sales"] * -1
    )
    df_proj["depreciationAndAmortization"] = (
        df_proj["totalRevenue"] * df_proj["dna_of_sales"]
    )
    df_proj["delta_nwc"] = df_proj["totalRevenue"] * df_proj["nwc_of_sales"] * -1
    df_proj["taxProvision"] = df_proj["ebit"] * df_proj["tax_of_ebit"]
    df_proj["ebiat"] = df_proj["ebit"] - df_proj["taxProvision"]
    df_proj["freeCashFlow"] = (
        df_proj["ebiat"]
        + df_proj["depreciationAndAmortization"]
        - df_proj["capitalExpenditures"]
        - df_proj["delta_nwc"]
    )

    df_proj["pv_FCF"] = calculate_present_value(df_proj["freeCashFlow"].values, WACC)
    # print(f"Present Value of Free Cash Flows: ${df_proj['pv_FCF']}")

    terminal_value = df_proj["freeCashFlow"].values[-1] * (1 + TGR) / (WACC - TGR)
    # print(f"Terminal Value: ${terminal_value:,.2f}")

    pv_terminal_value = terminal_value / (1 + WACC) ** n
    # print(f"Present Value of Terminal Value: ${pv_terminal_value:,.2f}")

    enterprise_value = np.sum(df_proj["pv_FCF"]) + pv_terminal_value
    # print(f"Enterprise Value: ${enterprise_value:,.2f}")

    cash = last_year["Cash And Cash Equivalents"]
    # print(f"Cash: ${cash:,.2f}")
    debt = stock.info.get("totalDebt")
    # print(f"Debt: ${debt:,.2f}")

    eq_value = enterprise_value - debt + cash
    # print(f"Equity Value: ${eq_value:,.2f}")

    implied_share_price = eq_value / OutShares
    # print(f"Implied Share Price: ${implied_share_price:.2f}")

    return implied_share_price


if __name__ == "__main__":
    # Years of projections
    n = 5
    # % Revenue Growth = 7%
    revenue_growth_T = 0.07
    # % Ebit/Sales = 23%
    ebit_perc_T = 0.23
    # % D&A/Sales = 3%
    dna_perc_T = 0.03
    # % Capex/Sales = 5%
    capex_perc_T = 0.05
    # % Δ Net Working Capital/Sales = 5%
    nwc_perc_T = 0.05
    # % Tax/Ebit = 21%
    tax_perc_T = 0.21
    # Terminal Growth Rate = 2.5%
    TGR = 0.025

    ticker = "RDDT"
    output_dir = "plots/sensitivity_anal"
    timestamp = "2025-03-17_012000"
    stock = yf.Ticker(ticker)
    OutShares = stock.info.get("sharesOutstanding")
    # get WACC
    wacc = get_WACC(ticker, tax_perc_T)
    print(f"WACC: {wacc:.2%}")
    # get data
    df = get_data(ticker)

    inputs = {
        "n": n,
        "OutShares": OutShares,
        "revenue_growth_T": revenue_growth_T,
        "ebit_perc_T": ebit_perc_T,
        "tax_perc_T": tax_perc_T,
        "dna_perc_T": dna_perc_T,
        "capex_perc_T": capex_perc_T,
        "nwc_perc_T": nwc_perc_T,
        "WACC": wacc,
        "TGR": TGR,
    }
    price = calculate_implied_share_price(ticker, **inputs)
    print(f"Estimated Implied Share Price for {ticker}: ${price:.2f}")
