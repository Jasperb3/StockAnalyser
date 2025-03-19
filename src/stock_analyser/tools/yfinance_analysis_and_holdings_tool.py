from typing import Type
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
# from stock_analyser.tools.tool_utils.intrinsic_value_calcs import calculate_intrinsic_value_dcf
from stock_analyser.tools.tool_utils.dcf_implied_price_calcs import calculate_implied_share_price, get_WACC

now = datetime.now()
TODAY = now.strftime("%Y-%m-%d")
ONE_YEAR_AGO = now - timedelta(days=365)
SIX_MONTHS_AGO = now - timedelta(days=180)
THREE_MONTHS_AGO = now - timedelta(days=90)
ONE_MONTH_AGO = now - timedelta(days=30)


class YFinanceAnalysisAndHoldingsToolInput(BaseModel):
    """Input schema for YFinanceAnalysisAndHoldingsTool."""

    ticker: str = Field(..., description="Ticker of the stock to analyze.")


class YFinanceAnalysisAndHoldingsTool(BaseTool):
    name: str = "YFinance Analysis and Holdings Tool"
    description: str = (
        "Analyzes a stock and its holdings."
    )
    args_schema: Type[BaseModel] = YFinanceAnalysisAndHoldingsToolInput

    def _run(self, ticker: str) -> str:
        company_ticker = yf.Ticker(ticker)

        # get recommendations
        recommendations = company_ticker.recommendations
        if recommendations is not None and not recommendations.empty:
            recommendations_list = []
            for index, row in recommendations.iterrows():
                period = row['period']
                strong_buy = row['strongBuy']
                buy = row['buy']
                hold = row['hold']
                sell = row['sell']
                strong_sell = row['strongSell']
                recommendations_list.append(
                    f"Period: {period}, Strong Buy: {strong_buy}, Buy: {buy}, "
                    f"Hold: {hold}, Sell: {sell}, Strong Sell: {strong_sell}\n"
                )
        else:
            recommendations_list = ["No recommendations available."]

        # get upgrades downgrades   
        upgrades_downgrades = company_ticker.upgrades_downgrades
        upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index >= ONE_MONTH_AGO]
        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
            upgrades_downgrades_list = []
            for index, row in upgrades_downgrades.iterrows():
                date = upgrades_downgrades.index[0].strftime("%Y-%m-%d")
                firm = row['Firm']
                ToGrade = row['ToGrade']
                FromGrade = row['FromGrade']
                upgrades_downgrades_list.append(
                    f"Date: {date}, Firm: {firm}, From Grade: {FromGrade}, To Grade: {ToGrade}\n"
                )
        else:
            upgrades_downgrades_list = ["No upgrades downgrades available."]

        # get analyst price targets
        analyst_price_targets = company_ticker.analyst_price_targets
        if analyst_price_targets:
            analyst_price_targets_list = [f"{type}: {price}\n" for type, price in analyst_price_targets.items()]
        else:
            analyst_price_targets_list = ["No analyst price targets available."]

        # # get estimated intrinsic value
        
        # in_val_inputs = {
        #     "rev_growth_rates": [0.10, 0.12, 0.08, 0.05, 0.03],
        #     "fcf_margins": [0.20, 0.22, 0.25, 0.25, 0.25],
        #     "discount_rate": 0.09,
        #     "terminal_growth_rate": 0.025
        # }

        # in_val_inputs['latest_revenue'] = company_ticker.financials.loc["Total Revenue"].dropna().iloc[0]
        # in_val_inputs['total_shares'] = company_ticker.info.get('sharesOutstanding')
        
        # estimated_intrinsic_value = calculate_intrinsic_value_dcf(**in_val_inputs)

        # get implied share price
        outShares = company_ticker.info.get('sharesOutstanding')
        tax_provision = company_ticker.financials.loc["Tax Provision"].dropna().iloc[0]
        ebit = company_ticker.financials.loc["EBIT"].dropna().iloc[0]

        tax_perc_T = tax_provision / ebit

        wacc = get_WACC(ticker, tax_perc_T)

        fcf_calcs_inputs = {
            "n": 5,
            "OutShares": outShares,
            "revenue_growth_T": 0.07,
            "ebit_perc_T": 0.23,
            "tax_perc_T": tax_perc_T,
            "dna_perc_T": 0.03,
            "capex_perc_T": 0.05,
            "nwc_perc_T": 0.05,
            "WACC": wacc,
            "TGR": 0.025
        }
        implied_share_price = calculate_implied_share_price(ticker, **fcf_calcs_inputs)

        # get earnings estimate
        earnings_estimate = company_ticker.earnings_estimate
        if earnings_estimate is not None and not earnings_estimate.empty:
            earnings_estimate_list = []
            for index, row in earnings_estimate.iterrows():
                period = index
                avg = row['avg']
                low = row['low']
                high = row['high']
                year_ago_eps = row['yearAgoEps']
                number_of_analysts = row['numberOfAnalysts']
                growth = row['growth']
                earnings_estimate_list.append(
                    f"Period: {period}, Average: {avg}, Low: {low}, High: {high}, "
                    f"Year Ago EPS: {year_ago_eps}, Number of Analysts: {int(number_of_analysts)}, Growth: {growth:.2%}\n"
                )
        else:
            earnings_estimate_list = ["No earnings estimates available."]

        # get revenue estimate
        revenue_estimate = company_ticker.revenue_estimate
        if revenue_estimate is not None and not revenue_estimate.empty:
            revenue_estimate_list = []
            for index, row in revenue_estimate.iterrows():
                period = index
                avg = row['avg']
                low = row['low']
                high = row['high']
                year_ago_rev = row['yearAgoRevenue']
                number_of_analysts = row['numberOfAnalysts']
                growth = row['growth']
                revenue_estimate_list.append(
                    f"Period: {period}, Average: ${avg:,.0f}, Low: ${low:,.0f}, High: ${high:,.0f}, "
                    f"Year Ago Revenue: ${year_ago_rev:,.0f}, Number of Analysts: {int(number_of_analysts)}, Growth: {growth:.2%}\n"
                )
        else:
            revenue_estimate_list = ["No revenue estimates available."]
        
        # get earnings history
        earnings_history = company_ticker.earnings_history
        if earnings_history is not None and not earnings_history.empty:
            earnings_history_list = []
            for index, row in earnings_history.iterrows():
                date = earnings_history.index[0].strftime("%Y-%m-%d")
                eps_actual = row['epsActual']
                eps_estimate = row['epsEstimate']
                eps_difference = row['epsDifference']
                surprise_percent = row['surprisePercent']
                earnings_history_list.append(
                    f"Date: {date}, EPS Actual: {eps_actual}, EPS Estimate: {eps_estimate}, "
                    f"EPS Difference: {eps_difference}, Surprise Percent: {surprise_percent:.2%}\n"
                )
        else:
            earnings_history_list = ["No earnings history available."]

        # get eps trend
        eps_trend = company_ticker.eps_trend
        if eps_trend is not None and not eps_trend.empty:
            eps_trend_list = []
            for index, row in eps_trend.iterrows():
                period = index
                current = row['current']
                week_ago = row['7daysAgo']
                month_ago = row['30daysAgo']
                two_months_ago = row['60daysAgo']
                three_months_ago = row['90daysAgo']
                eps_trend_list.append(
                    f"Period: {period}, Current: {current}, 1 Week Ago: {week_ago}, 1 Month Ago: {month_ago}, "
                    f"2 Months Ago: {two_months_ago}, 3 Months Ago: {three_months_ago}\n"
                )
        else:
            eps_trend_list = ["No eps trend available."]
        
        # get eps revisions
        eps_revisions = company_ticker.eps_revisions
        if eps_revisions is not None and not eps_revisions.empty:
            eps_revisions_list = []
            for index, row in eps_revisions.iterrows():
                period = index
                up_last_seven_days = row['upLast7days']
                down_last_seven_days = row['downLast7Days']
                up_last_thirty_days = row['upLast30days']
                down_last_thirty_days = row['downLast30days']
                eps_revisions_list.append(
                    f"Period: {period}, Up Last 7 Days: {up_last_seven_days}, Down Last 7 Days: {down_last_seven_days}, "
                    f"Up Last 30 Days: {up_last_thirty_days}, Down Last 30 Days: {down_last_thirty_days}\n"
                )

        # get growth estimates
        growth_estimates = company_ticker.growth_estimates
        if growth_estimates is not None and not growth_estimates.empty:
            growth_estimates_list = []
            for index, row in growth_estimates.iterrows():
                period = index
                stock_trend = row['stockTrend']
                index_trend = row['indexTrend']
                growth_estimates_list.append(
                    f"Period: {period}, Stock Trend: {stock_trend}, Index Trend: {index_trend}\n"
                )
        else:
            growth_estimates_list = ["No growth estimates available."]

        # # get funds data
        # funds_data = company_ticker.funds_data

        # get shares outstanding
        shares_outstanding = company_ticker.info.get('sharesOutstanding')

        # held percent insiders / institutions
        held_percent_insiders = company_ticker.info.get('heldPercentInsiders')
        held_percent_institutions = company_ticker.info.get('heldPercentInstitutions')

        # get insider purchases
        insider_purchases = company_ticker.insider_purchases
        if insider_purchases is not None and not insider_purchases.empty:
            insider_purchases_list = []
            for index, row in insider_purchases.iterrows():
                if index == 0:
                  purchases_shares = row['Shares']
                  purchases_trans = row['Trans']
                  insider_purchases_list.append(f"Purchases Shares: {purchases_shares:,.0f}, Transactions: {purchases_trans}\n")
                elif index == 1:
                  sales_shares = row['Shares']
                  sales_trans = row['Trans']
                  insider_purchases_list.append(f"Sales Shares: {sales_shares:,.0f}, Transactions: {sales_trans}\n")
                elif index == 2:
                  net_shares = row['Shares']
                  net_trans = row['Trans']
                  insider_purchases_list.append(f"Net Shares Purchased (Sold): {net_shares:,.0f}, Transactions: {net_trans}\n")
                elif index == 3:
                  total_shares = row['Shares']
                  insider_purchases_list.append(f"Total Insider Shares Held: {total_shares:,.0f}\n")
                elif index == 4:
                  net_shares_percent = row['Shares']
                  insider_purchases_list.append(f"% Net Shares Purchased (Sold): {net_shares_percent:.2%}\n")
                elif index == 5:
                  buy_shares_percent = row['Shares']
                  insider_purchases_list.append(f"% Buy Shares: {buy_shares_percent:.2%}\n")
                elif index == 6:
                  sell_shares_percent = row['Shares']
                  insider_purchases_list.append(f"% Sell Shares: {sell_shares_percent:.2%}\n")
        else:
            insider_purchases_list = ["No insider purchases data available."]
                
        
        # get insider transactions
        insider_transactions = company_ticker.insider_transactions
        insider_transactions = insider_transactions.loc[insider_transactions['Start Date'] >= ONE_MONTH_AGO]
        if insider_transactions is not None and not insider_transactions.empty:
            insider_transactions_list = []
            for index, row in insider_transactions.iterrows():
                date = row['Start Date'].strftime("%Y-%m-%d")
                shares = row['Shares']
                value = row['Value']
                text = row['Text']
                insider = row['Insider']
                position = row['Position']
                ownership = 'Direct' if row['Ownership'] == 'D' else 'Indirect'
                insider_transactions_list.append(
                    f"Date: {date}, Insider: {insider}, Position: {position}, "
                    f"Shares: {shares:,.0f}, Value: ${value:,.0f}, Transction type: {text}, Ownership: {ownership}\n"
                )
        else:
            insider_transactions_list = ["No insider transactions available."]


        # insider roster holders
        insider_roster_holders = company_ticker.insider_roster_holders
        if insider_roster_holders is not None and not insider_roster_holders.empty:
            insider_roster_holders_list = []
            for index, row in insider_roster_holders.iterrows():
                if not np.isnan(row['Shares Owned Directly']) and row['Shares Owned Directly'] > 0:
                    insider = row['Name']
                    position = row['Position']
                    shares = row['Shares Owned Directly']
                    insider_roster_holders_list.append(f"Insider: {insider}, Position: {position}, Shares owned directly: {shares:,.0f}\n")
        else:
            insider_roster_holders_list = ["No insider roster holders available."]


        # get major holders
        major_holders = company_ticker.major_holders.to_dict()
        if major_holders:
            values = major_holders.get('Value')
            major_holders_list = []
            major_holders_list.append(f"Percent held by insiders: {values['insidersPercentHeld']:.2%}\n")
            major_holders_list.append(f"Percent held by institutions: {values['institutionsPercentHeld']:.2%}\n")
            major_holders_list.append(f"Number of institutions holding shares: {values['institutionsCount']:,.0f}\n")
        else:
            major_holders_list = ["No major holders data available."]

        # get institutional holders
        institutional_holders = company_ticker.institutional_holders
        if institutional_holders is not None and not institutional_holders.empty:
            institutional_holders_list = []
            for index, row in institutional_holders.iterrows():
                date = row['Date Reported'].strftime("%Y-%m-%d")
                institution = row['Holder']
                percent_held = row['pctHeld']
                shares = row['Shares']
                value = row['Value']
                percent_change = row['pctChange']
                institutional_holders_list.append(
                    f"Institution: {institution}, Percent Held: {percent_held:.2%}, "
                    f"Shares: {shares:,.0f}, Value: ${value:,.0f}, Percent Change: {percent_change:.2%}\n"
                )
        else:
            institutional_holders_list = ["No institutional holders available."]

        # get mutual fund holders
        mutualfund_holders = company_ticker.mutualfund_holders
        if mutualfund_holders is not None and not mutualfund_holders.empty:
            mutualfund_holders_list = []
            for index, row in mutualfund_holders.iterrows():
                date = row['Date Reported'].strftime("%Y-%m-%d")
                mutualfund = row['Holder']
                percent_held = row['pctHeld']
                shares = row['Shares']
                value = row['Value']
                percent_change = row['pctChange']
                mutualfund_holders_list.append(
                    f"Mutual Fund: {mutualfund}, Percent Held: {percent_held:.2%}, Shares: {shares:,}, Value: ${value:,}, Percent Change: {percent_change:.2%}\n"
                )
        else:
            mutualfund_holders_list = ["No mutual fund holders available."]

        # return formatted data
        data = f"Analyst Recommendations:\n{''.join(recommendations_list)}\n"
        data += f"Upgrades Downgrades:\n{''.join(upgrades_downgrades_list)}\n"
        data += f"Analyst Price Targets:\n{''.join(analyst_price_targets_list)}\n"
        data += (
            f"Intrinsic value per share: ${implied_share_price:,.2f}\n"
            f"*Assumptions: "
            f"Number of years of projections for DCF model = {fcf_calcs_inputs['n']}, "
            f"Terminal Revenue Growth = {fcf_calcs_inputs['revenue_growth_T']:.2%}, "
            f"Terminal EBIT (Earnings Before Interest and Taxes) / Sales = {fcf_calcs_inputs['ebit_perc_T']:.2%}, "
            f"Terminal Tax Percentage of EBIT = {fcf_calcs_inputs['tax_perc_T']:.2%}, "
            f"Terminal Depreciation and Amortization / Sales = {fcf_calcs_inputs['dna_perc_T']:.2%}, "
            f"Terminal Capital Expenditures / Sales = {fcf_calcs_inputs['capex_perc_T']:.2%}, "
            f"Terminal Change in Net Working Capital / Sales = {fcf_calcs_inputs['nwc_perc_T']:.2%}, "
            f"Weighted Average Cost of Capital = {fcf_calcs_inputs['WACC']:.2%}, "
            f"Terminal Growth Rate = {fcf_calcs_inputs['TGR']:.2%})\n\n"
        )
        data += f"Earnings Estimate:\n{''.join(earnings_estimate_list)}\n"
        data += f"Revenue Estimate:\n{''.join(revenue_estimate_list)}\n"
        data += f"Earnings History:\n{''.join(earnings_history_list)}\n"
        data += f"EPS Trend:\n{''.join(eps_trend_list)}\n"
        data += f"EPS Revisions:\n{''.join(eps_revisions_list)}\n"
        data += f"Growth Estimates:\n{''.join(growth_estimates_list)}\n"
        data += f"Shares Outstanding: {shares_outstanding:,.0f}\n"
        data += f"Percentage held by insiders: {held_percent_insiders:.2%}\n"
        data += f"Percentage held by institutions: {held_percent_institutions:.2%}\n"
        data += f"Insider Purchases last 6 months:\n{''.join(insider_purchases_list)}\n"
        data += f"Insider Transactions last 30 days:\n{''.join(insider_transactions_list)}\n"
        data += f"Insider Roster Holders:\n{''.join(insider_roster_holders_list)}\n"
        data += f"Major Holders:\n{''.join(major_holders_list)}\n"
        data += f"Institutional Holders:\n{''.join(institutional_holders_list)}\n"
        data += f"Mutual Fund Holders:\n{''.join(mutualfund_holders_list)}\n"
        
        
        return data
    

if __name__ == "__main__":
    tool = YFinanceAnalysisAndHoldingsTool()
    print(tool.run("NVDA"))
