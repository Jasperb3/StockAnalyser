import yfinance as yf
import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime

# In-memory logger to collect warnings and logs
class _MemHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(self.format(record))

_mem_handler = _MemHandler()
logging.basicConfig(level=logging.INFO, handlers=[_mem_handler])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_mem_handler)
logger.propagate = False

# --------------------------- CONSTANTS ---------------------------
DEFAULT_ERP = 0.05
DEFAULT_TERMINAL_GROWTH_RATE = 0.025
DEFAULT_STAGE1_YEARS = 5
DEFAULT_TRANSITION_YEARS = 5
GROWTH_CAP = 0.25
GROWTH_FLOOR = 0.00
MIN_YEARS_FOR_AVG = 3
RISK_FREE_RATE_TICKER = "^TNX"
MARKET_INDEX_TICKER = "^GSPC"
DEFAULT_COST_OF_DEBT = 0.05
DEFAULT_TAX_RATE = 0.21

# ---------------------- CORE VALUATION FUNCTION ----------------------
def run_intrinsic_valuation(ticker: str) -> dict:
    """
    Non-interactive multi-stage DCF valuation for a given ticker.
    Returns dict with valuation outputs and warnings.
    """
    warnings_list = []

    # Helper: fetch statements
    def _fetch_statements(tkr):
        stk = yf.Ticker(tkr)
        info = stk.info or {}
        inc = stk.financials
        bs = stk.balance_sheet
        cf = stk.cash_flow
        if inc.empty or bs.empty or cf.empty:
            raise ValueError("Missing financial statements")
        # Chronological: oldest first
        inc = inc[inc.columns[::-1]]
        bs = bs[bs.columns[::-1]]
        cf = cf[cf.columns[::-1]]
        return info, inc, bs, cf

    # Last non-null
    def _series_last(s):
        if s is None:
            return None
        try:
            s = s.dropna()
            return s.iloc[-1] if not s.empty else None
        except Exception:
            return None

    # Safe average
    def _safe_avg(series, yrs=MIN_YEARS_FOR_AVG, clip=None):
        series = series.dropna()
        if series.empty:
            return None
        sub = series.iloc[-min(len(series), yrs):]
        if clip:
            sub = sub.clip(clip[0], clip[1])
        return sub.mean()

    # Grab series by names
    def _grab(df, names):
        for nm in names:
            if not isinstance(df, pd.DataFrame):
                break
            matches = [i for i in df.index if isinstance(i, str) and i.lower() == nm.lower()]
            if matches:
                ser = pd.to_numeric(df.loc[matches[0]], errors='coerce')
                if not ser.isnull().all():
                    return ser
        return None

    # Risk-free
    def _risk_free():
        try:
            h = yf.Ticker(RISK_FREE_RATE_TICKER).history(period='5d')
            rf = _series_last(h['Close']) / 100
            if rf and rf > 0:
                return rf
        except Exception:
            pass
        warnings_list.append('Fallback risk-free rate 3% used')
        return 0.03

    # Beta
    def _beta(info):
        b = info.get('beta')
        if b is not None:
            return b, 'yfinance'
        try:
            s = yf.Ticker(ticker).history(period='5y', interval='1mo')['Close']
            m = yf.Ticker(MARKET_INDEX_TICKER).history(period='5y', interval='1mo')['Close']
            df = pd.DataFrame({'s': s, 'm': m}).pct_change().dropna()
            cov = np.cov(df['s'], df['m'])
            return cov[0, 1] / cov[1, 1], 'calculated'
        except Exception:
            warnings_list.append('Beta fallback 1.0 used')
            return 1.0, 'default'

    # Tax rate
    def _tax_rate(income):
        tax = _grab(income, ['Tax Provision', 'Income Tax Expense'])
        pbt = _grab(income, ['Pretax Income', 'Income Before Tax'])
        if tax is None or pbt is None:
            warnings_list.append('Tax rate default 21% used')
            return DEFAULT_TAX_RATE, 'default'
        rates = (tax / pbt).clip(0, 0.6)
        r = _safe_avg(rates)
        if r is None:
            warnings_list.append('Tax rate default 21% used')
            return DEFAULT_TAX_RATE, 'default'
        return r, 'historical'

    # Cost of debt with explicit None checks
    def _cost_of_debt(income, bs, rf):
        int_exp = _grab(income, ['Interest Expense'])
        st_ser = _grab(bs, ['Current Debt', 'Short Term Debt'])
        if st_ser is None:
            st = pd.Series(0, index=income.columns)
            warnings_list.append('Short-term debt missing, assumed zero')
        else:
            st = st_ser
        lt_ser = _grab(bs, ['Long Term Debt'])
        if lt_ser is None:
            lt = pd.Series(0, index=income.columns)
            warnings_list.append('Long-term debt missing, assumed zero')
        else:
            lt = lt_ser
        debt_ser = st.add(lt, fill_value=0)
        avg_debt = (debt_ser + debt_ser.shift(1)) / 2
        if int_exp is not None and not avg_debt.dropna().empty:
            rd = _safe_avg((int_exp.abs() / avg_debt).replace([np.inf, -np.inf], np.nan), clip=(0, 0.3))
            if rd:
                return rd, 'interest/debt'
        warnings_list.append('Cost of debt default 5% used')
        return DEFAULT_COST_OF_DEBT, 'default'

    # Historical metrics with explicit None checks
    def _historical_metrics(income, bs, cf, tax_rt):
        ca_ser = _grab(bs, ['Current Assets'])
        if ca_ser is None:
            ca = pd.Series(0, index=income.columns)
            warnings_list.append('Current Assets missing, assumed zero')
        else:
            ca = ca_ser
        cash_ser = _grab(bs, ['Cash And Cash Equivalents'])
        if cash_ser is None:
            cash = pd.Series(0, index=income.columns)
            warnings_list.append('Cash missing, assumed zero')
        else:
            cash = cash_ser
        cl_ser = _grab(bs, ['Current Liabilities'])
        if cl_ser is None:
            cl = pd.Series(0, index=income.columns)
            warnings_list.append('Current Liabilities missing, assumed zero')
        else:
            cl = cl_ser
        std_ser = _grab(bs, ['Current Debt', 'Short Term Debt'])
        if std_ser is None:
            std = pd.Series(0, index=income.columns)
            warnings_list.append('Short-term debt missing, assumed zero')
        else:
            std = std_ser
        rev = _grab(income, ['Total Revenue', 'Revenues'])
        ebit = _grab(income, ['EBIT', 'Operating Income'])
        da_ser = _grab(cf, ['Depreciation And Amortization'])
        da = da_ser if da_ser is not None else pd.Series(0, index=cf.columns)
        capex_ser = _grab(cf, ['Capital Expenditure'])
        capex = capex_ser.abs() if capex_ser is not None else pd.Series(0, index=cf.columns)
        nwc = (ca - cash) - (cl - std)
        delta_nwc = nwc.diff().fillna(0)
        tax_on_ebit = (ebit * tax_rt).fillna(0) if ebit is not None else pd.Series(0, index=income.columns)
        ebiat = (ebit - tax_on_ebit).fillna(0) if ebit is not None else pd.Series(0, index=income.columns)
        ufcf = (ebiat + da - capex - delta_nwc).dropna()
        ratios = {}
        if rev is not None and ebit is not None:
            ratios['ebit_margin'] = (ebit / rev).clip(0, 1).dropna()
        else:
            warnings_list.append('EBIT margin unavailable')
        ratios['da_sales'] = ((da / rev) if rev is not None else pd.Series(0, index=income.columns)).clip(0).dropna()
        ratios['capex_sales'] = ((capex / rev) if rev is not None else pd.Series(0, index=income.columns)).clip(0).dropna()
        ratios['nwc_sales'] = ((nwc / rev) if rev is not None else pd.Series(0, index=income.columns)).dropna()
        avgs = {k: _safe_avg(v) or 0 for k, v in ratios.items()}
        # avoid ambiguous truth value of pandas Series by checking rev explicitly
        revenue_series = rev if rev is not None else pd.Series(dtype=float)
        return {'ufcf': ufcf, 'revenue': revenue_series, 'ratio_avgs': avgs}

    def _calculate_wacc(info, bs, cost_equity, rd, tax_rt):
        market_cap = info.get('marketCap') or 0
        debt = (_series_last(_grab(bs, ['Current Debt', 'Short Term Debt'])) or 0) + \
               (_series_last(_grab(bs, ['Long Term Debt'])) or 0)
        total = market_cap + debt
        if total > 0:
            return (market_cap / total) * cost_equity + (debt / total) * rd * (1 - tax_rt)
        warnings_list.append('WACC fallback to cost of equity')
        return cost_equity

    def _estimate_growth(hist):
        rev = hist['revenue'].dropna()
        if len(rev) >= 2:
            yrs = rev.index[-1].year - rev.index[0].year
            if yrs > 0:
                g = (rev.iloc[-1]/rev.iloc[0])**(1/yrs) - 1
                return min(max(g, GROWTH_FLOOR), GROWTH_CAP)
        warnings_list.append('Growth estimate fallback 5%')
        return 0.05

    def _project_cf(hist, base_growth, yrs1, yrs_tr, tgr, ratios):
        last_year = hist['revenue'].index[-1].year
        last_rev = hist['revenue'].iloc[-1]
        e_m = ratios.get('ebit_margin', 0)
        da_s = ratios.get('da_sales', 0)
        c_s = ratios.get('capex_sales', 0)
        n_s = ratios.get('nwc_sales', 0)
        nwc_lvl = last_rev * n_s
        cashflows = {}
        rates = [base_growth]*yrs1 + list(np.linspace(base_growth, tgr, yrs_tr+1)[1:])
        for i, g in enumerate(rates, 1):
            yr = last_year + i
            rev = last_rev*(1+g)
            ebit = rev*e_m
            tax = ebit*tax_rt
            ebiat = ebit - tax
            da = rev*da_s
            capex = rev*c_s
            nwc = rev*n_s
            d_nwc = nwc - nwc_lvl
            ufcf = ebiat + da - capex - d_nwc
            cashflows[yr] = float(ufcf)
            last_rev, nwc_lvl = rev, nwc
        return cashflows

    def _discount_cf(cf_dict, wacc):
        yrs = sorted(cf_dict)
        return sum(cf_dict[y]/((1+wacc)**(i+0.5)) for i, y in enumerate(yrs))

    def _terminal_value(last_ufcf, wacc, tgr):
        if tgr >= wacc:
            tgr = wacc - 0.001
            warnings_list.append('Terminal growth adjusted below WACC')
        return last_ufcf*(1+tgr)/(wacc-tgr)

    def _generate_sensitivity(cf_dict, bs, shares, wacc, tgr):
        waccs = np.linspace(wacc-0.01, wacc+0.01, 5)
        tgrs = np.linspace(tgr-0.005, tgr+0.005, 5)
        matrix = {}
        for tg in tgrs:
            row = {}
            for wc in waccs:
                if tg >= wc:
                    row[f"{wc:.3%}"] = None
                else:
                    pv = _discount_cf(cf_dict, wc)
                    tv = _terminal_value(list(cf_dict.values())[-1], wc, tg)
                    ev = pv + tv / ((1 + wc) ** len(cf_dict))
                    debt = (_series_last(_grab(bs, ['Current Debt', 'Short Term Debt'])) or 0) + \
                           (_series_last(_grab(bs, ['Long Term Debt'])) or 0)
                    cashbal = _series_last(_grab(bs, ['Cash And Cash Equivalents'])) or 0
                    eq = ev - debt + cashbal
                    row[f"{wc:.3%}"] = eq / shares
            matrix[f"{tg:.3%}"] = row
        return matrix

    def _generate_scenarios(cf, hist, ratios, shares, wacc, tgr):
        mods = {
            'Base': {'g':0,'w':0,'m':0},
            'Optimistic': {'g':0.02,'w':-0.005,'m':0.01},
            'Pessimistic': {'g':-0.02,'w':0.005,'m':-0.01}
        }
        results = {}
        for name, md in mods.items():
            g = min(max(base_growth+md['g'], GROWTH_FLOOR), GROWTH_CAP)
            new_ratios = ratios.copy()
            new_ratios['ebit_margin'] += md['m']
            cf_mod = _project_cf(hist, g, DEFAULT_STAGE1_YEARS, DEFAULT_TRANSITION_YEARS, tgr, new_ratios)
            wc = wacc + md['w']
            pv = _discount_cf(cf_mod, wc)
            tv = _terminal_value(list(cf_mod.values())[-1], wc, tgr)
            ev = pv + tv / ((1 + wc) ** len(cf_mod))
            debt = (_series_last(_grab(bs, ['Current Debt', 'Short Term Debt'])) or 0) + \
                   (_series_last(_grab(bs, ['Long Term Debt'])) or 0)
            cashbal = _series_last(_grab(bs, ['Cash And Cash Equivalents'])) or 0
            eq = ev - debt + cashbal
            results[name] = {'wacc': wc, 'growth_rate': g, 'intrinsic_value_per_share': eq / shares}
        return results

    def _format_summary(tkr, iv, price, ev, eq, wacc, tgr, g):
        diff = (iv-price)/price if price else 0
        if diff>0.15:
            verdict='**Undervalued**'
        elif diff< -0.15:
            verdict='**Overvalued**'
        else:
            verdict='**Fairly Valued**'
        return (f"### {tkr.upper()} DCF Valuation ({datetime.now().strftime('%Y-%m-%d')})\n"
                f"|Metric|Value|\n|---|---|\n"
                f"|Intrinsic/share|${iv:,.2f}|\n"
                f"|Current price|${price:,.2f}|\n"
                f"|Verdict|{verdict}|\n"
                f"|Enterprise value|${ev:,.0f}|\n"
                f"|Equity value|${eq:,.0f}|\n"
                f"|WACC|{wacc:.2%}|\n"
                f"|Stage-1 growth|{g:.2%}|\n"
                f"|Terminal growth|{tgr:.2%}|\n")

    # ------------------- Execute Valuation Flow -------------------
    try:
        info, income, bs, cf = _fetch_statements(ticker)
    except Exception as e:
        return {'error': f'Fetch error: {e}'}

    try:
        price_now = info.get('currentPrice')
        if price_now is None:
            price_hist = yf.Ticker(ticker).history(period='5d')
            price_now = price_hist['Close'].iloc[-1] if not price_hist.empty else None
    except Exception:
        price_now = None
        warnings_list.append("Could not fetch current price.")

    shares = info.get('sharesOutstanding') or 0

    rf = _risk_free()
    beta_val, beta_src = _beta(info)
    cost_equity = rf + beta_val * DEFAULT_ERP
    tax_rt, tax_src = _tax_rate(income)
    rd, rd_src = _cost_of_debt(income, bs, rf)

    # Defensive: always treat _series_last as possibly None, and sum as floats
    def _safe_add(a, b):
        a = a if a is not None else 0.0
        b = b if b is not None else 0.0
        return float(a) + float(b)

    debt = _safe_add(_series_last(_grab(bs, ['Current Debt', 'Short Term Debt'])),
                     _series_last(_grab(bs, ['Long Term Debt'])))
    cashbal = _series_last(_grab(bs, ['Cash And Cash Equivalents']))
    if cashbal is None:
        cashbal = 0.0

    wacc = _calculate_wacc(info, bs, cost_equity, rd, tax_rt)

    hist = _historical_metrics(income, bs, cf, tax_rt)
    if hist['ufcf'] is None or getattr(hist['ufcf'], 'empty', False):
        warnings_list.append('No UFCF history; projections may be off')
        # fallback: create a minimal cash flow projection if possible
        hist['ufcf'] = pd.Series([0.0])
        hist['revenue'] = pd.Series([0.0])
        hist['ratio_avgs'] = {k: 0.0 for k in ['ebit_margin', 'da_sales', 'capex_sales', 'nwc_sales']}
    base_growth = _estimate_growth(hist)
    ratios = hist['ratio_avgs']

    try:
        proj_cf = _project_cf(hist, base_growth, DEFAULT_STAGE1_YEARS,
                              DEFAULT_TRANSITION_YEARS, DEFAULT_TERMINAL_GROWTH_RATE, ratios)
    except Exception:
        proj_cf = {}
        warnings_list.append("Could not project future cash flows due to missing data.")

    try:
        pv_ufcf = _discount_cf(proj_cf, wacc)
        tv = _terminal_value(list(proj_cf.values())[-1], wacc, DEFAULT_TERMINAL_GROWTH_RATE) if proj_cf else 0.0
        pv_tv = tv / ((1+wacc)**len(proj_cf)) if proj_cf else 0.0
        enterprise_val = pv_ufcf + pv_tv
    except Exception:
        enterprise_val = 0.0
        warnings_list.append("Could not compute enterprise value due to missing data.")

    try:
        equity_val = enterprise_val - debt + cashbal
    except Exception:
        equity_val = 0.0
        warnings_list.append("Could not compute equity value due to missing data.")

    try:
        intrinsic_ps = equity_val / shares if shares and shares > 0 else 0.0
    except Exception:
        intrinsic_ps = 0.0
        warnings_list.append("Could not compute intrinsic value per share due to missing data.")

    try:
        sens = _generate_sensitivity(proj_cf, bs, shares, wacc, DEFAULT_TERMINAL_GROWTH_RATE)
    except Exception:
        sens = {}
        warnings_list.append("Could not generate sensitivity matrix due to missing data.")

    try:
        scen = _generate_scenarios(proj_cf, hist, ratios, shares, wacc, DEFAULT_TERMINAL_GROWTH_RATE)
    except Exception:
        scen = {}
        warnings_list.append("Could not generate scenario analysis due to missing data.")

    assumptions = {
        'risk_free_rate': rf,
        'beta': {'value': beta_val, 'source': beta_src},
        'equity_risk_premium': DEFAULT_ERP,
        'cost_of_equity': cost_equity,
        'cost_of_debt': {'value': rd, 'source': rd_src},
        'tax_rate': {'value': tax_rt, 'source': tax_src},
        'stage1_years': DEFAULT_STAGE1_YEARS,
        'transition_years': DEFAULT_TRANSITION_YEARS,
        'stage1_growth': base_growth,
        'terminal_growth_rate': DEFAULT_TERMINAL_GROWTH_RATE
    }
    summary_md = _format_summary(
        ticker, intrinsic_ps, price_now, enterprise_val,
        equity_val, wacc, DEFAULT_TERMINAL_GROWTH_RATE, base_growth
    )

    # Compose output dictionary with clear, non-redundant, well-labelled fields
    def _format_currency(val, decimals=2):
        try:
            return f"${val:,.{decimals}f}"
        except Exception:
            return str(val)

    def _format_percent(val, decimals=2):
        try:
            return f"{val*100:.{decimals}f}%"
        except Exception:
            return str(val)

    # Format projected cash flows with year and USD
    formatted_proj_cf = {
        f"Year {year} Projected UFCF (USD)": _format_currency(cf, 0) for year, cf in proj_cf.items()
    } if proj_cf else {}

    # Format sensitivity matrix with clear units and descriptive keys
    formatted_sens = {}
    for tgr, row in sens.items():
        formatted_row = {}
        for wacc_key, val in row.items():
            formatted_row[f"Equity Value/Share at WACC {wacc_key}"] = (
                _format_currency(val, 2) if val is not None else None
            )
        formatted_sens[f"Terminal Growth Rate {tgr}"] = formatted_row

    # Format scenario analysis with clear keys and units
    formatted_scen = {}
    for scen_name, scen_vals in scen.items():
        formatted_scen[scen_name] = {
            "Scenario WACC (%)": _format_percent(scen_vals.get("wacc", 0.0), 2),
            "Scenario Stage-1 Growth Rate (%)": _format_percent(scen_vals.get("growth_rate", 0.0), 2),
            "Scenario Intrinsic Value per Share (USD)": _format_currency(scen_vals.get("intrinsic_value_per_share", 0.0), 2)
            if scen_vals.get("intrinsic_value_per_share", None) is not None else None,
        }

    # Format assumptions for clarity and units
    formatted_assumptions = {
        "risk_free_rate (%)": _format_percent(assumptions["risk_free_rate"], 3) if isinstance(assumptions["risk_free_rate"], (float, np.floating)) else assumptions["risk_free_rate"],
        "beta": assumptions["beta"],
        "equity_risk_premium (%)": _format_percent(assumptions["equity_risk_premium"], 2) if isinstance(assumptions["equity_risk_premium"], (float, np.floating)) else assumptions["equity_risk_premium"],
        "cost_of_equity (%)": _format_percent(assumptions["cost_of_equity"], 3) if isinstance(assumptions["cost_of_equity"], (float, np.floating)) else assumptions["cost_of_equity"],
        "cost_of_debt (%)": {
            "value": _format_percent(assumptions["cost_of_debt"]["value"], 3) if isinstance(assumptions["cost_of_debt"]["value"], (float, np.floating)) else assumptions["cost_of_debt"]["value"],
            "source": assumptions["cost_of_debt"]["source"]
        },
        "tax_rate (%)": {
            "value": _format_percent(assumptions["tax_rate"]["value"], 2) if isinstance(assumptions["tax_rate"]["value"], (float, np.floating)) else assumptions["tax_rate"]["value"],
            "source": assumptions["tax_rate"]["source"]
        },
        "stage1_years": assumptions["stage1_years"],
        "transition_years": assumptions["transition_years"],
        "stage1_growth (%)": _format_percent(assumptions["stage1_growth"], 2) if isinstance(assumptions["stage1_growth"], (float, np.floating)) else assumptions["stage1_growth"],
        "terminal_growth_rate (%)": _format_percent(assumptions["terminal_growth_rate"], 2) if isinstance(assumptions["terminal_growth_rate"], (float, np.floating)) else assumptions["terminal_growth_rate"],
    }

    output = {
        'valuation_summary': {
            'intrinsic_value_per_share (USD)': _format_currency(intrinsic_ps, 2),
            'enterprise_value (USD)': _format_currency(enterprise_val, 0),
            'equity_value (USD)': _format_currency(equity_val, 0),
            'current_market_price (USD)': _format_currency(price_now, 2) if price_now is not None else None,
            'wacc (%)': _format_percent(wacc, 2),
            'terminal_growth_rate (%)': _format_percent(DEFAULT_TERMINAL_GROWTH_RATE, 2),
            'summary_markdown': summary_md
        },
        'assumptions_used': formatted_assumptions,
        'projected_cash_flows_by_year': formatted_proj_cf,
        'sensitivity_matrix_equity_value_per_share': formatted_sens,
        'scenario_analysis_intrinsic_value_per_share': formatted_scen,
        'warnings_and_fallbacks': list(dict.fromkeys(warnings_list + _mem_handler.records))  # deduplicate warnings
    }
    return output




if __name__ == "__main__":
    ticker = "NVDA"
    result = run_intrinsic_valuation(ticker)
    print(result)
