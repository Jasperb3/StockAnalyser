import pandas as pd

def get_sp500_tickers():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    sp500_tickers = sp500['Symbol'].to_list()
    return sp500_tickers



if __name__ == "__main__":
    sp500_tickers = get_sp500_tickers()
    print(sp500_tickers)













