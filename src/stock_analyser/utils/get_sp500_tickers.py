import io
import requests
import pandas as pd

def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
    response.raise_for_status()
    sp500 = pd.read_html(io.StringIO(response.text))[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    sp500_tickers = sp500['Symbol'].to_list()
    return sp500_tickers



if __name__ == "__main__":
    sp500_tickers = get_sp500_tickers()
    print(sp500_tickers)













