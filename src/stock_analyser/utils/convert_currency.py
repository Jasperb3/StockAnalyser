import yfinance as yf

def convert_currency(stock: str) -> float:

    stock = yf.Ticker(stock)

    currency = stock.info.get('financialCurrency')
            
    if currency == "USD":
        exchange_rate = 1
    else:
        try:
            exchange_rate_symbol = yf.Ticker(f"{currency}USD=X")
            exchange_rate = exchange_rate_symbol.history(period='1d')['Close'].iloc[0]
        except Exception as e:
            print(f"Error fetching exchange rate for {currency}: {e}")
            exchange_rate = 1

    return exchange_rate