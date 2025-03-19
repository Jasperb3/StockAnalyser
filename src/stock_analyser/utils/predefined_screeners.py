import yfinance as yf


predefined_screeners = {
    '1': 'aggressive_small_caps',
    '2': 'day_gainers',
    '3': 'day_losers',
    '4': 'growth_technology_stocks',
    '5': 'most_actives',
    '6': 'most_shorted_stocks',
    '7': 'small_cap_gainers',
    '8': 'undervalued_growth_stocks',
    '9': 'undervalued_large_caps',
    '10': 'conservative_foreign_funds',
    '11': 'high_yield_bond',
    '12': 'portfolio_anchors',
    '13': 'solid_large_growth_funds',
    '14': 'solid_midcap_growth_funds',
    '15': 'top_mutual_funds'
}

def make_predefined_query(index: str):
    screen = predefined_screeners[index]
    query = yf.PREDEFINED_SCREENER_QUERIES[screen]
    print(screen.replace('_', ' ').title())

    response = yf.screen(query['query'])
    quotes = response.get('quotes')[:250]
    sorted_quotes = sorted(quotes, key=lambda x: x.get('ask', '0'))

    for quote in sorted_quotes:
        name = quote.get('displayName') if quote.get('displayName') else quote.get('symbol')
        price = quote.get('ask')
        av_rating = quote.get('averageAnalystRating')
        print(f"{name} - ${price} - av.rating: {av_rating}")
    return quotes




if __name__ == "__main__":
    query = make_predefined_query('8')
    # print(query)