import os
import time
from stock_analyser.utils.models import SupportResistanceChart, SqueezeMomentumChart, SupertrendChart, MACDStochasticChart
from glob import glob
from pprint import pprint

from google import genai

api_key = os.getenv('GOOGLE_API_KEY')

client = genai.Client(api_key=api_key)

model_id =  "gemini-2.0-flash"


def describe_chart(prompt, image_path, model, sleep_time=4):
    start = time.perf_counter()
    response = client.models.generate_content(
    model=model_id,
    contents=[prompt, image_path],
    config={'response_mime_type': 'application/json', 'response_schema': model}
    )
    end = time.perf_counter()
    duration = end - start
    time.sleep(sleep_time - duration if duration < sleep_time else 0)
    return response.parsed



def get_model(image_path):
    if 'breakout' in image_path:
        return SupportResistanceChart
    elif 'macd' in image_path:
        return MACDStochasticChart
    elif 'squeeze_momentum' in image_path:
        return SqueezeMomentumChart
    elif 'supertrend' or 'backtest' in image_path:
        return SupertrendChart
    else:
        raise ValueError(f"No model found for image path: {image_path}")
    


def get_chart_details(image_path):
    if 'breakout' in image_path:
        chart_details = """
This chart identifies key price levels at which a stock historically experienced difficulty moving beyond (resistance) or found consistent buying support.
The green dashed lines show support levels, indicating prices where downward movements historically paused or reversed, while red dashed lines mark resistance levels, highlighting past peaks or price ceilings.
Traders watch these levels closely, interpreting a breakout—when the price moves convincingly beyond resistance or support—as a potential signal of significant future price movement.
The support and resistance lines are algorithmically determined based on historical price patterns, either via fractal candlestick patterns or window-shifting methods as described in the supporting analysis code.
"""
    elif 'macd' in image_path:
        chart_details = """
This comprehensive chart combines candlestick price action, moving averages (short-term and medium-term), trading volume, MACD (Moving Average Convergence Divergence), and the Stochastic Oscillator to provide insights into market momentum, price direction, and potential reversals.
The MACD subplot shows momentum through a histogram (green bars indicate bullish momentum, red indicate bearish momentum), alongside two lines—the MACD line (grey) and the signal line (blue)—where crossovers between these lines signal possible trend changes.
The Stochastic subplot at the bottom highlights overbought (above 80) and oversold (below 20) conditions, with crossovers between the %D (blue) and %SD (orange) lines providing potential entry and exit points.
Traders typically use these signals together with price action and volume to confirm trade opportunities.
"""
    elif 'squeeze_momentum' in image_path:
        chart_details = """
This chart visualizes periods of market consolidation ("squeeze") and subsequent momentum-driven breakouts, using Bollinger Bands (blue dashed lines) and Keltner Channels (red dashed lines).
When the Bollinger Bands move inside the Keltner Channels (marked by black X's), the market is considered in a "squeeze," indicating low volatility and potential upcoming significant price moves.
Momentum bars beneath the price chart show bullish (green/lime) or bearish (red/maroon) momentum, with bright colors indicating increasing momentum and darker colors decreasing momentum.
Traders typically look for the transition from squeeze (black X) to trending (grey X) accompanied by strong momentum bars to signal potential entry or exit points.
"""
    elif 'supertrend' or 'backtest' in image_path:
        chart_details = """
These charts show the Supertrend indicator—a volatility-based trend-following tool that visually indicates whether a stock is in an upward (green support line) or downward (red resistance line) trend.
The Supertrend calculation relies on the Average True Range (ATR) of price movements to adapt dynamically to volatility changes. Buy signals (green upward triangles) and sell signals (red downward triangles) appear when the stock price crosses above or below these Supertrend lines, respectively.
The backtest chart includes historical trading signals, clearly marking hypothetical buy and sell points, and provides a summary box showing the strategy's past performance, including return on investment (ROI) and total trades.
"""
    else:
        raise ValueError(f"No model found for image path: {image_path}")
    
    return chart_details




images = glob("/home/j/ai/crewAI/finance/stock_analyser/plots/for_description/*")
print(f"Images for description: {images}\n")

for image_path in images:
    model = get_model(image_path)
    prompt =f"""
You are an expert in reading stock charts.
Your job is to extract all relevant information from a stock chart that will help understand the stock's position and movement.
You are going to see a chart. Focus on the details of the chart - it is important that you read the chart correctly.
{get_chart_details(image_path)}
Extract information as per the response_schema model.
"""
    chart_details = get_chart_details(image_path)
    response_text = describe_chart(prompt, image_path, model)
    print(f"Image = {image_path}\n")
    print(f"Prompt = {prompt}\n")
    print(f"Model = {model}\n")
    pprint(response_text.model_dump_json())
    print("\n\n")





