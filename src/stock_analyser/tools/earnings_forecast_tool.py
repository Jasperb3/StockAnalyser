# earnings_forecast_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class EarningsForecastInput(BaseModel):
    """
    Input schema for EarningsForecastTool.
    Provide a single stock ticker symbol.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")

class EarningsForecastTool(BaseTool):
    name: str = "Earnings Forecast Tool"
    description: str = "Retrieves the earnings forecast for a given stock ticker."
    args_schema: Type[BaseModel] = EarningsForecastInput

    def _run(self, ticker: str) -> str:
        """
        Run method for the EarningsForecastTool.
        Retrieves earnings forecast data for the ticker from nasdaq-data.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_data(ticker, 4)  # 4 => Earnings Forecast
        return df.to_json(orient="records")