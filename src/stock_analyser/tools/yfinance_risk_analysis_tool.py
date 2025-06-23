from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Type, Dict, Any
import numpy as np

# Define the input schema using Pydantic
class YFinanceRiskAnalysisToolInput(BaseModel):
    """Input schema for YFinanceIncomeTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")
    benchmark: str = Field("^GSPC", description="Benchmark index for comparison (default: S&P 500).")
    period: int = Field(1, description="Time period to fetch income statement for. Default is 5 years.")

# Define the tool class
class YFinanceRiskAnalysisTool(BaseTool):
    name: str = "YFinance Income Tool"
    description: str = "Fetches detailed income statement for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceRiskAnalysisToolInput

    def _run(self, ticker: str, benchmark: str = "^GSPC", period: str = "5y") -> Dict[str, Any]:
        """
        Perform risk assessment for a given stock.
        
        Args:
            ticker (str): The stock ticker symbol.
            benchmark (str): Benchmark index for comparison (default: S&P 500).
            period (str): Time period for analysis.
        
        Returns:
            dict: Risk assessment results.
        """
        stock = yf.Ticker(ticker.upper())
        benchmark_index = yf.Ticker(benchmark)
        
        stock_data = stock.history(period=period)['Close']
        benchmark_data = benchmark_index.history(period=period)['Close']
        
        # Calculate returns
        stock_returns = stock_data.pct_change().dropna()
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        # Calculate beta
        covariance = np.cov(stock_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = stock_returns - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(stock_returns, 5)
        
        # Calculate Maximum Drawdown
        cumulative_returns = (1 + stock_returns).cumprod()
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
        
        return {
            "ticker": ticker,
            "beta": f"{beta:.3f}",
            "sharpe_ratio": F"{sharpe_ratio:.3f}",
            "value_at_risk_95": f"{var_95:.3f}",
            "max_drawdown": f"{max_drawdown:.3f}",
            "volatility": f"{stock_returns.std() * np.sqrt(252):.3f}"
        }


if __name__ == "__main__":
    # Example usage
    tool = YFinanceRiskAnalysisTool()
    result = tool._run(ticker="nvda", benchmark="^GSPC", period="5y")
    print(result)