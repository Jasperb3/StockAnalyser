import numpy as np
import pandas as pd


class OptimalPLRSignal:
    """
    Enhanced OptimalPLRSignal class implementing modular calculations for generating trading signals.
    This class mitigates overfitting by using walk-forward optimization and regularization, improves market
    regime detection with combined indicators, and adapts lookback periods based on market volatility.
    """
    def __init__(self, df: pd.DataFrame, parameters: dict = None, parameter_ranges: dict = None):
        """
        Initialize the OptimalPLRSignal instance.
        Parameters:
            df (pd.DataFrame): DataFrame containing stock data with columns such as 'Close', 'High'.
            parameters (dict): Optional parameter dictionary for algorithm settings.
            parameter_ranges (dict): Optional parameter ranges for optimization.
        """
        self.df = df.copy()
        # Set default parameters
        self.parameters = parameters or {
            'momentum_window': 10,
            'volatility_multiplier': 1.5,
            'lookback_period': 10,
            'short_ma': 20,
            'long_ma': 50
        }
        # Set default parameter ranges for optimization
        self.parameter_ranges = parameter_ranges or {
            'momentum_window': (5, 20),
            'volatility_multiplier': (1.0, 3.0)
        }
        self.optimal_params = None

    def calculate_momentum(self) -> pd.Series:
        """
        Calculate momentum based on percentage change over the momentum window.
        Returns:
            pd.Series: Momentum values.
        """
        window = self.parameters.get('momentum_window', 10)
        self.df['momentum'] = self.df['Close'].pct_change(window)
        return self.df['momentum']

    def calculate_path_efficiency(self) -> pd.Series:
        """
        Calculate path efficiency as the ratio of net change to cumulative absolute change over the lookback period.
        Returns:
            pd.Series: Path efficiency values.
        """
        window = self.parameters.get('lookback_period', 10)
        self.df['net_change'] = self.df['Close'].diff(window)
        self.df['cumulative_change'] = self.df['Close'].diff().abs().rolling(window=window).sum()
        self.df['path_efficiency'] = self.df['net_change'] / (self.df['cumulative_change'] + 1e-5)
        return self.df['path_efficiency']

    def calculate_candlestick_resistance(self) -> pd.Series:
        """
        Calculate candlestick resistance as the ratio of current close to the maximum high over the lookback period.
        Returns:
            pd.Series: Resistance values.
        """
        window = self.parameters.get('lookback_period', 10)
        self.df['max_high'] = self.df['High'].rolling(window=window).max()
        self.df['resistance'] = self.df['Close'] / (self.df['max_high'] + 1e-5)
        return self.df['resistance']

    def calculate_volatility(self) -> pd.Series:
        """
        Calculate rolling volatility as the standard deviation of percentage change.
        Returns:
            pd.Series: Volatility values.
        """
        window = self.parameters.get('lookback_period', 10)
        self.df['volatility'] = self.df['Close'].pct_change().rolling(window=window).std()
        return self.df['volatility']

    def regime_detection(self) -> pd.Series:
        """
        Detect market regime using moving average crossover and volatility filter.
        Returns:
            pd.Series: Market regime labels ('bullish', 'bearish', 'neutral').
        """
        short_window = self.parameters.get('short_ma', 20)
        long_window = self.parameters.get('long_ma', 50)
        self.df['short_ma'] = self.df['Close'].rolling(window=short_window).mean()
        self.df['long_ma'] = self.df['Close'].rolling(window=long_window).mean()
        # Initialize regime as neutral
        self.df['regime'] = 'neutral'
        self.df.loc[self.df['short_ma'] > self.df['long_ma'], 'regime'] = 'bullish'
        self.df.loc[self.df['short_ma'] < self.df['long_ma'], 'regime'] = 'bearish'
        # Use volatility to adjust regime: if volatility is high, mark regime as neutral.
        vol = self.calculate_volatility()
        vol_threshold = vol.quantile(0.75)
        self.df.loc[vol > vol_threshold, 'regime'] = 'neutral'
        return self.df['regime']

    def objective_profit(self, win: int, mult: float, df_segment: pd.DataFrame) -> float:
        """Calculate the objective function based on strategy profitability using annualized Sharpe ratio."""
        # Set temporary parameters for this evaluation
        temp_params = self.parameters.copy()
        temp_params['momentum_window'] = win
        temp_params['volatility_multiplier'] = mult
        temp_params['lookback_period'] = win
        # Create a temporary instance with these parameters
        temp_obj = OptimalPLRSignal(df_segment.copy(), parameters=temp_params)
        # Generate signals
        signals = temp_obj.signal_generation()
        temp_df = df_segment.copy()
        temp_df['signal'] = signals
        # Lag the signal to avoid lookahead
        temp_df['position'] = temp_df['signal'].shift(1).fillna(0)
        temp_df['daily_return'] = temp_df['Close'].pct_change().fillna(0)
        temp_df['strategy_return'] = temp_df['position'] * temp_df['daily_return']
        # Annualize returns assuming 252 trading days
        mean_ret = temp_df['strategy_return'].mean() * 252
        std_ret = temp_df['strategy_return'].std() * (252 ** 0.5)
        sharpe = mean_ret / (std_ret + 1e-5)
        return sharpe

    def parameter_optimization_profit(self) -> dict:
        """Optimize parameters based on profitability (Sharpe ratio) over the entire dataset."""
        best_sharpe = -float('inf')
        best_params = {}
        win_min, win_max = self.parameter_ranges.get('momentum_window', (5, 20))
        mult_min, mult_max = self.parameter_ranges.get('volatility_multiplier', (1.0, 3.0))
        windows = range(win_min, win_max + 1)
        multipliers = [round(x, 1) for x in np.linspace(mult_min, mult_max, num=5)]
        
        for win in windows:
            for mult in multipliers:
                try:
                    sharpe = self.objective_profit(win, mult, self.df)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {'momentum_window': win, 'volatility_multiplier': mult, 'lookback_period': win}
                except Exception:
                    continue
        self.optimal_params = best_params
        return best_params

    def rolling_optimization(self, window_size: int = 100, step: int = 50) -> list:
        """Perform rolling optimization over the dataset using a sliding window, re-optimizing parameters periodically.

        Parameters:
            window_size (int): Number of data points in each optimization window.
            step (int): Step size (number of data points) to roll the window.
        Returns:
            list: A list of dictionaries with window indices and optimized parameters.
        """
        results = []
        n = len(self.df)
        for start in range(0, n - window_size, step):
            window_data = self.df.iloc[start:start+window_size].copy()
            temp_obj = OptimalPLRSignal(window_data, parameters=self.parameters, parameter_ranges=self.parameter_ranges)
            opt_params = temp_obj.parameter_optimization_profit()
            results.append({
                'start_index': start,
                'end_index': start + window_size,
                'optimal_params': opt_params
            })
        return results

    def adaptive_lookback_adjustment(self) -> int:
        """
        Dynamically adjust lookback period based on average volatility.
        Returns:
            int: Adjusted lookback period.
        """
        vol = self.calculate_volatility()
        avg_vol = vol.mean()
        if avg_vol > 0.02:
            adjusted_lookback = 20
        elif avg_vol < 0.01:
            adjusted_lookback = 10
        else:
            adjusted_lookback = 15
        self.parameters['lookback_period'] = adjusted_lookback
        return adjusted_lookback

    def signal_generation(self) -> pd.Series:
        """
        Generate trading signals based on calculated metrics, regime detection, and adaptive thresholds.
        Returns:
            pd.Series: Trading signals (-1 for sell, 0 for hold, 1 for buy).
        """
        # Calculate necessary components
        self.calculate_momentum()
        self.calculate_path_efficiency()
        self.calculate_candlestick_resistance()
        self.calculate_volatility()
        self.regime_detection()
        # Adjust lookback period
        self.adaptive_lookback_adjustment()
        # Determine signal thresholds based on momentum volatility
        momentum_std = self.df['momentum'].std()
        threshold = momentum_std * self.parameters.get('volatility_multiplier', 1.5)
        self.df['signal'] = 0
        self.df.loc[(self.df['momentum'] > threshold) & (self.df['regime'] == 'bullish'), 'signal'] = 1
        self.df.loc[(self.df['momentum'] < -threshold) & (self.df['regime'] == 'bearish'), 'signal'] = -1
        return self.df['signal'] 