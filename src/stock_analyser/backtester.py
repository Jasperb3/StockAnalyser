import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000, commission: float = 0.0, slippage: float = 0.001, position_sizing: str = 'fixed'):
        """
        Initialize the Backtester with historical data and parameters for backtesting.

        Parameters:
            df (pd.DataFrame): DataFrame containing at least 'Close' and 'signal' columns.
            initial_capital (float): Starting capital for the simulation.
            commission (float): Commission per trade (applied as a fraction of trade value).
            slippage (float): Estimated slippage as a fraction (e.g., 0.001 for 0.1%).
            position_sizing (str): Position sizing strategy. Currently, only 'fixed' is supported.
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.results = {}
        self.trades = []

    def run_backtest(self) -> dict:
        """
        Run the backtest simulation based on the signals in self.df and calculate performance metrics.
        Returns:
            dict: A dictionary containing performance metrics and trade details.
        """
        df = self.df.copy()
        if 'signal' not in df.columns:
            raise ValueError('DataFrame must have a "signal" column.')

        # Lagging the signal to avoid lookahead bias
        df['position'] = df['signal'].shift(1).fillna(0)
        df['daily_return'] = df['Close'].pct_change().fillna(0)
        df['strategy_return'] = df['position'] * df['daily_return']
        df['equity_curve'] = (1 + df['strategy_return']).cumprod() * self.initial_capital

        # Performance metrics
        total_return = df['equity_curve'].iloc[-1] / self.initial_capital - 1
        num_days = (df.index[-1] - df.index[0]).days if isinstance(df.index[0], pd.Timestamp) else len(df)
        annualized_return = (df['equity_curve'].iloc[-1] / self.initial_capital) ** (365/num_days) - 1 if num_days > 0 else np.nan

        daily_strategy = df['strategy_return']
        sharpe_ratio = (daily_strategy.mean() / (daily_strategy.std() + 1e-5)) * np.sqrt(252)
        negative_returns = daily_strategy[daily_strategy < 0]
        sortino_ratio = (daily_strategy.mean() / (negative_returns.std() + 1e-5)) * np.sqrt(252)

        running_max = df['equity_curve'].cummax()
        drawdown = (df['equity_curve'] / running_max) - 1
        max_drawdown = drawdown.min()

        # Extract trade details with realistic price adjustments
        trades = self.extract_trades(df)

        # Calculate trade performance metrics
        win_rate = np.nan
        avg_win = np.nan
        avg_loss = np.nan
        if trades:
            wins = [t['return'] for t in trades if t['return'] > 0]
            losses = [t['return'] for t in trades if t['return'] <= 0]
            win_rate = len(wins) / len(trades) if trades else np.nan
            avg_win = np.mean(wins) if wins else np.nan
            avg_loss = np.mean(losses) if losses else np.nan

        self.results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'number_of_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades
        }

        return self.results

    def extract_trades(self, df: pd.DataFrame) -> list:
        """
        Extract trade details from the signal transitions in the DataFrame and adjust entry/exit prices for commission and slippage.
        Returns:
            list: A list of dictionaries, each representing a trade.
        """
        trades = []
        current_trade = None
        for i in range(1, len(df)):
            prev_signal = df['signal'].iloc[i - 1]
            curr_signal = df['signal'].iloc[i]
            price = df['Close'].iloc[i]
            date = df.index[i]

            # Check if there is a change in signal
            if prev_signal != curr_signal:
                # If exiting a trade
                if prev_signal != 0 and current_trade is not None:
                    # Adjust exit price for slippage and commission
                    if current_trade['position'] == 1:
                        exit_adjusted = price * (1 - self.slippage - self.commission)
                    else:
                        exit_adjusted = price * (1 + self.slippage + self.commission)
                    current_trade['exit_date'] = date
                    current_trade['exit_price'] = exit_adjusted
                    # Adjust entry price similarly (stored during trade entry)
                    entry_price = current_trade['entry_price']
                    if current_trade['position'] == 1:
                        trade_return = (exit_adjusted - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - exit_adjusted) / entry_price
                    current_trade['return'] = trade_return
                    trades.append(current_trade)
                    current_trade = None
                # If entering a new trade
                if curr_signal != 0:
                    # Adjust entry price for slippage and commission
                    if curr_signal == 1:
                        entry_adjusted = price * (1 + self.slippage + self.commission)
                    else:
                        entry_adjusted = price * (1 - self.slippage - self.commission)
                    current_trade = {
                        'entry_date': date,
                        'entry_price': entry_adjusted,
                        'position': curr_signal
                    }
        # Close any open trade at the end
        if current_trade is not None:
            price = df['Close'].iloc[-1]
            if current_trade['position'] == 1:
                exit_adjusted = price * (1 - self.slippage - self.commission)
            else:
                exit_adjusted = price * (1 + self.slippage + self.commission)
            current_trade['exit_date'] = df.index[-1]
            current_trade['exit_price'] = exit_adjusted
            entry_price = current_trade['entry_price']
            if current_trade['position'] == 1:
                trade_return = (exit_adjusted - entry_price) / entry_price
            else:
                trade_return = (entry_price - exit_adjusted) / entry_price
            current_trade['return'] = trade_return
            trades.append(current_trade)

        return trades

    def integrate_external_data(self, external_df: pd.DataFrame, on: str = 'Date') -> None:
        """
        Integrate external data by merging based on a common column (e.g., Date).
        This method allows the incorporation of additional data such as volume or sentiment.
        """
        self.df = pd.merge(self.df, external_df, on=on, how='left')

    def get_signal_details(self) -> pd.DataFrame:
        """
        Return detailed signal information including contributions from different components and risk metrics.
        This includes the computed momentum, path efficiency, resistance, volatility, regime, and a suggested stop-loss level.
        The stop-loss is estimated as Close * (1 - volatility_multiplier * volatility).
        """
        df_details = self.df.copy()
        vol_mult = self.parameters.get('volatility_multiplier', 1.5)
        df_details['stop_loss'] = df_details['Close'] * (1 - vol_mult * df_details.get('volatility', 0))
        # In a real-world case, more detailed reasoning could be added to explain the signal.
        return df_details[['signal', 'momentum', 'path_efficiency', 'resistance', 'volatility', 'regime', 'stop_loss']]

    def plot_performance(self, show: bool = True) -> None:
        """
        Plot key performance metrics including cumulative equity curve and drawdown.
        This enhanced visualization provides insight into the strategy's performance.
        
        Parameters:
            show (bool): Whether to display the plot immediately.
        """
        import matplotlib.pyplot as plt
        
        df = self.df.copy()
        if 'equity_curve' not in df.columns:
            # Ensure the backtest is run first
            self.run_backtest()
            df = self.df.copy()
        
        # Recalculate equity curve
        df['position'] = df['signal'].shift(1).fillna(0)
        df['daily_return'] = df['Close'].pct_change().fillna(0)
        df['strategy_return'] = df['position'] * df['daily_return']
        df['equity_curve'] = (1 + df['strategy_return']).cumprod() * self.initial_capital
        running_max = df['equity_curve'].cummax()
        drawdown = (df['equity_curve'] / running_max) - 1
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax[0].plot(df.index, df['equity_curve'], label='Equity Curve')
        ax[0].set_ylabel('Equity')
        ax[0].set_title('Cumulative Returns')
        ax[0].legend()
        
        ax[1].plot(df.index, drawdown, label='Drawdown', color='red')
        ax[1].set_ylabel('Drawdown')
        ax[1].set_title('Drawdown')
        ax[1].legend()
        
        plt.xlabel('Time')
        plt.tight_layout()
        if show:
            plt.show()