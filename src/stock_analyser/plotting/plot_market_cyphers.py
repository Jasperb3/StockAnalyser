import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib
from scipy.signal import find_peaks
from datetime import datetime


# --- Parameters ---
# Data
ticker = 'NVDA'
start_date = '2024-01-01'
end_date = None # None for current date
interval = '4h' # yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

# WaveTrend (WT)
wt_channel_len = 9   # n1 / wtChannelLen
wt_average_len = 12  # n2 / wtAverageLen
wt_ma_len = 3        # sp / wtMALen
wt_ob_level1 = 53
wt_ob_level2 = 60
wt_os_level1 = -53
wt_os_level2 = -60
wt_os_level3 = -80 # Used for Gold Circle (like osLevel3 in script 1, or derived from logic in script 2)

# RSI+MFI
rsi_mfi_period = 60
rsi_mfi_multiplier = 175 # From script 2 (more common default)
rsi_mfi_pos_y = 0 # Shift MFI area vertically if desired (like 2.5 in script 2)

# RSI
rsi_len = 14
rsi_ob_level = 60 # Note: PineScript had reversed logic in var names
rsi_os_level = 30 # Note: PineScript had reversed logic in var names

# Divergence WT
wt_show_div = True
wt_div_ob_level = 45 # WT Bearish Divergence min (from script 2)
wt_div_os_level = -65 # WT Bullish Divergence min (from script 2)
divergence_lookback = 60 # How far back to look for divergence pairs
min_divergence_separation = 5 # Minimum bars between divergence points

# Plotting
plot_rsi = False
plot_stoch = False # Stochastics not fully implemented here, focus on WT/MFI/RSI
plot_vwap_osc = True
plot_mfi_area = True


# --- Functions ---

def get_data(ticker, start_date, end_date, interval):
    """Fetches historical data using yfinance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data fetched. Check ticker, dates, and interval.")
        # Ensure column names are consistent (yfinance uses Title Case)
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        # Calculate hlc3 if needed later
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df.dropna(inplace=True) # Drop rows with missing values
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculates WT, RSI, MFI Area, and basic signals."""
    # Safety check for empty dataframe
    if df.empty:
        print("Error: Empty dataframe passed to calculate_indicators")
        return df
        
    # Ensure data types are float for TA-Lib calculations and reshape to 1D
    close = df['close'].values.astype(float).flatten()
    high = df['high'].values.astype(float).flatten()
    low = df['low'].values.astype(float).flatten()
    open_ = df['open'].values.astype(float).flatten()
    hlc3 = df['hlc3'].values.astype(float).flatten()
    
    # Check for valid inputs (non-empty arrays with no NaN/inf values)
    if len(close) == 0 or np.isnan(close).any() or np.isinf(close).any():
        print("Error: Invalid data in close array")
        return df

    # WaveTrend Calculation using TA-Lib
    df['ap'] = hlc3 # Keep the column for reference if needed
    
    try:
        # Use .values to pass numpy arrays to TA-Lib
        esa_np = talib.EMA(hlc3, timeperiod=wt_channel_len)
        d_np = talib.EMA(np.abs(hlc3 - esa_np), timeperiod=wt_channel_len)
        df['esa'] = esa_np # Assign back to DataFrame
        df['d'] = d_np    # Assign back to DataFrame

        # Handle potential division by zero in ci calculation
        # Note: d_np might have NaNs at the beginning
        ci_denominator = 0.015 * d_np
        # Use np.where for safe division, replacing 0 denominator with NaN
        ci_np = np.where(ci_denominator != 0, (hlc3 - esa_np) / ci_denominator, np.nan)

        df['ci'] = ci_np # Assign back to DataFrame

        # Calculate wt1 and wt2 using TA-Lib
        # Ensure input to EMA/SMA is float and handle potential NaNs from ci_np
        wt1_np = talib.EMA(ci_np, timeperiod=wt_average_len)
        # Ensure wt1_np is float before passing to SMA
        wt2_np = talib.SMA(wt1_np, timeperiod=wt_ma_len)

        df['wt1'] = wt1_np
        df['wt2'] = wt2_np
        df['vwap_osc'] = df['wt1'] - df['wt2'] # VWAP Oscillator (Pandas operation)

        # RSI Calculation using TA-Lib - This is where the error happens
        try:
            df['rsi'] = talib.RSI(close, timeperiod=rsi_len)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            print(f"Close array shape: {close.shape}, contains NaN: {np.isnan(close).any()}")
            # Fallback: Set RSI column to NaN
            df['rsi'] = np.nan

        # RSI + MFI Area Calculation using TA-Lib for SMA
        # Handle potential division by zero if high == low
        candle_range = high - low
        # Avoid division by zero using np.where
        candle_value_np = np.where(candle_range != 0, (close - open_) / candle_range, np.nan)
        df['candle_value'] = candle_value_np # Assign back if needed

        mfi_input_np = (candle_value_np * rsi_mfi_multiplier)

        # Calculate SMA using TA-Lib
        mvc_np = talib.SMA(mfi_input_np, timeperiod=rsi_mfi_period)
        df['mvc'] = mvc_np - rsi_mfi_pos_y
        df['mfi_color'] = np.where(df['mvc'] > 0, 'green', 'red') # Pandas/Numpy operation

        # WT Crossings
        df['wt_cross_up'] = (df['wt1'].shift(1) < df['wt2'].shift(1)) & (df['wt1'] > df['wt2'])
        df['wt_cross_down'] = (df['wt1'].shift(1) > df['wt2'].shift(1)) & (df['wt1'] < df['wt2'])
        df['wt_cross'] = df['wt_cross_up'] | df['wt_cross_down']

        # WT Overbought/Oversold Conditions
        df['wt_oversold'] = df['wt2'] < wt_os_level1
        df['wt_overbought'] = df['wt2'] > wt_ob_level1
        df['wt_deep_oversold'] = df['wt2'] < wt_os_level3 # For Gold signal
        
        # Make sure RSI exists before proceeding with RSI conditions
        if 'rsi' in df.columns and not df['rsi'].isna().all():
            # RSI Overbought/Oversold Conditions
            df['rsi_oversold'] = df['rsi'] < rsi_os_level
            df['rsi_overbought'] = df['rsi'] > rsi_ob_level
            df['rsi_below_20'] = df['rsi'] < 20 # For Gold signal
        else:
            # Set default values if RSI failed
            df['rsi_oversold'] = False
            df['rsi_overbought'] = False
            df['rsi_below_20'] = False
            print("Warning: RSI calculation failed, setting default values")

        # Basic WT Signals (Circles without divergence yet)
        df['signal_buy_basic'] = df['wt_cross_up'] & df['wt_oversold']
        df['signal_sell_basic'] = df['wt_cross_down'] & df['wt_overbought']
        
    except Exception as e:
        print(f"Error in calculate_indicators: {e}")
        # Handle error gracefully to avoid script crash
    
    return df

def find_divergences(df, indicator_col='wt2', price_high_col='high', price_low_col='low',
                     ob_level=wt_div_ob_level, os_level=wt_div_os_level,
                     lookback=divergence_lookback, min_separation=min_divergence_separation):
    """
    Finds regular bullish and bearish divergences using scipy.signal.find_peaks.
    Returns indices of divergence points.
    """
    # Ensure arrays are 1D for find_peaks
    indicator = df[indicator_col].values.flatten()
    price_high = df[price_high_col].values.flatten()
    price_low = df[price_low_col].values.flatten()

    # Check if arrays are empty
    if len(indicator) == 0 or len(price_high) == 0 or len(price_low) == 0:
        print("Warning: Empty arrays in find_divergences")
        return df

    try:
        # Find peaks (local maxima) in indicator and price highs
        ind_peaks, _ = find_peaks(indicator, distance=min_separation)
        price_high_peaks, _ = find_peaks(price_high, distance=min_separation)

        # Find troughs (local minima) by finding peaks in the negative series
        ind_troughs, _ = find_peaks(-indicator, distance=min_separation)
        price_low_troughs, _ = find_peaks(-price_low, distance=min_separation)

        bearish_div_indices = []
        bullish_div_indices = []

        # Bearish Divergence: Higher High in Price, Lower High in Indicator (check within lookback)
        for i in range(len(ind_peaks)):
            p1_idx = ind_peaks[i]
            relevant_peaks = ind_peaks[(ind_peaks < p1_idx) & (ind_peaks >= p1_idx - lookback)]
            for p0_idx in relevant_peaks:
                if not np.isnan(indicator[p1_idx]) and not np.isnan(indicator[p0_idx]) and \
                   indicator[p1_idx] < indicator[p0_idx] and indicator[p1_idx] > ob_level:
                     closest_price_p0 = price_high_peaks[np.argmin(np.abs(price_high_peaks - p0_idx))] if len(price_high_peaks)>0 else -1
                     closest_price_p1 = price_high_peaks[np.argmin(np.abs(price_high_peaks - p1_idx))] if len(price_high_peaks)>0 else -1
                     if closest_price_p0 != -1 and closest_price_p1 != -1 and \
                        closest_price_p1 > closest_price_p0 and \
                        abs(closest_price_p0 - p0_idx) < min_separation*2 and abs(closest_price_p1 - p1_idx) < min_separation*2 :
                         if not np.isnan(price_high[closest_price_p1]) and not np.isnan(price_high[closest_price_p0]) and \
                            price_high[closest_price_p1] > price_high[closest_price_p0]:
                             bearish_div_indices.append(p1_idx)
                             break

        # Bullish Divergence: Lower Low in Price, Higher Low in Indicator (check within lookback)
        for i in range(len(ind_troughs)):
            t1_idx = ind_troughs[i]
            relevant_troughs = ind_troughs[(ind_troughs < t1_idx) & (ind_troughs >= t1_idx - lookback)]
            for t0_idx in relevant_troughs:
                 if not np.isnan(indicator[t1_idx]) and not np.isnan(indicator[t0_idx]) and \
                    indicator[t1_idx] > indicator[t0_idx] and indicator[t1_idx] < os_level:
                    closest_price_t0 = price_low_troughs[np.argmin(np.abs(price_low_troughs - t0_idx))] if len(price_low_troughs)>0 else -1
                    closest_price_t1 = price_low_troughs[np.argmin(np.abs(price_low_troughs - t1_idx))] if len(price_low_troughs)>0 else -1
                    if closest_price_t0 != -1 and closest_price_t1 != -1 and \
                       closest_price_t1 > closest_price_t0 and \
                       abs(closest_price_t0 - t0_idx) < min_separation*2 and abs(closest_price_t1 - t1_idx) < min_separation*2:
                        if not np.isnan(price_low[closest_price_t1]) and not np.isnan(price_low[closest_price_t0]) and \
                           price_low[closest_price_t1] < price_low[closest_price_t0]:
                             bullish_div_indices.append(t1_idx)
                             break

        df['bear_div_wt'] = False
        if bearish_div_indices: # Ensure list is not empty before iloc
            df.iloc[bearish_div_indices, df.columns.get_loc('bear_div_wt')] = True
        df['bull_div_wt'] = False
        if bullish_div_indices: # Ensure list is not empty before iloc
            df.iloc[bullish_div_indices, df.columns.get_loc('bull_div_wt')] = True

        # Check if a divergence occurred recently (e.g., within 3 bars) *before* or *at* the cross
        lookback_div = 3
        df['bull_div_recent'] = df['bull_div_wt'].rolling(window=lookback_div).sum() > 0
        df['bear_div_recent'] = df['bear_div_wt'].rolling(window=lookback_div).sum() > 0

        # Purple Triangle Signals (WT Cross + OB/OS + Recent Divergence)
        df['signal_buy_div'] = df['signal_buy_basic'] & df['bull_div_recent']
        df['signal_sell_div'] = df['signal_sell_basic'] & df['bear_div_recent']

        # Gold Circle Signal (WT Cross Up + Deep OS + RSI < 20 + Recent Bull Div)
        df['signal_gold'] = df['wt_cross_up'] & df['wt_deep_oversold'] & df['rsi_below_20'] & df['bull_div_recent']

        return df

    except Exception as e:
        print(f"Error in find_divergences: {e}")
        return df


def plot_results(df, ticker, output_dir='', timestamp=''):
    """Plots the price and the Cipher B oscillator panel."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1]) # Allocate more space for price

    # Extract currency from ticker (assuming format like 'BTC-USD')
    currency = 'USD'
    if '-' in ticker:
        currency = ticker.split('-')[1]

    # --- Price Chart ---
    ax_price = fig.add_subplot(gs[0, 0])
    ax_price.set_title(f'{ticker} Price Chart ({interval})')
    ax_price.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.8, linewidth=1)

    ax_price.set_ylabel(f'Price ({currency})')
    ax_price.legend()
    ax_price.grid(True, alpha=0.3)
    # Format x-axis to show only date, not time
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax_price.get_xticklabels(), rotation=15, ha='right')

    # --- Oscillator Panel ---
    ax_osc = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_osc.set_title('WaveTrend Oscillator & Signals')

    # Plot WT1 and WT2
    ax_osc.plot(df.index, df['wt1'], label='WT1 (Fast)', color='#8ec7fb', linewidth=1.5)
    ax_osc.plot(df.index, df['wt2'], label='WT2 (Slow)', color='#1353ac', linewidth=1.5)

    # Plot VWAP Oscillator (Optional)
    if plot_vwap_osc:
        ax_osc.plot(df.index, df['vwap_osc'], label='VWAP Osc', color='yellow', linestyle='--', linewidth=1, alpha=0.7)
        # Fill VWAP area (optional)
        # ax_osc.fill_between(df.index, df['vwap_osc'], 0, color='yellow', alpha=0.15, interpolate=True)


    # Plot Overbought/Oversold Lines
    ax_osc.axhline(wt_ob_level1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_osc.axhline(wt_ob_level2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax_osc.axhline(wt_os_level1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_osc.axhline(wt_os_level2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax_osc.axhline(wt_os_level3, color='darkgoldenrod', linestyle=':', linewidth=1, alpha=0.5) # Gold level line
    ax_osc.axhline(0, color='white', linestyle='-', linewidth=0.5, alpha=0.8) # Zero line

    # Plot WT Signals (Circles)
    # Little circles (All WT crosses)
    wt_cross_idx_up = df.index[df['wt_cross_up']]
    wt_cross_val_up = df.loc[df['wt_cross_up'], 'wt2']
    ax_osc.scatter(wt_cross_idx_up, wt_cross_val_up, marker='o', s=25, color='lime', alpha=0.7, label='WT Cross Up (Small)')

    wt_cross_idx_down = df.index[df['wt_cross_down']]
    wt_cross_val_down = df.loc[df['wt_cross_down'], 'wt2']
    ax_osc.scatter(wt_cross_idx_down, wt_cross_val_down, marker='o', s=25, color='red', alpha=0.7, label='WT Cross Down (Small)')

    # Big Green Circle (Basic Buy - WT Cross Up in OS)
    buy_basic_idx = df.index[df['signal_buy_basic'] & ~df['signal_buy_div']] # Exclude div signals here
    buy_basic_val = df.loc[buy_basic_idx, 'wt2']
    ax_osc.scatter(buy_basic_idx, buy_basic_val, marker='o', s=100, edgecolor='lime', facecolor='none', linewidth=1.5, label='Buy Signal (Green Circle)')

    # Big Red Circle (Basic Sell - WT Cross Down in OB)
    sell_basic_idx = df.index[df['signal_sell_basic'] & ~df['signal_sell_div']] # Exclude div signals here
    sell_basic_val = df.loc[sell_basic_idx, 'wt2']
    ax_osc.scatter(sell_basic_idx, sell_basic_val, marker='o', s=100, edgecolor='red', facecolor='none', linewidth=1.5, label='Sell Signal (Red Circle)')

    # Plot Divergence Markers (Optional Lines)
    if wt_show_div:
        # Purple Triangles (Buy/Sell signals WITH Divergence)
        buy_div_idx = df.index[df['signal_buy_div']]
        buy_div_val = df.loc[buy_div_idx, 'wt2']
        ax_osc.scatter(buy_div_idx, buy_div_val, marker='^', s=150, color='purple', label='Buy Signal w/ Div (Purple Up)')

        sell_div_idx = df.index[df['signal_sell_div']]
        sell_div_val = df.loc[sell_div_idx, 'wt2']
        ax_osc.scatter(sell_div_idx, sell_div_val, marker='v', s=150, color='purple', label='Sell Signal w/ Div (Purple Down)')

    # Gold Circle Signal
    gold_idx = df.index[df['signal_gold']]
    gold_val = df.loc[gold_idx, 'wt2']
    ax_osc.scatter(gold_idx, gold_val, marker='o', s=100, color='gold', label='Gold Signal')


    ax_osc.set_ylabel('Oscillator Value')
    ax_osc.legend(fontsize='small', loc='upper left')
    ax_osc.grid(True, alpha=0.3)
    ax_osc.tick_params(axis='x', which='major', labelbottom=False) # Hide x-axis labels


    # --- MFI / RSI Panel ---
    ax_mfi = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_mfi.set_title('Money Flow Index (MFI) Area / RSI')

    # Plot MFI Area
    if plot_mfi_area:
        ax_mfi.fill_between(df.index, df['mvc'], 0, where=df['mvc'] >= 0, color='green', alpha=0.4, interpolate=True, label='MFI Area (Bullish)')
        ax_mfi.fill_between(df.index, df['mvc'], 0, where=df['mvc'] < 0, color='red', alpha=0.4, interpolate=True, label='MFI Area (Bearish)')
        ax_mfi.axhline(0, color='white', linestyle='-', linewidth=0.5, alpha=0.8) # Zero line for MFI
        
        # Find the most recent flip in MFI (bullish to bearish or vice versa)
        if len(df) > 1:
            # Create a series of 1 (bullish) and -1 (bearish) based on MFI sign
            mfi_sign = np.sign(df['mvc'].fillna(0))
            # Find where sign changes (diffs != 0)
            sign_changes = mfi_sign.diff().fillna(0).ne(0)
            
            if sign_changes.any():
                # Get the index of the last sign change
                last_flip_idx = sign_changes[sign_changes].index[-1]
                last_flip_date = last_flip_idx.strftime('%Y-%m-%d')
                
                # Extract the MFI value as a scalar using .iloc[0] to avoid FutureWarning
                last_flip_value = df.loc[last_flip_idx, 'mvc'].iloc[0]
                flip_type = "Bearish to Bullish" if last_flip_value > 0 else "Bullish to Bearish"
                
                # Draw a vertical line at the flip point
                ax_mfi.axvline(x=last_flip_idx, color='yellow', linestyle='-', linewidth=1.5, 
                             label=f'Latest Flip: {flip_type} on {last_flip_date}')
                
                # Add an annotation
                y_pos = max(abs(df['mvc'].min()), abs(df['mvc'].max())) * 0.8
                y_pos = y_pos if last_flip_value > 0 else -y_pos
                
                ax_mfi.annotate(f'{flip_type}\n{last_flip_date}', 
                              xy=(last_flip_idx, y_pos),
                              xytext=(15, 0), textcoords='offset points',
                              ha='left', va='center',
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot RSI (Optional)
    if plot_rsi:
        ax_mfi.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1)
        ax_mfi.axhline(rsi_ob_level, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_mfi.axhline(rsi_os_level, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_mfi.set_ylim(0, 100) # RSI scale

    ax_mfi.set_ylabel('MFI / RSI Value')
    ax_mfi.legend(fontsize='small', loc='upper left')
    ax_mfi.grid(True, alpha=0.3)

    # Improve date formatting for the bottom chart - Show only date, not time
    ax_mfi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax_mfi.get_xticklabels(), rotation=15, ha='right')
    plt.xlabel('Date')

    # Add more space between subplots to avoid overlap
    plt.tight_layout(pad=1.0) # Adjust layout

    # Save the plot to a file
    filename = f'{output_dir}/{ticker}_cypher_plot_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    output_dir = '/home/j/ai/crewAI/finance/stock_analyser/plots/market_cyphers'
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Fetching data for {ticker}...")
    df_raw = get_data(ticker, start_date, end_date, interval)

    if not df_raw.empty:
        print("Calculating indicators using TA-Lib...")
        df_ind = calculate_indicators(df_raw.copy()) # Work on a copy

        print("Finding divergences...")
        df_final = find_divergences(df_ind.copy()) # Add divergence signals

        # Clean up initial NaN values produced by moving averages/RSI
        first_valid_index = df_final['wt2'].first_valid_index()
        if first_valid_index is not None:
             df_plot = df_final.loc[first_valid_index:]
             print(f"Plotting results from {first_valid_index}...")
             plot_results(df_plot, ticker, output_dir, timestamp)
        else:
            print("Could not plot: Not enough data after indicator calculation.")
            print(df_final.tail()) # Print tail for debugging
    else:
        print("Failed to retrieve or process data.")