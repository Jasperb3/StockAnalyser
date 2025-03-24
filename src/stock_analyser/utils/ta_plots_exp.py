import talib
from talib import abstract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

from stock_analyser.utils.constants import PLOTS_DIR


def HT_DCPERIOD(df):
    """
    Function Name: HT_DCPERIOD
    Name: Hilbert Transform - Dominant Cycle Period

    Introduction: Uses price as an information signal to calculate the position of the price within its cycle, serving as a basis for timing.

    NOTE: The ``HT_DCPERIOD`` function has an unstable period.
    python API
    real=HT_DCPERIOD(close)
    :return:
    """
    close = df["Close"]
    return talib.HT_DCPERIOD(close)


def HT_DCPHASE(df):
    """
     Function Name: HT_DCPHASE
     Name: Hilbert Transform - Dominant Cycle Phase

     Introduction: Measures the phase of the Dominant Cycle, aiding in identifying turning points.

     NOTE: The ``HT_DCPHASE`` function has an unstable period.
    python API
    real=HT_DCPHASE(close)
    :return:
    """
    close = df["Close"]
    return talib.HT_DCPHASE(close)


def HT_PHASOR(df):
    """
     Function Name: HT_PHASOR
     Name: Hilbert Transform - Phasor Components

     Introduction: Decomposes the price into In-Phase and Quadrature components, revealing the cyclical characteristics.

     NOTE: The ``HT_PHASOR`` function has an unstable period.
    python API
    inphase, quadrature=HT_PHASOR(close)
    :return:
    """
    close = df["Close"]
    return talib.HT_PHASOR(close)


def HT_SINE(df):
    """
     Function Name: HT_SINE
     Name: Hilbert Transform - Sine Wave

     Introduction: Generates Sine and Lead-Sine waves based on the Dominant Cycle, providing insights into cyclical patterns.

     NOTE: The ``HT_SINE`` function has an unstable period.
    python API
    sine, leadsine=HT_SINE(close)
    :return:
    """
    close = df["Close"]
    return talib.HT_SINE(close)


def HT_TRENDMODE(df):
    """
     Function Name: HT_TRENDMODE
     Name: Hilbert Transform - Trend vs Cycle Mode

     Introduction: Determines whether the market is in a trending or cyclical mode, assisting in strategy selection.

     NOTE: The ``HT_TRENDMODE`` function has an unstable period.
    python API
    integer=HT_TRENDMODE(close)
    :return:
    """
    close = df["Close"]
    return talib.HT_TRENDMODE(close)


################
# TA-LIB
# Momentum Indicator Functions
#################


def ADD(df):
    """
    Function Name: ADD
    Name: Vector addition

    python API
    real=ADD(high, low)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.ADD(high, low)


def DIV(df):
    """
    Function Name: DIV
    Name: Vector division

    python API
    real=DIV(high, low)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.DIV(high, low)


def MAXINDEX(df, time_period=30):
    """
    Function Name: MAXINDEX
    Name: Index of highest value over a specified period

    Introduction: Returns the index of the highest value within the given lookback period.

    python API
    integer=MAXINDEX(close, timeperiod=30)
    :param time_period:
    :return:
    """
    close = df["Close"]
    return talib.MAXINDEX(close, timeperiod=time_period)


def MININDEX(df, time_period=30):
    """
    Function Name: MININDEX
    Name: Index of lowest value over a specified period

    Introduction: Returns the index of the lowest value within the given lookback period.

    python API
    integer=MININDEX(close, timeperiod=30)
    :param time_period:
    :return:
    """
    close = df["Close"]
    return talib.MININDEX(close, timeperiod=time_period)


def MINMAX(df, time_period=30):
    """
    Function Name: MINMAX
    Name: Minimum and maximum values over a specified period

    Introduction: Returns the minimum and maximum values within the given lookback period.
    """
    close = df["Close"]
    return talib.MINMAX(close, timeperiod=time_period)


def MINMAXINDEX(df, time_period=30):
    """
    Function Name: MINMAXINDEX
    Name: Indexes of minimum and maximum values over a specified period

    Introduction: Returns the indexes of the minimum and maximum values within the given lookback period.
    """
    close = df["Close"]
    return talib.MINMAXINDEX(close, timeperiod=time_period)


def MULT(df):
    """
    Function Name: MULT
    Name: Vector multiplication

    python API
    real=MULT(high, low)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.MULT(high, low)


def SUB(df):
    """
    Function Name: SUB
    Name: Vector subtraction

    python API
    real=SUB(high, low)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.SUB(high, low)


################
# TA-LIB
# Math Transform Functions
################


def ACOS(df):
    """
    Function Name: ACOS
    Name: Vector Arccosine

    Introduction: Calculates the arccosine of each element in the data series.
    """
    close = df["Close"]
    return talib.ACOS(close)


def ASIN(df):
    """
    Function Name: ASIN
    Name: Vector Arcsine

    Introduction: Calculates the arcsine of each element in the data series.
    """
    close = df["Close"]
    return talib.ASIN(close)


def ATAN(df):
    """
    Function Name: ATAN
    Name: Vector Arctangent

    Introduction: Calculates the arctangent of each element in the data series.
    """
    close = df["Close"]
    return talib.ATAN(close)


def CEIL(df):
    """
    Function Name: CEIL
    Name: Vector Ceiling

    Introduction: Returns the smallest integer greater than or equal to each element in the data series.
    """
    close = df["Close"]
    return talib.CEIL(close)


def COS(df):
    """
    Function Name: COS
    Name: Vector Cosine

    Introduction: Calculates the cosine of each element in the data series.
    """
    close = df["Close"]
    return talib.COS(close)


def COSH(df):
    """
    Function Name: COSH
    Name: Vector Hyperbolic Cosine

    Introduction: Calculates the hyperbolic cosine of each element in the data series.
    """
    close = df["Close"]
    return talib.COSH(close)


def EXP(df):
    """
    Function Name: EXP
    Name: Vector Exponential

    Introduction: Calculates the exponential of each element in the data series.
    """
    close = df["Close"]
    return talib.EXP(close)


def FLOOR(df):
    """
    Function Name: FLOOR
    Name: Vector Floor

    Introduction: Returns the largest integer less than or equal to each element in the data series.
    """
    close = df["Close"]
    return talib.FLOOR(close)


def LN(df):
    """
    Function Name: LN
    Name: Natural Logarithm

    python API
    real=LN(close)
    :return:
    """
    close = df["Close"]
    return talib.LN(close)


def LOG10(df):
    """
    Function Name: LOG10
    Name: Logarithm (base 10)

    python API
    real=LOG10(close)
    :return:
    """
    close = df["Close"]
    return talib.LOG10(close)


def SIN(df):
    """
    Function Name: SIN
    Name: Sine, trigonometric function

    python API
    real=SIN(close)
    :return:
    """
    close = df["Close"]
    return talib.SIN(close)


def SINH(df):
    """
    Function Name: SINH
    Name: Hyperbolic sine, trigonometric function

    python API
    real=SINH(close)
    :return:
    """
    close = df["Close"]
    return talib.SINH(close)


def SQRT(df):
    """
    Function Name: SQRT
    Name: Vector Square Root

    Introduction: Calculates the square root of each element in the data series.
    """
    close = df["Close"]
    return talib.SQRT(close)


def TAN(df):
    """
    Function Name: TAN
    Name: Tangent, trigonometric function

    python API
    real=TAN(close)
    :return:
    """
    close = df["Close"]
    return talib.TAN(close)


def TANH(df):
    """
    Function Name: TANH
    Name: Hyperbolic tangent, trigonometric function

    python API
    real=TANH(close)
    :return:
    """
    close = df["Close"]
    return talib.TANH(close)


################
# TA-LIB
# Momentum Indicator Functions
#################


def ADX(df, time_period=14):
    """
    Function Name: ADX
    Name: Average Directional Movement Index

    Introduction: Measures the strength of a trend, regardless of direction.

    Formula:
    ADX = 100 * Wilder_MA(abs(DX), N)

    Where:
    DX = 100 * abs((+DI) - (-DI)) / ((+DI) + (-DI))
    +DI = 100 * Wilder_MA(UpMove, N) / ATR(N)
    -DI = 100 * Wilder_MA(DownMove, N) / ATR(N)

    UpMove = Current High - Previous High
    DownMove = Previous Low - Current Low

    If UpMove > DownMove and UpMove > 0, then UpMove, else 0
    If DownMove > UpMove and DownMove > 0, then DownMove, else 0

    Wilder_MA is Welles Wilder's Moving Average (often similar to EMA)

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        time_period (int): The time period (N) for the Wilder's Moving Average.

    Returns:
        pd.Series: The ADX values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.ADX(high, low, close, timeperiod=time_period)


def ACCER(df, N=8):
    """
    Function Name: ACCER
    Name: Acceleration

    Introduction: Calculates the acceleration of price, which is the rate of change of velocity.

    Formula:
    AC = (Velocity(High, N) + Velocity(Low, N)) / 2
    Velocity(High, N) = Highest(High, N) - High
    Velocity(Low, N) = Low - Lowest(Low, N)

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        N (int): The time period.
    """
    CLOSE = df["Close"]
    return talib.LINEARREG_SLOPE(CLOSE, timeperiod=N)


def ADXR(df, time_period=14):
    """
    Function Name: ADXR
    Name: Average Directional Movement Index Rating

    Introduction: A smoothed version of the ADX, providing a less responsive but potentially more reliable trend strength indicator.

    Formula:
    ADXR = (ADX + ADX[time_period days ago]) / 2

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        time_period (int): The time period.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.ADXR(high, low, close, timeperiod=time_period)


def APO(df, fast_period=12, slow_period=26, ma_type=0):
    """
    Function Name: APO
    Name: Absolute Price Oscillator

    Introduction: Shows the difference between two moving averages of different lengths, expressed as an absolute value.

    python API
    real=APO(close, fastperiod=12, slowperiod=26, matype=0)
    :return:
    """
    close = df["Close"]
    return talib.APO(
        close, fastperiod=fast_period, slowperiod=slow_period, matype=ma_type
    )


def AROON(df, time_period=14):
    """
    Function Name: AROON
    Name: Aroon

    Introduction:  Indicates the time since the highest high and lowest low occurred within the lookback period.

    Formula:
    Aroon Up = ((N - Number of periods since highest high) / N) * 100
    Aroon Down = ((N - Number of periods since lowest low) / N) * 100

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        time_period (int): The time period (N).

    Returns:
      pd.DataFrame: DataFrame with 'Aroon Up' and 'Aroon Down' columns.
    """
    high = df["High"]
    low = df["Low"]
    return talib.AROON(high, low, timeperiod=time_period)


def AROONOSC(df, time_period=14):
    """
    Function Name: AROONOSC
    Name: Aroon Oscillator

    Introduction:  Calculates the difference between Aroon Up and Aroon Down.

    Formula:  Aroon Oscillator = Aroon Up - Aroon Down
    """
    high = df["High"]
    low = df["Low"]
    return talib.AROONOSC(high, low, timeperiod=time_period)


def BOP(df):
    """
    Function Name: BOP
    Name: Balance Of Power

    Introduction: Measures the buying and selling pressure.

    Formula: BOP = (Close - Open) / (High - Low)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    open = df["Open"]
    return (close - open) / (high - low)


def CMO(df, time_period=14):
    """
    Function Name: CMO
    Name: Chande Momentum Oscillator

    Introduction: A momentum oscillator that measures the difference between the sum of higher closes and the sum of lower closes, normalized by the total price movement.

    Formula:
    CMO = 100 * ((Sum of Upward Movements - Sum of Downward Movements) / (Sum of Upward Movements + Sum of Downward Movements))

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        time_period (int):  Lookback period.

    Returns:
        pd.Series: CMO values.
    """
    close = df["Close"]
    return talib.CMO(close, timeperiod=time_period)


def DX(df, time_period=14):
    """
    Function Name: DX
    Name: Directional Movement Index

    Introduction: Measures the strength of price movement in a particular direction.

    Formula:
    DX = 100 * abs((+DI) - (-DI)) / ((+DI) + (-DI))

    Refer to ADX for +DI and -DI calculations.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.DX(high, low, close, timeperiod=time_period)


def MACDEXT(
    df,
    fast_period=12,
    fast_ma_type=0,
    slow_period=26,
    slow_ma_type=0,
    signal_period=9,
    signal_ma_type=0,
):
    """
    Function Name: MACDEXT
    Name: MACD with controllable MA types

    Introduction:  A more flexible version of the MACD, allowing the use of different moving average types.
    """
    close = df["Close"]
    return talib.MACDEXT(
        close,
        fastperiod=fast_period,
        fastmatype=fast_ma_type,
        slowperiod=slow_period,
        slowmatype=slow_ma_type,
        signalperiod=signal_period,
        signalmatype=signal_ma_type,
    )


def MACDFIX(df, signal_period=9):
    """
    python API
    macd, macdsignal, macdhist=MACDFIX(close, signalperiod=9)
    :return:
    """
    close = df["Close"]
    return talib.MACDFIX(close, signalperiod=signal_period)


def MFI(df, time_period=14):
    """
    Function Name: MFI
    Name: Money Flow Index

    Introduction: An oscillator that uses both price and volume to measure buying and selling pressure.

    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price * Volume
    Name: Money Flow Index

    Introduction: It is a kind of volume and price index, reflecting the trend of the market

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/mfi/7429225?fr=aladdin)
    [iWencai School](http://www.iwencai.com/school/search?cg=100&w=MFI)

    python API
    real=MFI(high, low, close, volume, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]
    return talib.MFI(high, low, close, volume, timeperiod=time_period)


def MINUS_DI(df, time_period=14):
    """
    Function Name: MINUS_DI
    Name: Minus Directional Indicator

    Introduction:  Part of the DMI indicator, the Negative Directional Indicator. It analyzes the change of the equilibrium point between buying and selling forces during the rise and fall of stock prices. This change, influenced by price fluctuations, goes through cycles of equilibrium to disequilibrium, providing a basis for judging trends.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/DMI%E6%8C%87%E6%A0%87/3423254?fr=aladdin)
    [Wikipedia](https://zh.wikipedia.org/wiki/% E5%8B%95%E5%90%91%E6%8C%87%E6%95%B8)
    [iWencai School](http://www.iwencai.com/school/search?cg=100&w=DMI)

    python API
    real=MINUS_DI(high, low, close, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.MINUS_DI(high, low, close, timeperiod=time_period)


def MINUS_DM(df, time_period=14):
    """
    Function Name: MINUS_DM
    Name: Minus Directional Movement

    Introduction: In DMI, DM represents the positive trend change value, which is the upward movement value. It analyzes the change of the equilibrium point between buying and selling forces during the rise and fall of stock prices. This change, influenced by price fluctuations, goes through cycles of equilibrium to disequilibrium, providing a basis for judging trends.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/DMI%E6%8C%87%E6%A0%87/3423254?fr=aladdin)
    [Wikipedia](https://zh.wikipedia.org/wiki/% E5%8B%95%E5%90%91%E6%8C%87%E6%95%B8)
    [iWencai School](http://www.iwencai.com/school/search?cg=100&w=DMI)

    python API
    real=MINUS_DM(high, low, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.MINUS_DM(high, low, timeperiod=time_period)


def MOM(df, time_period=14):
    """
    Function Name: MOM
    Name: Momentum

    Introduction: In investment, it refers to the ability of a stock (or economic index) to continue growing. Research has found that winning portfolios have a positive momentum effect in bull markets, and losing portfolios have a negative momentum effect in bear markets.

    Analysis and Application:
    [Wikipedia](https://zh.wikipedia.org/wiki/% E5%8B%95%E9%87%8F%E6%8C%87%E6%A8%99)
    [iWencai School](http://www.iwencai.com/yike/detail/auid/cb18b2dbe2f455e6)

    python API
    real=MOM(close, timeperiod=10)
    :return:
    """
    close = df["Close"]
    return talib.MOM(close, timeperiod=time_period)


def PLUS_DI(df, time_period=14):
    """
    Function Name: PLUS_DI
    Name: Plus Directional Indicator (+DI)

    Introduction: Measures the strength of upward price movement.

    Formula:
     +DI = 100 * Wilder_MA(UpMove, N) / ATR(N)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.PLUS_DI(high, low, close, timeperiod=time_period)


def PLUS_DM(df, time_period=14):
    """
    Function Name: PLUS_DM
    Name: Plus Directional Movement (+DM)

    Introduction: Measures the upward movement of price.

    Formula: +DM = Wilder_MA(UpMove, N)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.PLUS_DM(high, low, close, timeperiod=time_period)


def PPO(df, fast_period=12, slow_period=26, ma_type=0):
    """
    Function Name: PPO
    Name: Percentage Price Oscillator

    Introduction: A momentum oscillator used to identify overbought or oversold conditions.

    Formula:
    PPO = 100 * (EMA(close, fastperiod) - EMA(close, slowperiod)) / EMA(close, slowperiod)
    """
    close = df["Close"]
    return talib.PPO(
        close, fastperiod=fast_period, slowperiod=slow_period, matype=ma_type
    )


def STOCHF(df, fastk_period=5, fastd_period=3, fastd_matype=0):
    """
    python API
    fastk, fastd=STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.STOCHF(
        high,
        low,
        close,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )


def STOCHRSI(df, time_period=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    """
     python API
    fastk, fastd=STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    :return:
    """
    close = df["Close"]
    return talib.STOCHRSI(
        close,
        timeperiod=time_period,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )


def TRIX(df, time_period=30):
    """
    python API
    real=TRIX(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.TRIX(close, timeperiod=time_period)


def ULTOSC(df, time_period1=7, time_period2=14, time_period3=28):
    """
    Function Name: ULTOSC
    Name: Ultimate Oscillator

    Introduction: UOS is a multi-functional indicator. In addition to confirming trends and identifying overbought/oversold conditions, its "breakthrough" signal can not only provide the most appropriate trading timing, but also further enhance the reliability of the indicator.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/% E7%BB%88%E6%9E%81%E6%B3%A2%E5%8A%A8%E6%8C%87%E6%A0%87/1982936?fr=aladdin&fromid=12610066&fromtitle=% E7%BB%88%E6%9E%81%E6%8C%87%E6%A0%87)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/e89b98d39da975e4)

    python API
    real=ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.ULTOSC(
        high,
        low,
        close,
        timeperiod1=time_period1,
        timeperiod2=time_period2,
        timeperiod3=time_period3,
    )


def WILLR(df, time_period=14):
    """
    Function Name: WILLR
    Name: Williams %R

    Introduction: WMS%R indicates whether the market is in an overbought or oversold state. Stock investment analysis methods mainly include the following three: fundamental analysis, technical analysis, and evolutionary analysis. In practical applications, they are both interconnected and have important differences.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/% E5%A8%81%E5%BB%89%E6%8C%87%E6%A0%87?fr=aladdin)
    [Wikipedia](https://zh.wikipedia.org/wiki/% E5%A8%81%E5%BB%89%E6%8C%87%E6%A8%99)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/967febb0316c57c1)

    python API
    real=WILLR(high, low, close, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.WILLR(high, low, close, timeperiod=time_period)


################
# TA-LIB Quantitative Analysis Data
# Overlap Studies Functions
#################


def BBANDS(df, time_period=5, nb_de_vup=2, nb_dev_dn=2, ma_type=0):
    """
    Function Name: BBANDS
    Name: Bollinger Bands

    Introduction: It uses statistical principles to calculate the standard deviation and confidence interval of the stock price, so as to determine the fluctuation range and future trend of the stock price. It uses the band to display the safe high and low price levels of the stock price, so it is also called the Bollinger Band.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/bollinger%20bands/1612394?fr=aladdin)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/56d0d9be66b4f7a0?rid=53)

    python API
    upperband, middleband, lowerband=BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    :return:
    """
    close = df["Close"]
    return talib.BBANDS(
        close,
        timeperiod=time_period,
        nbdevup=nb_de_vup,
        nbdevdn=nb_dev_dn,
        matype=ma_type,
    )


def DEMA(df, time_period=30):
    """
    Function Name: DEMA
    Name: Double Exponential Moving Average

    Introduction: Two moving averages are used to generate trend signals. The longer-term one is used to identify the trend, and the shorter-term one is used to choose the timing. It is the interaction of the two moving averages and the price that together generate the trend signal.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/%E5%8F%8C%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF/1831921?fr=aladdin)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/a04d723659318237)

    python API
    real=DEMA(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.DEMA(close, timeperiod=time_period)


def HT_TRENDLINE(df):
    """
    Function Name: HT_TRENDLINE
    Name: Hilbert Transform - Trendline

    Introduction: It is a trend-following indicator. Its construction principle is still to perform arithmetic averaging on the closing price of the stock, and analyze it based on the calculation results to determine the future trend of the price.

    Analysis and Application:
    [Baidu Wenku](https://wenku.baidu.com/view/0e35f6eead51f01dc281f18e.html)

    python API
    real=HT_TRENDLINE(close)
    :return:
    """
    close = df["Close"]
    return talib.HT_TRENDLINE(close)


def KAMA(df, time_period=30):
    """
    Function Name: KAMA
    Name: Kaufman's Adaptive Moving Average

    Introduction: Short-term moving averages are close to the price trend and are highly sensitive, but there will be a lot of noise, resulting in false signals; long-term moving averages are generally accurate in judging trends, but long-term moving averages have serious lag problems. We want to get such a moving average. When the price moves rapidly in one direction, the short-term moving average is the most suitable; when the price is in a sideways process, the long-term moving average is suitable.

    Analysis and Application:
    [Reference 1](http://blog.sina.com.cn/s/blog_62d0bbc701010p7d.html)
    [Reference 2](https://wenku.baidu.com/view/bc4bc9c59ec3d5bbfd0a7454.html?from=search)

    python API
    real=KAMA(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.KAMA(close, timeperiod=time_period)


def MAMA(df, fast_limit=0, slow_limit=0):
    """
    python API
    mama, fama=MAMA(close, fastlimit=0, slowlimit=0)
    :return:
    """
    close = df["Close"]
    return talib.MAMA(close, fastlimit=fast_limit, slowlimit=slow_limit)


def MAVP(df, min_period=2, max_period=30, ma_type=0):
    """
     python API
    real=MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
    :return:
    """
    close = df["Close"]
    low = df["Low"]
    return talib.MAVP(
        close, low, minperiod=min_period, maxperiod=max_period, matype=ma_type
    )


def MIDPOINT(df, time_period=14):
    """
    python API
    real=MIDPOINT(close, timeperiod=14)
    :return:
    """
    close = df["Close"]
    return talib.MIDPOINT(close, timeperiod=time_period)


def MIDPRICE(df, time_period=14):
    """
    python API
    real=MIDPRICE(high, low, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.MIDPRICE(high, low, timeperiod=time_period)


def SAR(df, acceler_ation=0, max_imum=0):
    """
    Function Name: SAR
    Name: Parabolic SAR

    Introduction: Parabolic Stop and Reverse, also known as Parabolic SAR, uses a parabolic curve to dynamically adjust the stop-loss level, helping to identify buying and selling points.  The stop-loss level (SAR) moves in a curved manner.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/SAR/2771135#viewPageContent)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/d9d94e65be7f6b5e)

    python API
    real=SAR(high, low, acceleration=0, maximum=0)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.SAR(high, low, acceleration=acceler_ation, maximum=max_imum)


def SAREXT(
    df,
    start_value=0,
    offset_onreverse=0,
    acceleration_init_long=0,
    acceleration_long=0,
    acceleration_max_long=0,
    acceleration_init_short=0,
    acceleration_short=0,
    acceleration_max_short=0,
):
    """
    python API
    real=SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.SAREXT(
        high,
        low,
        startvalue=start_value,
        offsetonreverse=offset_onreverse,
        accelerationinitlong=acceleration_init_long,
        accelerationlong=acceleration_long,
        accelerationmaxlong=acceleration_max_long,
        accelerationinitshort=acceleration_init_short,
        accelerationshort=acceleration_short,
        accelerationmaxshort=acceleration_max_short,
    )


def SMA(df, time_period=30):
    """
    Function Name: SMA
    Name: Simple Moving Average

    Introduction:  A moving average (MA) is the average of prices over a certain period. It's called "moving" because the average is calculated for each successive time period. For example, a 5-day SMA is the sum of the closing prices for the last 5 days, divided by 5.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF/217887?fromtitle=MA&fromid=1511750#viewPageContent)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/a04d723659318237?rid=96)

    python API
    real=SMA(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.SMA(close, timeperiod=time_period)


def T3(df, time_period=5, v_factor=0):
    """
    Function Name: T3
    Name: Triple Exponential Moving Average

    Introduction: When TRIX is used for long-term operations, following its signals for extended periods generally results in a higher percentage of profits than losses, leading to considerable gains. For example, a 5-day MA is calculated by summing closing prices over 5 days and dividing by 5.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/%E4%B8%89%E9%87%8D%E6%8C%87%E6%95%B0%E5%B9%B3%E6%BB%91%E5%B9%B3%E5%9D%87%E7%BA%BF/15749345?fr=aladdin)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/6c22c15ccbf24e64?rid=80)

    python API
    real=T3(close, timeperiod=5, vfactor=0)
    :return:
    """
    close = df["Close"]
    return talib.T3(close, timeperiod=time_period, vfactor=v_factor)


def TEMA(df, time_period=30):
    """
    Function Name: TEMA (Difference from T3?)
    Name: Triple Exponential Moving Average

    python API
    real=TEMA(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.TEMA(close, timeperiod=time_period)


def TRIMA(df, time_period=30):
    """
    python API
    real=TRIMA(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.TRIMA(close, timeperiod=time_period)


def WMA(df, time_period=30):
    """
    Function Name: WMA
    Name: Weighted Moving Average

    Introduction: The weighted moving average method calculates a weighted average unit cost by adding the cost of each purchase to the cost of existing inventory and dividing by the total number of units (purchased units plus existing inventory units). This average cost is then used to determine the cost of goods sold and ending inventory.

    Analysis and Application:
    [Baidu Baike](https://baike.baidu.com/item/%E7%A7%BB%E5%8A%A8%E5%8A%A0%E6%9D%83%E5%B9%B3%E5%9D%87%E6%B3%95/10056490?fr=aladdin&fromid=16799870&fromtitle=%E5%8A%A0%E6%9D%83%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87)
    [Tonghuashun Academy](http://www.iwencai.com/yike/detail/auid/262b1dfd1c68ee30)

    python API
    real=WMA(close, timeperiod=30)
    :return:
    """
    close = df["Close"]
    return talib.WMA(close, timeperiod=time_period)


################
# TA-LIB
# Pattern Recognition Functions
#################


def CDL2CROWS(df):
    """
    Function Name: CDL2CROWS
    Name: Two Crows

    Introduction: A three-day K-line pattern. The first day is a long positive line. The second day opens higher and closes with a negative line. The third day opens higher again and continues to close with a negative line, with the closing price lower than the previous day's closing price, indicating a price decline.

    python API
    integer=CDL2CROWS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL2CROWS(open, high, low, close)


def CDL3BLACKCROWS(df):
    """
    Function Name: CDL3BLACKCROWS
    Name: Three Black Crows

    Introduction: A three-day K-line pattern with three consecutive negative lines. Each day's closing price is lower and near the lowest price, and each day's opening price is within the previous day's body, indicating a price decline.

    python API
    integer=CDL3BLACKCROWS(open, high, low, close)
    :param df:
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL3BLACKCROWS(open, high, low, close)


def CDL3INSIDE(df):
    """
    Function Name: CDL3INSIDE
    Name: Three Inside Up/Down

    Introduction: A three-day K-line pattern, a mother-child signal + a long K-line. Taking Three Inside Up as an example, the K-lines are negative, positive, positive. The third day's closing price is higher than the first day's opening price, and the second day's K-line is inside the first day's K-line, indicating a price increase.

    python API
    integer=CDL3INSIDE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL3INSIDE(open, high, low, close)


def CDL3LINESTRIKE(df):
    """
    Function Name: CDL3LINESTRIKE
    Name: Three-Line Strike

    Introduction: A four-day K-line pattern. The first three are positive lines, and each day's closing price is higher than the previous day's, with the opening price within the previous day's body. On the fourth day, the market opens higher, and the closing price is lower than the first day's opening price, indicating a price decline.

    python API
    integer=CDL3LINESTRIKE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL3LINESTRIKE(open, high, low, close)


def CDL3OUTSIDE(df):
    """
    Function Name: CDL3OUTSIDE
    Name: Three Outside Up/Down

    Introduction: A three-day K-line pattern, similar to Three Inside Up/Down. The K-lines are negative, positive, positive. However, the first and second day's K-line patterns are reversed. Taking Three Outside Up as an example, the first day's K-line is inside the second day's K-line, indicating a price increase.

    python API
    integer=CDL3OUTSIDE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL3OUTSIDE(open, high, low, close)


def CDL3STARSINSOUTH(df):
    """
    Function Name: CDL3STARSINSOUTH
    Name: Three Stars In The South

    Introduction: A three-day K-line pattern, the opposite of Advance Block. All three K-lines are negative. The first day has a long lower shadow. The second day is similar to the first day, but the K-line is smaller overall. The third day has no lower shadow and is a small body signal. The trading prices are all within the first day's range, indicating a reversal of the downtrend and a price increase.

    python API
    integer=CDL3STARSINSOUTH(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL3STARSINSOUTH(open, high, low, close)


def CDL3WHITESOLDIERS(df):
    """
    Function Name: CDL3WHITESOLDIERS
    Name: Three Advancing White Soldiers

    Introduction: A three-day K-line pattern. All three K-lines are positive. Each day's closing price is higher and near the highest price, and the opening price is in the upper half of the previous day's body, indicating a price increase.

    python API
    integer=CDL3WHITESOLDIERS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDL3WHITESOLDIERS(open, high, low, close)


def CDLABANDONEDBABY(df):
    """
    Function Name: CDLABANDONEDBABY
    Name: Abandoned Baby

    Introduction: A three-day K-line pattern. The second day's price gaps and closes with a Doji (the opening price and closing price are close, and the difference between the highest and lowest prices is small), indicating a trend reversal. It occurs as a decline at the top and a rise at the bottom.

    python API
    integer=CDLABANDONEDBABY(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLABANDONEDBABY(open, high, low, close)


def CDLADVANCEBLOCK(df):
    """
    Function Name: CDLADVANCEBLOCK
    Name: Advance Block

    Introduction: A three-day K-line pattern. All three days close with positive lines. Each day's closing price is higher than the previous day's, and the opening price is within the previous day's body. The body becomes shorter, and the upper shadow becomes longer.

    python API
    integer=CDLADVANCEBLOCK(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLADVANCEBLOCK(open, high, low, close)


def CDLBELTHOLD(df):
    """
    Function Name: CDLBELTHOLD
    Name: Belt-hold

    Introduction: A two-day K-line pattern. In a downtrend, the first day is a negative line. The second day's opening price is the lowest price, a positive line, and the closing price is near the highest price, indicating a price increase.

    python API
    integer=CDLBELTHOLD(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLBELTHOLD(open, high, low, close)


def CDLBREAKAWAY(df):
    """
    Function Name: CDLBREAKAWAY
    Name: Breakaway

    Introduction: A five-day K-line pattern. Taking a bullish breakaway as an example, in a downtrend, the first day is a long negative line, the second day is a gap-down negative line, the trend continues to fluctuate, and the fifth day is a long positive line. The closing price is between the first day's closing price and the second day's opening price, indicating a price increase.

    python API
    integer=CDLBREAKAWAY(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLBREAKAWAY(open, high, low, close)


def CDLCLOSINGMARUBOZU(df):
    """
    Function Name: CDLCLOSINGMARUBOZU
    Name: Closing Marubozu

    Introduction: A one-day K-line pattern. Taking a positive line as an example, the lowest price is lower than the opening price, and the closing price is equal to the highest price, indicating a continuation of the trend.

    python API
    integer=CDLCLOSINGMARUBOZU(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLCLOSINGMARUBOZU(open, high, low, close)


def CDLCONCEALBABYSWALL(df):
    """
     Function Name: CDLCONCEALBABYSWALL
    Name: Concealing Baby Swallow

    Introduction: A four-day K-line pattern. In a downtrend, the first two days are negative lines with no shadows. The second day's opening and closing prices are both lower than the second day's. The third day is an inverted hammer. The fourth day's opening price is higher than the previous day's highest price, and the closing price is lower than the previous day's lowest price, indicating a bottom reversal.

    python API
    integer=CDLCONCEALBABYSWALL(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLCONCEALBABYSWALL(open, high, low, close)


def CDLCOUNTERATTACK(df):
    """
    Function Name: CDLCOUNTERATTACK
    Name: Counterattack
    Introduction: Two-day K-line pattern, similar to separating lines.

    python API
    integer=CDLCOUNTERATTACK(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLCOUNTERATTACK(open, high, low, close)


def CDLDARKCLOUDCOVER(df):
    """
    Function Name: CDLDARKCLOUDCOVER
    Name: Dark Cloud Cover

    Introduction: Two-day K-line pattern. The first day is a long positive line. The second day opens above the highest price of the previous day, and the closing price is below the middle of the previous day's body, indicating a price decline.

    python API
    integer=CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLDARKCLOUDCOVER(open, high, low, close)


def CDLDOJI(df):
    """
    Function Name: CDLDOJI
    Name: Doji

    Introduction: One-day K-line pattern. The opening price and closing price are basically the same.

    python API
    integer=CDLDOJI(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLDOJI(open, high, low, close)


def CDLDOJISTAR(df):
    """
    Function Name: CDLDOJISTAR
    Name: Doji Star

    Introduction: One-day K-line pattern. The opening price and closing price are basically the same, and the upper and lower shadows are not very long, indicating a reversal of the current trend.

    python API
    integer=CDLDOJISTAR(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLDOJISTAR(open, high, low, close)


def CDLDRAGONFLYDOJI(df):
    """
    Function Name: CDLDRAGONFLYDOJI
    Name: Dragonfly Doji

    Introduction: One-day K-line pattern. After opening, the price goes down all the way, and then recovers. The closing price is the same as the opening price, indicating a trend reversal.

    python API
    integer=CDLDRAGONFLYDOJI(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLDRAGONFLYDOJI(open, high, low, close)


def CDLENGULFING(df):
    """
    Function Name: CDLENGULFING
    Name: Engulfing Pattern

    Introduction: Two-day K-line pattern, divided into bullish engulfing and bearish engulfing, which are opposite to each other. Taking bullish engulfing as an example, the first day is a negative line, and the second day is a positive line. The opening price and closing price of the first day are within the opening and closing prices of the second day, but they cannot be exactly the same.

    python API
    integer=CDLENGULFING(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLENGULFING(open, high, low, close)


def CDLEVENINGDOJISTAR(df):
    """
    Function Name: CDLEVENINGDOJISTAR
    Name: Evening Doji Star

    Introduction: Three-day K-line pattern. The basic pattern is Evening Star. The closing price and opening price of the second day are the same, indicating a top reversal.

    python API
    integer=CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLEVENINGDOJISTAR(open, high, low, close)


def CDLEVENINGSTAR(df):
    """
    Function Name: CDLEVENINGSTAR
    Name: Evening Star

    Introduction: Three-day K-line pattern, opposite to Morning Star. In an upward trend, the first day is a positive line, the second day has a small price fluctuation, and the third day is a negative line, indicating a top reversal.

    python API
    integer=CDLEVENINGSTAR(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLEVENINGSTAR(open, high, low, close)


def CDLGAPSIDESIDEWHITE(df):
    """
    Function Name: CDLGAPSIDESIDEWHITE
    Name: Up/Down - gap side - by - side white lines

    Introduction: Two-day K-line pattern. In an upward trend, there is a gap up. In a downward trend, there is a gap down. The first day and the second day have the same opening price, and the body lengths are similar, then the trend continues.

    python API
    integer=CDLGAPSIDESIDEWHITE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLGAPSIDESIDEWHITE(open, high, low, close)


def CDLGRAVESTONEDOJI(df):
    """
    Function Name: CDLGRAVESTONEDOJI
    Name: Gravestone Doji

    Introduction: One-day K-line pattern. The opening price and closing price are the same, with a long upper shadow and no lower shadow, indicating a bottom reversal.

    python API
    integer=CDLGRAVESTONEDOJI(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLGRAVESTONEDOJI(open, high, low, close)


def CDLHAMMER(df):
    """
    Function Name: CDLHAMMER
    Name: Hammer

    Introduction: One-day K-line pattern. The body is short, there is no upper shadow, and the lower shadow is more than twice the length of the body. It is at the bottom of a downward trend, indicating a reversal.

    python API
    integer=CDLHAMMER(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHAMMER(open, high, low, close)


def CDLHANGINGMAN(df):
    """
    Function Name: CDLHANGINGMAN
    Name: Hanging Man

    Introduction: One-day K-line pattern, similar in shape to the Hammer, at the top of an upward trend, indicating a trend reversal.

    python API
    integer=CDLHANGINGMAN(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHANGINGMAN(open, high, low, close)


def CDLHARAMI(df):
    """
    Function Name: CDLHARAMI
    Name: Harami Pattern

    Introduction: Two-day K-line pattern, divided into bullish harami and bearish harami, which are opposite to each other. Taking bullish harami as an example, in a downward trend, the first K-line is a long negative line, and the second day's opening and closing prices are within the price range of the first day, and it is a positive line, indicating a trend reversal and a price increase.

    python API
    integer=CDLHARAMI(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHARAMI(open, high, low, close)


def CDLHARAMICROSS(df):
    """
    Function Name: CDLHARAMICROSS
    Name: Harami Cross Pattern

    Introduction: Two-day K-line pattern, similar to the Harami pattern. If the second K-line is a Doji, it is called a Harami Cross, indicating a trend reversal.

    python API
    integer=CDLHARAMICROSS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHARAMICROSS(open, high, low, close)


def CDLHIGHWAVE(df):
    """
    Function Name: CDLHIGHWAVE
    Name: High - Wave Candle

    Introduction: Three-day K-line pattern with very long upper/lower shadows and a short body, indicating a trend reversal.

    python API
    integer=CDLHIGHWAVE(open, high, low, close)
    :return:
    """
    close = df["Close"]
    return talib.CDLHIGHWAVE(open, high, low, close)


def CDLHIKKAKE(df):
    """
    Function Name: CDLHIKKAKE
    Name: Hikkake Pattern

    Introduction: Three-day K-line pattern, similar to Harami. The price of the second day is within the body of the previous day. The closing price of the third day is higher than the previous two days. The reversal fails and the trend continues.

    python API
    integer=CDLHIKKAKE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHIKKAKE(open, high, low, close)


def CDLHIKKAKEMOD(df):
    """
    Function Name: CDLHIKKAKEMOD
    Name: Modified Hikkake Pattern

    Introduction: Three-day K-line pattern, similar to Hikkake. In an upward trend, the third day gaps up; in a downward trend, the third day gaps down. The reversal fails and the trend continues.

    python API
    integer=CDLHIKKAKEMOD(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHIKKAKEMOD(open, high, low, close)


def CDLHOMINGPIGEON(df):
    """
    Function Name: CDLHOMINGPIGEON
    Name: Homing Pigeon

    Introduction: Two-day K-line pattern, similar to the Harami pattern. The difference is that the two K-lines have the same color. The highest and lowest prices of the second day are within the body of the first day, indicating a trend reversal.

    python API
    integer=CDLHOMINGPIGEON(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLHOMINGPIGEON(open, high, low, close)


def CDLIDENTICAL3CROWS(df):
    """
    Function Name: CDLIDENTICAL3CROWS
    Name: Identical Three Crows

    Introduction: Three-day K-line pattern. In an upward trend, all three days are negative lines, with approximately the same length. The opening price of each day is equal to the closing price of the previous day, and the closing price is close to the lowest price of the day, indicating a price decline.

    python API
    integer=CDLIDENTICAL3CROWS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLIDENTICAL3CROWS(open, high, low, close)


def CDLINNECK(df):
    """
    Function Name: CDLINNECK
    Name: In - Neck Pattern

    Introduction: Two-day K-line pattern. In a downward trend, the first day is a long negative line. The second day opens lower and closes slightly above the closing price of the first day. It is a positive line with a short body, indicating that the decline continues.

    python API
    integer=CDLINNECK(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLINNECK(open, high, low, close)


def CDLINVERTEDHAMMER(df):
    """
    Function Name: CDLINVERTEDHAMMER
    Name: Inverted Hammer

    Introduction: One-day K-line pattern. The upper shadow is long, more than twice the length of the body, and there is no lower shadow. It is at the bottom of a downward trend, indicating a trend reversal.

    python API
    integer=CDLINVERTEDHAMMER(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLINVERTEDHAMMER(open, high, low, close)


def CDLKICKING(df):
    """
    Function Name: CDLKICKING
    Name: Kicking

    Introduction: Two-day K-line pattern, similar to separating lines. The two K-lines are marubozu lines with opposite colors and a gap.

    python API
    integer=CDLKICKING(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLKICKING(open, high, low, close)


def CDLKICKINGBYLENGTH(df):
    """
    Function Name: CDLKICKINGBYLENGTH
    Name: Kicking - bull/bear determined by the longer marubozu

    Introduction: Two-day K-line pattern, similar to Kicking. The longer marubozu determines the price increase or decrease.

    python API
    integer=CDLKICKINGBYLENGTH(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLKICKINGBYLENGTH(open, high, low, close)


def CDLLADDERBOTTOM(df):
    """
    Function Name: CDLLADDERBOTTOM
    Name: Ladder Bottom

    Introduction: Five-day K-line pattern. In a downward trend, the first three days are negative lines, and the opening and closing prices are lower than the opening and closing prices of the previous day. The fourth day is an inverted hammer. The fifth day opens higher than the opening price of the previous day, is a positive line, and the closing price is higher than the price range of the previous few days, indicating a bottom reversal.

    python API
    integer=CDLLADDERBOTTOM(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLLADDERBOTTOM(open, high, low, close)


def CDLLONGLEGGEDDOJI(df):
    """
    Function Name: CDLLONGLEGGEDDOJI
    Name: Long Legged Doji

    Introduction: One-day K-line pattern. The opening price and closing price are the same and in the middle of the day's price. Long upper and lower shadows indicate market uncertainty.

    python API
    integer=CDLLONGLEGGEDDOJI(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLLONGLEGGEDDOJI(open, high, low, close)


def CDLLONGLINE(df):
    """
    Function Name: CDLLONGLINE
    Name: Long Line Candle

    Introduction: One-day K-line pattern. The K-line body is long, with no upper and lower shadows.

    python API
    integer=CDLLONGLINE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLLONGLINE(open, high, low, close)


def CDLMARUBOZU(df):
    """
    Function Name: CDLMARUBOZU
    Name: Marubozu

    Introduction: One-day K-line pattern. There are no shadows on the body. A negative line indicates a continuation of the bear market or a bull market reversal, and vice versa for a positive line.

    python API
    integer=CDLMARUBOZU(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLMARUBOZU(open, high, low, close)


def CDLMATCHINGLOW(df):
    """
    Function Name: CDLMATCHINGLOW
    Name: Matching Low

    Introduction: Two-day K-line pattern. In a downward trend, the first day is a long negative line, and the second day is a negative line. The closing price is the same as the previous day, indicating bottom confirmation, and this price is the support level.

    python API
    integer=CDLMATCHINGLOW(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLMATCHINGLOW(open, high, low, close)


def CDLMATHOLD(df):
    """
    Function Name: CDLMATHOLD
    Name: Mat Hold

    Introduction: Five-day K-line pattern. In an upward trend, the first day is a positive line, the second day gaps up, the third and fourth days are short-bodied shadows, and the fifth day is a positive line. The closing price is higher than the previous four days, indicating that the trend continues.

    python API
    integer=CDLMATHOLD(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLMATHOLD(open, high, low, close)


def CDLMORNINGDOJISTAR(df):
    """
    Function Name: CDLMORNINGDOJISTAR
    Name: Morning Doji Star

    Introduction: Three-day K-line pattern. The basic pattern is Morning Star. The second K-line is a Doji star, indicating a bottom reversal.

    python API
    integer=CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLMORNINGDOJISTAR(open, high, low, close)


def CDLMORNINGSTAR(df):
    """
    Function Name: CDLMORNINGSTAR
    Name: Morning Star

    Introduction: Three-day K-line pattern. In a downward trend, the first day is a negative line, the second day has a small price fluctuation, and the third day is a positive line, indicating a bottom reversal.

    python API
    integer=CDLMORNINGSTAR(open, high, low, close, penetration=0)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLMORNINGSTAR(open, high, low, close)


def CDLONNECK(df):
    """
    Function Name: CDLONNECK
    Name: On - Neck Pattern

    Introduction: Two-day K-line pattern. In a downward trend, the first day is a long negative line. The second day opens lower and the closing price is the same as the lowest price of the previous day. It is a positive line with a short body, indicating a continuation of the downward trend.

    python API
    integer=CDLONNECK(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLONNECK(open, high, low, close)


def CDLPIERCING(df):
    """
    Function Name: CDLPIERCING
    Name: Piercing Pattern

    Introduction: Two-day K-line pattern. In a downward trend, the first day is a negative line. The second day opens below the lowest price of the previous day, and the closing price is in the upper part of the body of the first day, indicating a bottom reversal.

    python API
    integer=CDLPIERCING(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLPIERCING(open, high, low, close)


def CDLRICKSHAWMAN(df):
    """
    Function Name: CDLRICKSHAWMAN
    Name: Rickshaw Man

    Introduction: One-day K-line pattern, similar to Long Legged Doji. If the body is exactly at the midpoint of the price range, it is called Rickshaw Man.

    python API
    integer=CDLRICKSHAWMAN(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLRICKSHAWMAN(open, high, low, close)


def CDLRISEFALL3METHODS(df):
    """
     Function Name: CDLRISEFALL3METHODS
    Name: Rising/Falling Three Methods

    Introduction: Five-day K-line pattern. Taking Rising Three Methods as an example, in an upward trend, the first day is a long positive line, the middle three days have small fluctuations within the range of the first day, and the fifth day is a long positive line. The closing price is higher than the closing price of the first day, indicating a price increase.

    python API
    integer=CDLRISEFALL3METHODS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLRISEFALL3METHODS(open, high, low, close)


def CDLSEPARATINGLINES(df):
    """
     Function Name: CDLSEPARATINGLINES
    Name: Separating Lines

    Introduction: Two-day K-line pattern. In an upward trend, the first day is a negative line, and the second day is a positive line. The opening price of the second day is the same as the first day and is the lowest price, indicating that the trend continues.

    python API
    integer=CDLSEPARATINGLINES(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLSEPARATINGLINES(open, high, low, close)


def CDLSHOOTINGSTAR(df):
    """
    Function Name: CDLSHOOTINGSTAR
    Name: Shooting Star

    Introduction: One-day K-line pattern. The upper shadow is at least twice the length of the body, and there is no lower shadow, indicating a price decline.

    python API
    integer=CDLSHOOTINGSTAR(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLSHOOTINGSTAR(open, high, low, close)


def CDLSHORTLINE(df):
    """
    Function Name: CDLSHORTLINE
    Name: Short Line Candle

    Introduction: One-day K-line pattern. The body is short, with no upper and lower shadows.

    python API
    integer=CDLSHORTLINE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLSHORTLINE(open, high, low, close)


def CDLSPINNINGTOP(df):
    """
    Function Name: CDLSPINNINGTOP
    Name: Spinning Top

    Introduction: One-day K-line, small body.

    python API
    integer=CDLSPINNINGTOP(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLSPINNINGTOP(open, high, low, close)


def CDLSTALLEDPATTERN(df):
    """
    Function Name: CDLSTALLEDPATTERN
    Name: Stalled Pattern

    Introduction: Three-day K-line pattern. In an upward trend, the second day is a long positive line, and the third day opens near the closing price of the previous day and is a short positive line, indicating the end of the rise.

    python API
    integer=CDLSTALLEDPATTERN(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLSTALLEDPATTERN(open, high, low, close)


def CDLSTICKSANDWICH(df):
    """
    Function Name: CDLSTICKSANDWICH
    Name: Stick Sandwich

    Introduction: Three-day K-line pattern. The first day is a long negative line, the second day is a positive line, the opening price is higher than the closing price of the previous day, and the third day opens higher than the highest prices of the previous two days, and the closing price is the same as the closing price of the first day.

    python API
    integer=CDLSTICKSANDWICH(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLSTICKSANDWICH(open, high, low, close)


def CDLTAKURI(df):
    """
    Function Name: CDLTAKURI
    Name: Takuri(Dragonfly Doji with very long lower shadow)

    Introduction: One-day K-line pattern, roughly the same as Dragonfly Doji, with a long lower shadow.

    python API
    integer=CDLTAKURI(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLTAKURI(open, high, low, close)


def CDLTASUKIGAP(df):
    """
    Function Name: CDLTASUKIGAP
    Name: Tasuki Gap

    Introduction: Three-day K-line pattern, divided into rising and falling. Taking rising as an example, the first two days are positive lines, the second day gaps up, and the third day is a negative line. The closing price is in the gap, and the upward trend continues.

    python API
    integer=CDLTASUKIGAP(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLTASUKIGAP(open, high, low, close)


def CDLTHRUSTING(df):
    """
    Function Name: CDLTHRUSTING
    Name: Thrusting Pattern

    Introduction: Two-day K-line pattern, similar to the In-Neck pattern. In a downward trend, the first day is a long negative line. The second day opens with a gap, and the closing price is slightly lower than the middle of the previous day's body. Compared with the In-Neck pattern, the body is longer, indicating that the trend continues.

    python API
    integer=CDLTHRUSTING(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLTHRUSTING(open, high, low, close)


def CDLTRISTAR(df):
    """
    Function Name: CDLTRISTAR
    Name: Tristar Pattern

    Introduction: Three-day K-line pattern, composed of three Dojis. The second Doji must be higher or lower than the first and third, indicating a reversal.

    python API
    integer=CDLTRISTAR(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLTRISTAR(open, high, low, close)


def CDLUNIQUE3RIVER(df):
    """
    Function Name: CDLUNIQUE3RIVER
    Name: Unique 3 River

    Introduction: Three-day K-line pattern. In a downward trend, the first day is a long negative line, the second day is a hammer, the lowest price hits a new low, the third day opens lower than the closing price of the second day, is a positive line, and the closing price is not higher than the closing price of the second day, indicating a reversal. The longer the lower shadow of the second day, the greater the possibility.

    python API
    integer=CDLUNIQUE3RIVER(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLUNIQUE3RIVER(open, high, low, close)


def CDLUPSIDEGAP2CROWS(df):
    """
    Function Name: CDLUPSIDEGAP2CROWS
    Name: Upside Gap Two Crows

    Introduction: Three-day K-line pattern. The first day is a positive line. The second day gaps up and opens higher than the highest price of the first day, and closes as a negative line. The third day opens higher than the second day, closes as a negative line, and there is still a gap compared with the first day.

    python API
    integer=CDLUPSIDEGAP2CROWS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLUPSIDEGAP2CROWS(open, high, low, close)


def CDLXSIDEGAP3METHODS(df):
    """
    Function Name: CDLXSIDEGAP3METHODS
    Name: Upside/Downside Gap Three Methods

    Introduction: Five-day K-line pattern. Taking the Upside Gap Three Methods as an example, in an upward trend, the first day is a long positive line, the second day is a short positive line, the third day is a gap up positive line, the fourth day is a negative line, the opening and closing prices are within the bodies of the previous two days, and the fifth day is a long positive line. The closing price is higher than the closing price of the first day, indicating a price increase.

    python API
    integer=CDLXSIDEGAP3METHODS(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.CDLXSIDEGAP3METHODS(open, high, low, close)


################
# TA-LIB Quantitative Analysis Data
# Price Transform Functions
#################


def AVGPRICE(df):
    """
    Function Name: AVGPRICE
    Name: Average Price Function

    python API
    real=AVGPRICE(open, high, low, close)
    :return:
    """
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.AVGPRICE(open, high, low, close)


def MEDPRICE(df):
    """
    Function Name: MEDPRICE
    Name: Median Price

    python API
    real=MEDPRICE(high, low)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.MEDPRICE(high, low)


def TYPPRICE(df):
    """
    Function Name: TYPPRICE
    Name: Typical Price

    python API
    real=TYPPRICE(high, low, close)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.TYPPRICE(high, low, close)


def WCLPRICE(df):
    """
    Function Name: WCLPRICE
    Name: Weighted Close Price

    python API
    real=WCLPRICE(high, low, close)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.WCLPRICE(high, low, close)


################
# TA-LIB
# Statistic Functions
#################


def BETA(df, time_period=5):
    """
    Function Name: BETA
    Name: Beta Coefficient

    Introduction: A risk index used to measure the price volatility of individual stocks or
    stock funds relative to the overall market.
    The beta coefficient measures the overall volatility of a stock's return relative to the benchmark return. It is a relative indicator. A higher beta means that the stock is more volatile relative to the benchmark. A beta greater than 1 indicates
    that the stock's volatility is greater than the benchmark's volatility, and vice versa.

    Uses:
    1) Calculate the cost of capital and make investment decisions (only projects with returns higher than the cost of capital should be invested in);
    2) Calculate the cost of capital and formulate performance assessment and incentive standards;
    3) Calculate the cost of capital and conduct asset valuation (Beta is the basis of the discounted cash flow model);
    4) Determine the systematic risk of a single asset or portfolio, and use it for portfolio investment management, especially for hedging (or speculation) of stock index futures or other financial derivatives.

    python API
    real=BETA(high, low, timeperiod=5)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.BETA(high, low, timeperiod=time_period)


def CORREL(df, time_period=30):
    """
    Function Name: CORREL
    Name: Pearson Correlation Coefficient

    Introduction: Used to measure the correlation (linear correlation) between two variables X and Y, with a value between -1 and 1.
    The Pearson correlation coefficient is a measure of the correlation between two variables. It is a value between 1 and -1,
    where 1 indicates a perfect positive correlation, 0 indicates no correlation, and -1 indicates a perfect negative correlation.

    python API
    real=CORREL(high, low, timeperiod=30)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    return talib.CORREL(high, low, timeperiod=time_period)


def LINEARREG(df, time_period=14):
    """
    Linear Regression Equation: When two variables x and y have a significant linear correlation, the least squares method is used to determine the linear equation y=a+bx of an optimal straight line. The distance between this regression line and the correlation points is smaller than the distance between any other straight line and the correlation points, making it the best ideal straight line.
    Regression Intercept a: Represents the intercept of the straight line on the y-axis, representing the starting point of the straight line.
    Regression Coefficient b: Represents the slope of the straight line. Its practical meaning is to indicate the average change in y for every unit change in x.
    That is, for every 1 unit increase in x, y changes by b units.

    Function Name: LINEARREG
    Name: Linear Regression
    Introduction: A statistical analysis method used to determine the quantitative relationship between two or more variables.
    Its expression is y=w'x+e, where e is the error and follows a normal distribution with a mean of 0.

    python API
    real=LINEARREG(close, timeperiod=14)
    :return:
    """
    close = df["Close"]
    return talib.LINEARREG(close, timeperiod=time_period)


def LINEARREG_ANGLE(df, time_period=14):
    """
    Function Name: LINEARREG_ANGLE
    Name: Linear Regression Angle

    Introduction: Used to determine the angle change of the price.
    [Reference](http://blog.sina.com.cn/s/blog_14c9f45b20102vv8p.md)

    python API
    real=LINEARREG_ANGLE(close, timeperiod=14)
    :return:
    """
    close = df["Close"]
    return talib.LINEARREG_ANGLE(close, timeperiod=time_period)


def LINEARREG_INTERCEPT(df, time_period=14):
    """
    Function Name: LINEARREG_INTERCEPT
    Name: Linear Regression Intercept

    python API
    real=LINEARREG_INTERCEPT(close, timeperiod=14)
    :return:
    """
    close = df["Close"]
    return talib.LINEARREG_INTERCEPT(close, timeperiod=time_period)


def LINEARREG_SLOPE(df, time_period=14):
    """
    Function Name: LINEARREG_SLOPE
    Name: Linear Regression Slope Indicator

    python API
    real=LINEARREG_SLOPE(close, timeperiod=14)
    :return:
    """
    close = df["Close"]
    return talib.LINEARREG_SLOPE(close, timeperiod=time_period)


def STDDEV(df, time_period=5, nb_dev=1):
    """
    Function Name: STDDEV
    Name: Standard Deviation
    Introduction: A measure of the dispersion of data distribution, used to measure the degree to which data values deviate from the arithmetic mean. The smaller the standard deviation, the less these values deviate from the mean, and vice versa. The magnitude of the standard deviation can be measured by the multiple relationship between the standard deviation and the mean.

    python API
    real=STDDEV(close, timeperiod=5, nbdev=1)
    :return:
    """
    close = df["Close"]
    return talib.STDDEV(close, timeperiod=time_period, nbdev=nb_dev)


def TSF(df, time_period=14):
    """
    Function Name: TSF
    Name: Time Series Forecast
    Introduction: A type of historical data extrapolation prediction, also known as historical extrapolation prediction. It is a method of extrapolating and predicting the development trend of socioeconomic phenomena based on the development process and regularity reflected by the time series.

    python API
    real=TSF(close, timeperiod=14)
    :return:
    """
    close = df["Close"]
    return talib.TSF(close, timeperiod=time_period)


def VAR(df, time_period=5, nb_dev=1):
    """
    Function Name: VAR
    Name: Variance
    Introduction: Variance is used to calculate the difference between each variable (observed value) and the overall mean. To avoid the sum of deviations from the mean being zero and the sum of squared deviations from the mean being affected by the sample size, statistics uses the average of the squared deviations from the mean to describe the degree of variation of the variable.

    python API
    real=VAR(close, timeperiod=5, nbdev=1)
    :return:
    """
    close = df["Close"]
    return talib.VAR(close, timeperiod=time_period, nbdev=nb_dev)


################
# TA-LIB
# Volatility Indicator Functions
#################


def ATR(df, time_period=14):
    """
    Function Name: ATR
    Name: Average True Range
    Introduction: The Average True Range (ATR) is
    the N-day exponentially smoothed moving average of the trading range.
    Calculation Formula: A day's trading range is simply the maximum - minimum.
    The True Range includes yesterday's closing price if it is outside today's range:
    True Range = max(maximum, yesterday's closing price) - min(minimum, yesterday's closing price) The Average True Range is the N-day exponential moving average of the "True Range".

    Characteristics:
    * The concept of range can show the trader's expectations and enthusiasm.
    * A large or increasing range indicates that traders may be prepared to continue buying or selling the stock during the day.
    * A decrease in range indicates that traders are not very interested in the market.

    python API
    real=ATR(high, low, close, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.ATR(high, low, close, timeperiod=time_period)


def NATR(df, time_period=14):
    """
    Function Name: NATR
    Name: Normalized Average True Range

    Introduction: The Normalized Average True Range (NATR) is

    python API
    real=NATR(high, low, close, timeperiod=14)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.NATR(high, low, close, timeperiod=time_period)


def TRANGE(df):
    """
    Function Name: TRANGE
    Name: True Range

    python API
    real=TRANGE(high, low, close)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    return talib.TRANGE(high, low, close)


################
# TA-LIB
# Volume Indicators
#################


def AD(df):
    """
    Function Name: AD
    Name: Chaikin A/D Line (Accumulation/Distribution Line)
    Introduction: Proposed by Marc Chaikin, this is a volume-based indicator that uses the closing price to estimate the trading volume for the day. It is used to estimate the cumulative flow of funds into or out of a security over a period of time.

    Calculation Formula:
    A/D = Yesterday's A/D + (Bull/Bear Comparison) * Today's Volume
    Bull/Bear Comparison = [(Close - Low) - (High - Close)] / (High - Low)
    If High equals Low: Bull/Bear Comparison = (Close / Yesterday's Close) - 1

    Analysis and Application:
    1. A/D measures the direction of fund flow. An upward A/D indicates that buyers are dominant, while a downward A/D indicates that sellers are dominant.
    2. Divergence between A/D and price can be considered as a trading signal. Bullish divergence suggests buying, and bearish divergence suggests selling.
    3. It should be noted that A/D ignores the impact of gaps. In fact, the significance of gaps should not be easily ignored.
    The A/D indicator does not require parameter settings, but in application, it can be analyzed in conjunction with the indicator's moving average.

    Python API
    real=AD(high, low, close, volume)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]
    return talib.AD(high, low, close, volume)


def ADOSC(df, fast_period=3, slow_period=10):
    """
    Function Name: ADOSC
    Name: Chaikin A/D Oscillator
    Introduction: Compares fund flow with price action to detect the inflow and outflow of funds in the market.

    Calculation Formula: fastperiod A/D - slowperiod A/D

    Analysis and Application:
    1. Trading signals are based on divergence: bullish divergence for long positions, bearish divergence for short positions.
    2. Combine stock price with 90-day moving average, and with other indicators.
    3. Sell when it changes from positive to negative, and buy when it changes from negative to positive.

    Python API
    real=ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    :return:
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]
    return talib.ADOSC(
        high, low, close, volume, fastperiod=fast_period, slowperiod=slow_period
    )


def OBV(df):
    """
    Function Name: OBV
    Name: On Balance Volume
    Introduction: Proposed by Joe Granville, it infers price trends by statistically analyzing the trend of volume changes.

    Calculation Formula: Starting from a certain day as the base period, the daily total trading volume of listed stocks is accumulated. If the index or stock price rises the next day,
    then the base period OBV plus today's trading volume is today's OBV. If the index or stock price falls the next day,
    then the base period OBV minus today's trading volume is today's OBV.

    Analysis and Application:
    1. Using "N" shaped fluctuations as the unit of fluctuation, a wave higher than the previous wave is called an "uptrend", and a wave lower than the previous is called a "downtrend"; buy during uptrends and sell during downtrends.
    2. Must be used in conjunction with the K-line chart.
    3. It is corrected using the net bull/bear ratio method, but it is not clear which method TA-Lib uses.

    Calculation formula: Net Bull/Bear Ratio = [(Close - Low) - (High - Close)] / (High - Low) * Volume

    Python API
    real=OBV(close, volume)
    :return:
    """
    close = df["Close"]
    volume = df["Volume"]
    return talib.OBV(close, volume)


TICKER = "MSFT"

company_ticker = yf.Ticker(TICKER)

df = company_ticker.history(period="1y", interval="1d")

open, close, high, low, volume = df.Open, df.Close, df.High, df.Low, df.Volume

# Choose two important metrics: ADOSC (Chaikin A/D Oscillator) and OBV (On Balance Volume)

# Calculate ADOSC
adosc = ADOSC(df)

# Calculate OBV
obv = OBV(df)

# Create subplots for ADOSC and OBV
apds = [
    mpf.make_addplot(adosc, panel=2, color="blue", title="Chaikin A/D Oscillator"),
    mpf.make_addplot(obv, panel=3, color="green", title="On Balance Volume"),
]

output_file = PLOTS_DIR / f"{TICKER}_candlestick_adosc_obv.png"
# Plot the candlestick chart with ADOSC and OBV
mpf.plot(
    df,
    type="candle",
    style="yahoo",
    volume=True,
    addplot=apds,
    title=f"{TICKER} - Candlestick, ADOSC, and OBV",
    savefig=output_file,
)
print(f"Plot saved to {output_file}")


data = yf.Ticker("MSFT").history(period="1y", interval="1d")

# 1. Trend Identification (200-day SMA)
data["SMA_200"] = talib.SMA(data["Close"], timeperiod=200)

# 2. Momentum Confirmation (RSI and MACD)
data["RSI"] = talib.RSI(data["Close"], timeperiod=14)
data["MACD"], data["MACD_Signal"], data["MACD_Histogram"] = talib.MACD(
    data["Close"], fastperiod=12, slowperiod=26, signalperiod=9
)

# 3. Volatility Filter (Bollinger Bands)
data["BB_Upper"], data["BB_Middle"], data["BB_Lower"] = talib.BBANDS(
    data["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
)


# Generate Trading Signals
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals["signal"] = 0  # 0: No signal, 1: Buy, -1: Sell

    for i in range(1, len(data)):
        # Buy Signal: Price above 200-day SMA, RSI oversold, MACD crossover
        if (
            data["Close"].iloc[i] > data["SMA_200"].iloc[i]
            and data["RSI"].iloc[i] < 30
            and data["MACD"].iloc[i] > data["MACD_Signal"].iloc[i]
        ):
            signals.iloc[i, signals.columns.get_loc("signal")] = 1

        # Sell Signal: Price below 200-day SMA, RSI overbought, MACD crossover
        elif (
            data["Close"].iloc[i] < data["SMA_200"].iloc[i]
            and data["RSI"].iloc[i] > 70
            and data["MACD"].iloc[i] < data["MACD_Signal"].iloc[i]
        ):
            signals.iloc[i, signals.columns.get_loc("signal")] = -1

        # Reset signal if outside criteria
        else:
            signals.iloc[i, signals.columns.get_loc("signal")] = 0

    return signals


signals = generate_signals(data.copy())  # Important to copy the dataframe

print(signals.index[signals["signal"] == 1])
