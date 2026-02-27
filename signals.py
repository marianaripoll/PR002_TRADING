import pandas as pd
import numpy as np




def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(close: pd.Series, period: int, num_std: float):
    """Returns (upper_band, middle_band, lower_band)."""
    middle = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return upper, middle, lower




def add_indicators(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute all three indicators and append their columns to the DataFrame.

    Parameters
    ----------
    data   : OHLCV DataFrame
    params : dict with keys:
               rsi_period    – RSI lookback (int)
               rsi_ob        – RSI overbought threshold (float, e.g. 70)
               rsi_os        – RSI oversold  threshold (float, e.g. 30)
               macd_fast     – MACD fast EMA period (int)
               macd_slow     – MACD slow EMA period (int)
               macd_signal   – MACD signal EMA period (int)
               bb_period     – Bollinger Bands SMA period (int)
               bb_std        – Bollinger Bands std multiplier (float)
    """
    df    = data.copy()
    close = df["Close"]

    # RSI
    df["rsi"] = _rsi(close, params["rsi_period"])

    # MACD
    df["macd_line"], df["macd_signal_line"], df["macd_hist"] = _macd(
        close, params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger_bands(
        close, params["bb_period"], params["bb_std"]
    )

    return df


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply the 2-out-of-3 confirmation rule to generate a 'signal' column.

    Individual indicator votes (state-based, fires every bar while condition holds)
    -------------------------------------------------------------------------------
    RSI:
        < rsi_os  → bullish (+1)   [oversold zone]
        > rsi_ob  → bearish (-1)   [overbought zone]
        else      → neutral ( 0)

    MACD histogram (state, not crossover):
        > 0  → bullish (+1)        [momentum is upward]
        < 0  → bearish (-1)        [momentum is downward]
        == 0 → neutral ( 0)

    Bollinger Bands (state):
        Close < bb_lower → bullish (+1)   [price below lower band]
        Close > bb_upper → bearish (-1)   [price above upper band]
        else             → neutral ( 0)

    Final signal (2-of-3 rule):
        sum >=  2  →  +1 (long)
        sum <= -2  →  -1 (short)
        else       →   0 (flat)

    Parameters
    ----------
    data   : DataFrame with indicator columns from add_indicators()
    params : dict with rsi_ob, rsi_os keys

    Returns
    -------
    pd.DataFrame with 'signal' column added (int: -1, 0, +1).
    """
    df     = data.copy()
    rsi_ob = params["rsi_ob"]
    rsi_os = params["rsi_os"]

    # RSI
    rsi_vote = pd.Series(0, index=df.index)
    rsi_vote[df["rsi"] < rsi_os] =  1
    rsi_vote[df["rsi"] > rsi_ob] = -1

    # MACD histogram
    macd_vote = pd.Series(0, index=df.index)
    macd_vote[df["macd_hist"] > 0] =  1
    macd_vote[df["macd_hist"] < 0] = -1

    # Bollinger Bands (state)
    bb_vote = pd.Series(0, index=df.index)
    bb_vote[df["Close"] < df["bb_lower"]] =  1
    bb_vote[df["Close"] > df["bb_upper"]] = -1


    total = rsi_vote + macd_vote + bb_vote

    df["signal"] = 0
    df.loc[total >=  2, "signal"] =  1
    df.loc[total <= -2, "signal"] = -1

    return df



DEFAULT_PARAMS = {
    # RSI
    "rsi_period": 14,
    "rsi_ob":     70.0,
    "rsi_os":     30.0,
    # MACD
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal":  9,
    # Bollinger Bands
    "bb_period": 20,
    "bb_std":     2.0,
    # Risk management
    "sl":    0.02,    # 2% stop-loss
    "tp":    0.04,    # 4% take-profit
    "n_btc": 0.01,    # BTC per trade
}