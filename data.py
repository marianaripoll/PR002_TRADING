import pandas as pd



TRAIN_PATH = "btc_project_train.csv"
TEST_PATH  = "btc_project_test.csv"

OHLC_COLUMNS   = ["Open", "High", "Low", "Close"]

KEEP_COLUMNS   = ["Open", "High", "Low", "Close", "Volume"]

DATETIME_COL   = "Datetime"




def load_data(path: str) -> pd.DataFrame:
    """
    Load a BTCUSDT 5-minute CSV and return a clean DataFrame.

    Processing steps
    ----------------
    1. Read CSV.
    2. Parse ``Datetime`` column and set it as a DatetimeIndex.
    3. Cast OHLC + Volume to float (coerce errors to NaN).
    4. Sort by time ascending and drop duplicate timestamps.
    5. Drop only rows where **OHLC** values are NaN (17 rows in test set).
       Volume NaNs are kept — Volume is not used in strategy logic.
    6. Keep only [Open, High, Low, Close, Volume].

    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Clean OHLCV DataFrame with a DatetimeIndex, integer-reset index
        NOT applied here (that is done inside backtest.py per-window).
    """
    df = pd.read_csv(path)

    missing = [c for c in OHLC_COLUMNS + [DATETIME_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.  Found: {list(df.columns)}"
        )

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.set_index(DATETIME_COL).sort_index()

    for col in KEEP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[available]

    df = df[~df.index.duplicated(keep="first")]

    before = len(df)
    df = df.dropna(subset=OHLC_COLUMNS)
    dropped = before - len(df)

    print(
        f"[data] Loaded '{path}': {len(df):,} rows  |  "
        f"{df.index[0]}  →  {df.index[-1]}  |  "
        f"dropped {dropped} NaN-OHLC rows"
    )

    _check_gaps(df, path)

    return df


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper that loads both the training and test sets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train = load_data(TRAIN_PATH)
    test  = load_data(TEST_PATH)
    return train, test



def _check_gaps(df: pd.DataFrame, path: str, threshold_hours: int = 2) -> None:
    """
    Warn if the DataFrame contains time gaps larger than ``threshold_hours``.
    This is informational only — no data is modified.
    """
    gaps = df.index.to_series().diff()
    big  = gaps[gaps > pd.Timedelta(hours=threshold_hours)]
    if not big.empty:
        print(
            f"  [data] WARNING: '{path}' contains {len(big)} gap(s) > {threshold_hours}h. "
            f"Largest gap: {gaps.max()}  at index {big.idxmax()}.  "
            f"Bar-based P&L is unaffected, but calendar period returns "
            f"should be interpreted carefully."
        )
