import numpy as np
import pandas as pd



BARS_PER_YEAR_5MIN = 365 * 24 * 12   # 5-min bars in a year (crypto, 24/7)



def _bar_returns(portfolio_values) -> np.ndarray:
    pv = np.array(portfolio_values, dtype=float)
    return np.diff(pv) / pv[:-1]


def _max_drawdown(portfolio_values) -> float:
    """Maximum drawdown as a positive fraction (0 to 1)."""
    pv      = np.array(portfolio_values, dtype=float)
    peak    = np.maximum.accumulate(pv)
    dd      = (pv - peak) / peak
    return float(-dd.min()) if len(dd) > 0 else 0.0


def _annualised_return(portfolio_values, bars_per_year: int) -> float:
    pv    = np.array(portfolio_values, dtype=float)
    n     = len(pv) - 1
    if n <= 0 or pv[0] == 0:
        return 0.0
    total = pv[-1] / pv[0]
    return float(total ** (bars_per_year / n) - 1)



def calculate_metrics(
    portfolio_values,
    trades: pd.DataFrame,
    rf: float = 0.0,
    bars_per_year: int = BARS_PER_YEAR_5MIN,
) -> dict:
    """
    Compute all required performance metrics.

    Parameters
    ----------
    portfolio_values : list or array of bar-by-bar portfolio values.
    trades           : DataFrame with at least a 'pnl' column.
    rf               : Risk-free rate (annualised, default 0).
    bars_per_year    : Number of bars in one year (default 5-min crypto).

    Returns
    -------
    dict with keys:
        sharpe_ratio, sortino_ratio, calmar_ratio,
        max_drawdown, win_rate,
        total_return, annualised_return,
        total_trades, avg_pnl
    """
    rets      = _bar_returns(portfolio_values)
    ann_ret   = _annualised_return(portfolio_values, bars_per_year)
    mdd       = _max_drawdown(portfolio_values)


    excess    = rets - rf / bars_per_year
    sharpe    = (
        float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(bars_per_year))
        if np.std(excess, ddof=1) > 0 else 0.0
    )


    downside  = excess[excess < 0]
    down_std  = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino   = (
        float(np.mean(excess) / down_std * np.sqrt(bars_per_year))
        if down_std > 0 else 0.0
    )

    # Calmar
    calmar    = float(ann_ret / mdd) if mdd > 0 else 0.0

    # Win rate
    if len(trades) > 0 and "pnl" in trades.columns:
        wins     = (trades["pnl"] > 0).sum()
        win_rate = float(wins / len(trades))
        avg_pnl  = float(trades["pnl"].mean())
        total_trades = len(trades)
    else:
        win_rate, avg_pnl, total_trades = 0.0, 0.0, 0

    # Total return
    pv = np.array(portfolio_values, dtype=float)
    total_return = float((pv[-1] - pv[0]) / pv[0]) if pv[0] != 0 else 0.0

    return {
        "sharpe_ratio":    round(sharpe, 4),
        "sortino_ratio":   round(sortino, 4),
        "calmar_ratio":    round(calmar, 4),
        "max_drawdown":    round(mdd, 4),
        "win_rate":        round(win_rate, 4),
        "total_return":    round(total_return, 4),
        "annualised_return": round(ann_ret, 4),
        "total_trades":    total_trades,
        "avg_pnl":         round(avg_pnl, 4),
    }


def print_metrics(metrics_dict: dict, label: str = "Performance") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")
    for k, v in metrics_dict.items():
        print(f"  {k:<22}: {v}")
    print(f"{'─' * 40}\n")
