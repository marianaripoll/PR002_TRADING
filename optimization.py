import time
import warnings
from datetime import timedelta

import optuna
import pandas as pd

import backtest
from signals import DEFAULT_PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")



INITIAL_CASH  = 10_000.0
COM           = 0.00125          # 0.125 % one-way

TRAIN_DAYS    = 28               # ~1 month
TEST_DAYS     = 7                # 1 week
STEP_DAYS     = 7                # step forward weekly

N_TRIALS      = 150              # Optuna trials per window




def _suggest_params(trial: optuna.Trial) -> dict:
    """
    Define the hyper-parameter search space for Optuna.
    All indicator and risk-management params are tuned here.
    """
    return {
        # RSI
        "rsi_period": trial.suggest_int("rsi_period", 7, 28),
        "rsi_ob":     trial.suggest_float("rsi_ob", 60.0, 80.0),
        "rsi_os":     trial.suggest_float("rsi_os", 20.0, 40.0),
        # MACD
        "macd_fast":   trial.suggest_int("macd_fast", 6, 20),
        "macd_slow":   trial.suggest_int("macd_slow", 21, 52),
        "macd_signal": trial.suggest_int("macd_signal", 5, 15),
        # Bollinger Bands
        "bb_period": trial.suggest_int("bb_period", 10, 40),
        "bb_std":    trial.suggest_float("bb_std", 1.5, 3.0),
        # Risk management
        "sl":    trial.suggest_float("sl", 0.005, 0.05),
        "tp":    trial.suggest_float("tp", 0.01,  0.10),
        "n_btc": trial.suggest_float("n_btc", 0.005, 0.05),
    }



def optimise_window(train_data: pd.DataFrame, n_trials: int = N_TRIALS) -> dict:
    """
    Run Optuna optimisation on a single training window.

    Parameters
    ----------
    train_data : OHLCV DataFrame for the training period.
    n_trials   : Number of Optuna trials.

    Returns
    -------
    dict : Best parameters found.
    """
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)

        # Ensure MACD slow > fast (hard constraint)
        if params["macd_slow"] <= params["macd_fast"]:
            return -999.0

        result = backtest.run_backtest(train_data, params, INITIAL_CASH, COM)
        calmar = result["metrics"]["calmar_ratio"]

        # Penalise strategies with no trades
        if result["metrics"]["total_trades"] == 0:
            return -999.0

        return calmar if not (calmar != calmar) else -999.0  # guard NaN

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params



def walk_forward(data: pd.DataFrame, n_trials: int = N_TRIALS) -> dict:
    """
    Run the full walk-forward analysis.

    Parameters
    ----------
    data     : Full training DataFrame (DatetimeIndex).
    n_trials : Optuna trials per window.

    Returns
    -------
    dict:
        windows      : list of per-window result dicts
        oos_portfolio: combined out-of-sample portfolio values
        oos_metrics  : metrics computed on the combined OOS series
        best_params  : params from the last window (used for final test)
        elapsed_time : total optimisation time in seconds
    """
    import metrics as m

    start_time  = time.time()
    windows     = []
    oos_pv      = []           # out-of-sample portfolio values (concatenated)
    oos_trades  = []

    # Build window start dates
    dates      = data.index
    start_date = dates[0]
    end_date   = dates[-1]

    window_id  = 0
    best_params = DEFAULT_PARAMS.copy()

    current = start_date

    while True:
        train_end = current + timedelta(days=TRAIN_DAYS)
        test_end  = train_end + timedelta(days=TEST_DAYS)

        if test_end > end_date:
            break

        train_slice = data[(data.index >= current) & (data.index < train_end)]
        test_slice  = data[(data.index >= train_end) & (data.index < test_end)]

        if len(train_slice) < 100 or len(test_slice) < 20:
            current += timedelta(days=STEP_DAYS)
            continue

        #Optimise on training window
        print(f"  Window {window_id:>3} | train: {current.date()} → {train_end.date()} "
              f"| test: {train_end.date()} → {test_end.date()}", end=" ... ")

        opt_params  = optimise_window(train_slice, n_trials)
        best_params = opt_params

        # Evaluate on test window
        oos_result  = backtest.run_backtest(test_slice, opt_params, INITIAL_CASH, COM)
        calmar_oos  = oos_result["metrics"]["calmar_ratio"]

        print(f"Calmar OOS = {calmar_oos:.4f}")

        oos_pv.extend(oos_result["portfolio"])

        if len(oos_result["trades"]) > 0:
            oos_trades.append(oos_result["trades"])

        windows.append({
            "window_id":   window_id,
            "train_start": current,
            "train_end":   train_end,
            "test_start":  train_end,
            "test_end":    test_end,
            "best_params": opt_params,
            "oos_metrics": oos_result["metrics"],
        })

        window_id += 1
        current   += timedelta(days=STEP_DAYS)


    all_trades = (
        pd.concat(oos_trades, ignore_index=True)
        if oos_trades
        else pd.DataFrame(columns=["bar", "type", "entry", "exit", "pnl"])
    )

    oos_metrics = m.calculate_metrics(oos_pv, all_trades, rf=0.0)
    elapsed     = round(time.time() - start_time, 1)

    print(f"\n[optimization] Walk-forward complete | {window_id} windows | "
          f"{elapsed}s elapsed")

    return {
        "windows":       windows,
        "oos_portfolio": oos_pv,
        "oos_metrics":   oos_metrics,
        "best_params":   best_params,
        "elapsed_time":  elapsed,
    }



def sensitivity_analysis(
    data: pd.DataFrame,
    best_params: dict,
    variation: float = 0.20,
) -> pd.DataFrame:
    """
    Vary each numeric parameter by ±variation (default ±20 %) and record
    the impact on Calmar Ratio.

    Parameters
    ----------
    data        : DataFrame to run the backtests on.
    best_params : Optimal parameters to perturb.
    variation   : Fractional variation (0.20 = ±20 %).

    Returns
    -------
    pd.DataFrame with columns:
        parameter, direction, value, calmar_ratio, delta_calmar
    """
    import backtest as bt

    baseline = bt.run_backtest(data, best_params, INITIAL_CASH, COM)
    base_calmar = baseline["metrics"]["calmar_ratio"]

    rows = []
    numeric_params = {k: v for k, v in best_params.items() if isinstance(v, (int, float))}

    for param, base_val in numeric_params.items():
        for direction, mult in [("up", 1 + variation), ("down", 1 - variation)]:
            perturbed = best_params.copy()

            # Preserve int type for integer params
            new_val = base_val * mult
            if isinstance(base_val, int):
                new_val = max(1, int(round(new_val)))
            perturbed[param] = new_val

            try:
                result = bt.run_backtest(data, perturbed, INITIAL_CASH, COM)
                calmar = result["metrics"]["calmar_ratio"]
            except Exception:
                calmar = float("nan")

            rows.append({
                "parameter":    param,
                "direction":    direction,
                "base_value":   base_val,
                "new_value":    new_val,
                "calmar_ratio": calmar,
                "delta_calmar": calmar - base_calmar,
            })

    return pd.DataFrame(rows).sort_values("delta_calmar")
