import pandas as pd
import metrics
import signals



def run_backtest(data: pd.DataFrame, params: dict, initial_cash: float, COM: float) -> dict:
    """
    Run a complete backtest over the provided DataFrame.

    Bar-by-bar flow:
      1. If a position is open → check SL / TP and close if triggered.
      2. If no position is open → open long or short based on the 2/3 signal.
      3. Record the portfolio value at the current bar's close price.

    NOTE: The index is reset to integers (positional). Timestamps are NOT used
    for any trading logic; they are only present in the raw data for reference.

    Parameters
    ----------
    data         : OHLCV DataFrame (DatetimeIndex or integer index).
    params       : Strategy and risk-management parameters. Expected keys:
                     sl     – stop-loss fraction  (e.g. 0.02 = 2%)
                     tp     – take-profit fraction (e.g. 0.04 = 4%)
                     n_btc  – position size in BTC
                     + all indicator params required by signals.py
    initial_cash : Starting cash in USD.
    COM          : One-way commission rate (e.g. 0.00125 = 0.125%).

    Returns
    -------
    dict:
        portfolio : list[float]  – bar-by-bar portfolio value
        trades    : pd.DataFrame – executed trade log with net PnL
        metrics   : dict         – performance metrics
    """
    data = data.copy()


    data = signals.add_indicators(data, params)
    data = signals.generate_signals(data, params)
    data = data.dropna()               # only keep bars where all indicators valid
    data = data.reset_index(drop=True)

    sl    = params["sl"]
    tp    = params["tp"]
    n_btc = params["n_btc"]

    lost_to_com  = 0.0
    cash         = initial_cash
    position     = None               # None | dict describing the open position
    portfolio_values: list[float] = []
    trades: list[dict]            = []

    for idx, row in data.iterrows():
        price = row["Close"]

        if position is not None:

            if position["type"] == "long":
                pnl_pct = (price - position["entry"]) / position["entry"]

                if pnl_pct <= -sl or pnl_pct >= tp:
                    proceeds     = price * position["n_btc"] * (1 - COM)
                    lost_to_com += price * position["n_btc"] * COM
                    # Net PnL = exit proceeds minus what was paid at entry
                    net_pnl  = proceeds - position["cost"]
                    cash    += proceeds
                    trades.append({
                        "bar":        idx,
                        "type":       "long",
                        "entry":      position["entry"],
                        "exit":       price,
                        "pnl":        net_pnl,
                        "reason":     "sl" if pnl_pct <= -sl else "tp",
                    })
                    position = None

            elif position["type"] == "short":
                pnl_pct = (position["entry"] - price) / position["entry"]

                if pnl_pct <= -sl or pnl_pct >= tp:
                    cost_cover   = price * position["n_btc"] * COM
                    lost_to_com += cost_cover
                    # Net PnL for short: price difference minus both commissions
                    net_pnl  = (position["entry"] - price) * position["n_btc"] \
                               - cost_cover - position["coms"]
                    cash    += (position["entry"] - price) * position["n_btc"] \
                               - cost_cover
                    trades.append({
                        "bar":    idx,
                        "type":   "short",
                        "entry":  position["entry"],
                        "exit":   price,
                        "pnl":    net_pnl,
                        "reason": "sl" if pnl_pct <= -sl else "tp",
                    })
                    position = None

        sig = row["signal"]

        if position is None:

            if sig == 1:                                    # LONG
                cost = price * n_btc * (1 + COM)
                if cash >= cost:
                    lost_to_com += price * n_btc * COM
                    cash        -= cost
                    position     = {
                        "type":  "long",
                        "entry": price,
                        "n_btc": n_btc,
                        "cost":  cost,
                    }

            elif sig == -1:                                 # SHORT
                collateral = price * n_btc * (1 + COM)
                if cash >= collateral:
                    contract_cost = price * n_btc * COM
                    lost_to_com  += contract_cost
                    cash         -= contract_cost
                    position      = {
                        "type":       "short",
                        "entry":      price,
                        "n_btc":      n_btc,
                        "coms":       contract_cost,
                        "collateral": collateral,
                    }

        if position is None:
            pv = cash
        elif position["type"] == "long":
            pv = cash + price * position["n_btc"]
        else:  # short: unrealised P&L on open position
            pv = cash + (position["entry"] - price) * position["n_btc"]

        portfolio_values.append(pv)

    if position is not None:
        last_price = data["Close"].iloc[-1]

        if position["type"] == "long":
            proceeds     = last_price * position["n_btc"] * (1 - COM)
            lost_to_com += last_price * position["n_btc"] * COM
            net_pnl      = proceeds - position["cost"]
            cash        += proceeds
        else:
            cost_cover   = last_price * position["n_btc"] * COM
            lost_to_com += cost_cover
            net_pnl      = (position["entry"] - last_price) * position["n_btc"] \
                           - cost_cover - position["coms"]
            cash        += (position["entry"] - last_price) * position["n_btc"] \
                           - cost_cover

        trades.append({
            "bar":    len(data) - 1,
            "type":   position["type"],
            "entry":  position["entry"],
            "exit":   last_price,
            "pnl":    net_pnl,
            "reason": "end_of_period",
        })

    # Results
    tr_df = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=["bar", "type", "entry", "exit", "pnl", "reason"])
    )

    perf                = metrics.calculate_metrics(portfolio_values, tr_df, rf=0.0)
    perf["lost_to_com"] = round(lost_to_com, 4)

    return {
        "portfolio": portfolio_values,
        "trades":    tr_df,
        "metrics":   perf,
    }
