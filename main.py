import os
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import data as dt
import backtest
import optimization as opt
import metrics as m


INITIAL_CASH  = 10_000.0
COM           = 0.00125            # 0.125 %
RESULTS_DIR   = "results"




def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_json(obj, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  [saved] {path}")


def _plot_portfolio(portfolio_values, title, filename):
    """Line chart of portfolio value over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(portfolio_values, linewidth=1.0, color="#2196F3")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def _build_periodic_returns(portfolio_values, index=None) -> dict:
    """
    Compute monthly, quarterly, and annual return tables.

    If index (DatetimeIndex) is provided, periods are calendar-based.
    Otherwise falls back to equal-length chunks.
    """
    pv = pd.Series(portfolio_values)

    if index is not None and len(index) == len(pv):
        pv.index = index[:len(pv)]

        monthly   = pv.resample("ME").last().pct_change().dropna().rename("return")
        quarterly = pv.resample("QE").last().pct_change().dropna().rename("return")
        annual    = pv.resample("YE").last().pct_change().dropna().rename("return")
    else:
        # Fallback: split into ~monthly, ~quarterly, ~annual chunks
        monthly   = pv.iloc[::max(1, len(pv) // 12)].pct_change().dropna().rename("return")
        quarterly = pv.iloc[::max(1, len(pv) // 4)].pct_change().dropna().rename("return")
        annual    = pv.iloc[::max(1, len(pv) // 1)].pct_change().dropna().rename("return")

    return {
        "monthly":   monthly.round(4).to_dict(),
        "quarterly": quarterly.round(4).to_dict(),
        "annual":    annual.round(4).to_dict(),
    }


def _plot_sensitivity(sensitivity_df: pd.DataFrame, filename: str):
    """Horizontal bar chart showing ΔCalmar per parameter perturbation."""
    df = sensitivity_df.copy()
    df["label"] = df["parameter"] + " (" + df["direction"] + ")"
    df = df.sort_values("delta_calmar")

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df["delta_calmar"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    ax.barh(df["label"], df["delta_calmar"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Sensitivity Analysis – ΔCalmar Ratio (±20 % parameter variation)", fontsize=12)
    ax.set_xlabel("ΔCalmar Ratio")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")




def main():
    _ensure_results_dir()

    print("\n" + "═" * 60)
    print("  BTC Systematic Trading Strategy – Full Pipeline")
    print("═" * 60)


    print("\n[1] Loading data...")
    train_df, test_df = dt.load_train_test()

    #Walk-forward optimisation on TRAIN
    print("\n[2] Walk-forward optimisation (training set)...")
    wf_results = opt.walk_forward(train_df, n_trials=opt.N_TRIALS)

    best_params  = wf_results["best_params"]
    oos_metrics  = wf_results["oos_metrics"]
    elapsed      = wf_results["elapsed_time"]

    print(f"\n  Total optimisation time : {elapsed}s")
    m.print_metrics(oos_metrics, label="Walk-Forward OOS Metrics (Train Set)")

    _plot_portfolio(
        wf_results["oos_portfolio"],
        title="Walk-Forward Out-of-Sample Portfolio (Train Set)",
        filename="wf_oos_portfolio.png",
    )
    _save_json({"best_params": best_params, "oos_metrics": oos_metrics,
                "elapsed_time": elapsed}, "wf_results.json")

    # Periodic returns for WF OOS
    periodic = _build_periodic_returns(wf_results["oos_portfolio"])
    _save_json(periodic, "wf_periodic_returns.json")

    # Final evaluation on TEST set
    print("\n[3] Final evaluation on TEST set...")
    test_result  = backtest.run_backtest(test_df, best_params, INITIAL_CASH, COM)
    test_metrics = test_result["metrics"]

    m.print_metrics(test_metrics, label="Final Test Set Performance")

    _plot_portfolio(
        test_result["portfolio"],
        title="Final Test Set Portfolio Value",
        filename="test_portfolio.png",
    )
    _save_json({"best_params": best_params, "test_metrics": test_metrics},
               "test_results.json")

    # Save trade log
    if len(test_result["trades"]) > 0:
        test_result["trades"].to_csv(
            os.path.join(RESULTS_DIR, "test_trades.csv"), index=False
        )
        print(f"  [saved] {RESULTS_DIR}/test_trades.csv")

    # Periodic returns for test
    periodic_test = _build_periodic_returns(
        test_result["portfolio"],
        index=test_df.index,
    )
    _save_json(periodic_test, "test_periodic_returns.json")

    # Sensitivity analysis on TEST set
    print("\n[4] Sensitivity analysis (±20 %)...")
    sens_df = opt.sensitivity_analysis(test_df, best_params, variation=0.20)
    sens_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity.csv"), index=False)
    print(f"  [saved] {RESULTS_DIR}/sensitivity.csv")
    _plot_sensitivity(sens_df, "sensitivity_chart.png")

    # Summary
    print("\n" + "═" * 60)
    print("  Pipeline complete.  All outputs saved to:", RESULTS_DIR)
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
