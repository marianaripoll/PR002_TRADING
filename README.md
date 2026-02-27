# PR002_TRADING
## Project Structure

```
├── data.py            # Data loading and preprocessing
├── signals.py         # Technical indicators and signal generation
├── backtest.py        # Backtesting engine
├── metrics.py         # Performance metrics (Sharpe, Sortino, Calmar, etc.)
├── optimization.py    # Walk-forward optimization with Optuna
├── main.py            # Entry point – runs the full pipeline
├── requirements.txt
└── README.md
```

## Strategy Overview

The strategy uses **3 technical indicators** with a **2-out-of-3 signal confirmation** rule:

| Indicator | Role |
|-----------|------|
| RSI       | Momentum / overbought-oversold |
| MACD      | Trend direction confirmation  |
| Bollinger Bands | Volatility breakout filter |

Positions: **Long & Short**, no leverage.  
Transaction fee: **0.125%** per side.

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset files in the project root:
- `btc_project_train.csv`
- `btc_project_test.csv`

## Usage

```bash
# Run full pipeline: walk-forward optimization + final test evaluation
python main.py
```

Results and plots are saved in the `results/` folder.

## Walk-Forward Configuration

| Parameter       | Value   |
|----------------|---------|
| Training window | 1 month |
| Test window     | 1 week  |
| Step forward    | 1 week  |
| Trials per window | 100–200 |

## Performance Metrics

- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio *(objective function)*
- Maximum Drawdown
- Win Rate
