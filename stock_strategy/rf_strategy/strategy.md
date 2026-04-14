# RF Stock Classification Strategy

## Goal

Predict which A-share mainboard stocks will rise >2% over the next 5 trading days using a Random Forest classifier. Use the predictions to select a weekly portfolio.

## Data

- **Universe:** SZ mainboard (000/001/002/003) + SH mainboard (600/601/603/605)
- **Source:** baostock daily OHLCV + PE/PB/PS/PCF, akshare stock list
- **Time range:** 2022-01-01 onwards (pre-2022 excluded — different macro environment)
- **Exclusions:** ST stocks

## Label Definition

For each stock on each trading day:

```
label = 1  if  close[t+5] / close[t] - 1 > 2%
label = 0  otherwise
```

This is a binary classification: "can earn" vs "can't earn" over a 5-day horizon.

## Features

All features are computed per stock per day using rolling windows.

| Group | Feature | Description |
|---|---|---|
| Momentum | `ret_5d` | 5-day past return |
| | `ret_10d` | 10-day past return |
| | `ret_20d` | 20-day past return |
| | `ret_60d` | 60-day past return |
| Volatility | `vol_20d` | 20-day rolling std of daily returns |
| | `vol_60d` | 60-day rolling std of daily returns |
| Volume | `volume_ratio_5d` | today's volume / 5-day avg volume |
| | `volume_ratio_20d` | today's volume / 20-day avg volume |
| | `turnover_rate` | daily turnover rate (from data) |
| Price pattern | `ma5_ratio` | close / 5-day MA |
| | `ma20_ratio` | close / 20-day MA |
| | `ma60_ratio` | close / 60-day MA |
| | `high_low_range` | (high - low) / close |
| | `upper_shadow` | (high - max(open, close)) / close |
| | `lower_shadow` | (min(open, close) - low) / close |
| Fundamental | `pe_ttm` | trailing twelve month P/E |
| | `pb_mrq` | most recent quarter P/B |
| | `ps_ttm` | trailing twelve month P/S |
| | `pcf_ttm` | trailing twelve month P/CF |
| Industry | `industry_code` | integer-encoded industry category |

## Model

**Algorithm:** Random Forest Classifier

```
n_estimators    = 300
max_features    = "sqrt"
max_depth       = 12
min_samples_leaf = 50
class_weight    = "balanced"
```

`class_weight="balanced"` upweights the minority class (label=1) during training, since most stocks don't rise >2% in any given 5-day window.

Training capped at 1,000,000 samples per window to avoid OOM.

## Validation: Walk-Forward

No random split — strictly time-ordered to avoid lookahead bias. The training window expands over time:

| Window | Train Period | Test Period |
|---|---|---|
| 1 | 2022-01 ~ 2023-06 | 2023-07 ~ 2023-12 |
| 2 | 2022-01 ~ 2023-12 | 2024-01 ~ 2024-12 |
| 3 | 2022-01 ~ 2024-12 | 2025-01 ~ 2025-12 |

Metrics per window: accuracy, precision, recall, F1, AUC-ROC.

## Buying Strategy

Weekly rebalance using model predictions:

1. **Score:** At the start of each week, run the trained RF model on all eligible stocks. The model outputs `P(label=1)` — the predicted probability that the stock will return >2% over the next 5 days.
2. **Select:** Rank all stocks by `P(label=1)` descending. Pick the **top 20** stocks.
3. **Buy:** Equal-weight allocation across the 20 selected stocks.

## Selling Strategy

- **Hold period:** 5 trading days (aligned with the prediction horizon).
- **Weekly rotation:** At the end of each week, sell all positions and re-run the model to build the new portfolio.
- No stop-loss or trailing stop — the position is held for the full 5-day period.

## Benchmark Comparison

The backtest compares three strategies over the same test periods:

| Strategy | Description |
|---|---|
| **RF Top-20** | Buy 20 stocks with highest predicted probability each week |
| **Random-20** | Buy 20 randomly selected stocks each week |
| **Market Average** | Average return of all stocks in the universe each week |

Cumulative returns are plotted for all three.

## Evaluation Outputs

All saved to `results/`:

- `walk_forward_metrics.png` — metrics across the 3 walk-forward windows
- `feature_importance.png` — top 20 features by permutation importance (last window)
- `backtest_cumulative.png` — cumulative return: RF top-20 vs random-20 vs market avg

## Limitations

- **No transaction costs.** Real slippage and commissions would reduce returns.
- **No position sizing.** Equal weight across all 20 picks.
- **No intraday signal.** Assumes you can buy at the close price on the signal day.
- **Forward-adjusted prices.** Uses baostock `adjustflag=2` (forward-adjusted), which may introduce minor lookahead in price levels (though returns are unaffected).
- **5-day horizon only.** No exploration of other holding periods.
