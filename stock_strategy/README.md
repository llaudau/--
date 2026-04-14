# stock_strategy

A modular, portable A-share mainboard stock prediction system using Machine Learning.

## Architecture

```
stock_strategy/
├── config.py               # Shared project-level paths
├── requirements.txt        # Single shared venv
│
├── data_downloader/        # 1. Raw data acquisition
├── data_washer/            # 2. Data cleaning & validation
├── stock_pool/             # 3. Stock universe filtering
├── model/                  # 4. ML models (RF and extensible to others)
│   ├── base.py             #    Abstract BaseModel interface
│   ├── features.py         #    Shared feature engineering
│   └── RF/                 #    Random Forest implementation
│
├── test/                   # 5. Walk-forward validation, backtest, reporting
│
└── data/                   # All data (gitignored)
    ├── raw/                #   from data_downloader
    ├── cleaned/            #   from data_washer
    ├── pool/               #   from stock_pool
    ├── models/             #   trained model artifacts
    └── results/            #   plots and reports
```

## Full Pipeline

```bash
# 1. Download latest A-share data (incremental — only fetches new days)
python -m data_downloader.download_all

# 2. Clean and validate raw data
python -m data_washer.run

# 3. Build stock universe (filter ST, illiquid, new IPOs, etc.)
python -m stock_pool.run

# 4+5. Train RF model, walk-forward backtest, generate report
python -m test.run --model RF

# Get today's top-5 stock picks
python -m model.RF.predict_today
```

---

## Module 1: Data Downloader

Downloads daily OHLCV + fundamentals for ~3,194 Chinese A-share mainboard stocks.

**Coverage:** SZ mainboard (000xxx, 001xxx, 002xxx, 003xxx) + SH mainboard (600xxx, 601xxx, 603xxx, 605xxx).
Data sourced from [baostock](http://baostock.com) (OHLCV + PE/PB/PS/PCF) and [akshare](https://akshare.akfamily.xyz).

**Incremental design:** `data/raw/meta/download_log.json` stores the actual last downloaded trading date per stock. Each run only fetches new rows.

```bash
python -m data_downloader.download_all           # OHLCV update (daily use)
python -m data_downloader.download_all --basics  # update stock basics
python -m data_downloader.download_all --all     # both
```

---

## Module 2: Data Washer

Cleans raw downloaded data before it reaches any model.

| Problem | Action |
|---|---|
| OHLC inconsistency (`high < close`) | Auto-correct: `high = max(open, high, close)` |
| Zero-volume rows (suspended days) | Flag with `is_suspended=True` column (kept, not removed) |
| NaN fundamentals (PE/PB/turnover) | Forward-fill then backward-fill per stock |
| Duplicate dates | Drop, keep last |
| Wrong dtype (`is_st` as string) | Convert to `int8` |
| Price spikes >22% single day | Flag with `price_spike=True` (not removed) |

**Output:** `data/cleaned/ohlcv/` + `data/cleaned/reports/wash_report_YYYYMMDD.json`

---

## Module 3: Stock Pool

Filters unreliable stocks out of the trading universe before modeling.

| Filter | Threshold | Reason |
|---|---|---|
| ST / \*ST / PT name | always | Special treatment / near delisting |
| `is_st` flag | == 1 | Current ST flag from exchange |
| History too short | < 120 trading days | Insufficient data for features |
| Illiquid | avg daily amount < 5M CNY | Hard to execute trades |
| New IPO | < 60 trading days after listing | Abnormal early price behavior |
| Excessive suspension | > 10% days suspended | Unreliable price data |

**Result:** ~2,938 / 3,194 stocks pass (132 ST, 20 short history, 104 over-suspended excluded).

---

## Module 4: Model

### Abstract Interface (`model/base.py`)

Every model under `model/` implements `BaseModel`:

```python
class BaseModel(ABC):
    def train(self, X: DataFrame, y: Series) -> None: ...
    def predict_proba(self, X: DataFrame) -> Series: ...   # ranking score
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def get_feature_importance(self, X, y, feature_names) -> DataFrame: ...
```

**Adding a new model** (e.g. XGBoost):
1. Create `model/XGB/xgb_model.py` implementing `BaseModel`
2. Add one line to `MODEL_REGISTRY` in `test/run.py`
3. Run `python -m test.run --model XGB`

### Shared Feature Engineering (`model/features.py`)

| Group | Features |
|---|---|
| Momentum | `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` |
| Volatility | `vol_20d`, `vol_60d` |
| Volume | `volume_ratio_5d`, `volume_ratio_20d`, `turnover_rate` |
| Price patterns | `ma5_ratio`, `ma20_ratio`, `ma60_ratio`, `high_low_range`, `upper_shadow`, `lower_shadow` |
| Fundamentals | `pe_ttm`, `pb_mrq`, `ps_ttm`, `pcf_ttm` |
| Industry | integer-encoded industry category |

### RF Configuration (`model/RF/config.py`)

Best parameters from systematic sweep:

```python
FORWARD_DAYS = 3       # predict 3-day forward return
RETURN_THRESHOLD = 0.03  # >3% = buy signal
TOP_N = 5              # select top 5 stocks

RF_PARAMS = dict(
    n_estimators=300, max_depth=8, min_samples_leaf=100,
    max_features="sqrt", class_weight="balanced",
)
```

---

## Module 5: Test

### Walk-Forward Validation

Expanding training window — no lookahead bias:

| Window | Train | Test |
|---|---|---|
| 1 | 2022–2023 H1 | 2023 H2 |
| 2 | 2022–2023 | 2024 |
| 3 | 2022–2024 | 2025 |

### Backtest Strategy

- **Rebalance:** Weekly (ISO week)
- **Selection:** Top-N stocks by RF predicted probability
- **Position sizing:** Equal weight (1/N per stock)
- **Holding period:** `FORWARD_DAYS` trading days
- **No stop-loss:** Fixed time exit only

### Backtest Results (RF top-5, all windows combined)

| Metric | Value |
|---|---|
| Cumulative Return | **183.4x** |
| Sharpe Ratio | **3.69** |
| Max Drawdown | **-21.6%** |
| Mean AUC | **0.638** |
| Mean Precision | **0.246** |

### Outputs (`data/results/RF/`)

| File | Contents |
|---|---|
| `walk_forward_metrics.png` | Accuracy, Precision, Recall, F1, AUC per window |
| `feature_importance.png` | Top 20 features by permutation importance |
| `backtest_cumulative.png` | RF vs Random vs Market cumulative return |
| `overfit_diagnostic.png` | AUC per window (overfitting check) |
| `return_stability.png` | Weekly return distribution box plot |
| `report.txt` | Full text summary with all metrics |

---

## Trading Strategy

### Buying Rules (Entry)
- **Signal:** RF model ranks all pool stocks by probability of >3% return in 3 days
- **Selection:** Top 5 highest-confidence stocks each week
- **Sizing:** Equal weight — 20% of capital per stock
- **Timing:** Buy at market close on signal date (or next open)

### Selling Rules (Exit)
- **Timing:** Sell at market close after 3 trading days
- **Rule:** Fixed holding period only — no stop-loss, no dynamic exit

### Risk Management
- Long-only, no leverage
- Stock pool pre-filters bad stocks (ST, illiquid, new IPO)
- RF uses `class_weight="balanced"` to handle imbalanced labels

---

## Notes

- **Backtest is simulation only.** Does not account for transaction costs, slippage, or market impact.
- **Top-5 is concentrated.** Higher potential returns but also higher volatility vs top-20.
- **183.4x backtest return** spans 3 years of test data. Actual live performance will differ.
- The `rf_strategy/` folder is the original monolithic version kept for reference.
