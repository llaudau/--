"""Parameter sweep with risk-adjusted scoring (Sharpe, drawdown, Calmar ratio).

Sweeps forward_days × return_threshold × top_n, evaluates each combo
on all walk-forward windows, and ranks by a risk-adjusted composite score.

Usage:
    python -m test.param_sweep
"""
import logging
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_OHLCV_DIR, RAW_BASICS_DIR, POOL_DIR, RESULTS_DIR
from model.features import load_and_build_features, get_feature_columns
from model.RF.rf_model import RFModel
from stock_pool.pool_builder import load_pool
from test.backtest import backtest_return
from test.config import WALK_FORWARD_WINDOWS, MAX_TRAIN_SAMPLES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Sweep grid ────────────────────────────────────────────────────────────────
FORWARD_DAYS_GRID   = [3, 5]
THRESHOLD_GRID      = [0.015, 0.02, 0.025, 0.03]
TOP_N_GRID          = [5, 10, 20]

RF_PARAMS = dict(
    n_estimators=300, max_features="sqrt", max_depth=8,
    min_samples_leaf=100, n_jobs=-1, random_state=42, class_weight="balanced",
)


def train_and_predict_window(model, df, feature_cols, window, forward_days):
    """Train on one window, return test predictions."""
    train_start, train_end, test_start, test_end = window

    train_dates = sorted(
        df.loc[(df["date"] >= train_start) & (df["date"] <= train_end), "date"].unique()
    )
    if len(train_dates) > forward_days:
        safe_end = train_dates[-(forward_days + 1)]
    else:
        safe_end = train_dates[0]

    train_mask = (df["date"] >= train_start) & (df["date"] <= safe_end)
    test_mask  = (df["date"] >= test_start)  & (df["date"] <= test_end)

    train_df = df.loc[train_mask]
    if len(train_df) > MAX_TRAIN_SAMPLES:
        train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    test_df = df.loc[test_mask]
    X_test  = test_df[feature_cols]

    if len(X_test) == 0:
        return pd.DataFrame()

    model.train(X_train, y_train)
    y_prob = model.predict_proba(X_test)

    preds = test_df[["date", "code", "fwd_ret", "label"]].copy()
    preds["y_prob"] = y_prob.values
    return preds


def score_combo(bt_result: dict) -> float:
    """Risk-adjusted composite score.

    score = Sharpe * 0.5 + Calmar * 0.3 + log(cumulative) * 0.2

    This balances:
      - Sharpe: risk-adjusted return consistency
      - Calmar: return / max drawdown (penalises big dips)
      - Log cumulative: raw profitability
    """
    sharpe = bt_result["sharpe"]
    max_dd = bt_result["max_drawdown"]
    cum_ret = float(bt_result["rf_cum"].iloc[-1])

    calmar = 0.0
    if abs(max_dd) > 0.001:
        annual_ret = cum_ret ** (1/3) - 1  # approx 3 years
        calmar = annual_ret / abs(max_dd)

    log_cum = np.log(max(cum_ret, 0.01))
    return sharpe * 0.5 + calmar * 0.3 + log_cum * 0.2


def main():
    logger.info("=== Risk-Adjusted Parameter Sweep ===\n")

    pool_df = load_pool(POOL_DIR)
    pool_codes = set(pool_df["code"].tolist())
    basics_path = sorted(RAW_BASICS_DIR.glob("basics_*.parquet"))[-1]

    results_rows = []

    for fwd, thresh in product(FORWARD_DAYS_GRID, THRESHOLD_GRID):
        logger.info(f"\n--- forward_days={fwd}, threshold={thresh:.1%} ---")

        df, feature_cols, _ = load_and_build_features(
            ohlcv_dir=CLEANED_OHLCV_DIR, basics_path=basics_path,
            pool_codes=pool_codes, forward_days=fwd, return_threshold=thresh,
        )

        config = {"rf_params": RF_PARAMS, "forward_days": fwd, "return_threshold": thresh}
        model = RFModel(config)

        all_preds = []
        for w in WALK_FORWARD_WINDOWS:
            preds = train_and_predict_window(model, df, feature_cols, w, fwd)
            if not preds.empty:
                all_preds.append(preds)

        if not all_preds:
            continue
        combined_preds = pd.concat(all_preds, ignore_index=True)

        for top_n in TOP_N_GRID:
            bt = backtest_return(combined_preds, top_n=top_n)
            score = score_combo(bt)

            row = {
                "forward_days": fwd,
                "return_threshold": thresh,
                "top_n": top_n,
                "cumulative_return": float(bt["rf_cum"].iloc[-1]),
                "sharpe": bt["sharpe"],
                "max_drawdown": bt["max_drawdown"],
                "score": score,
            }
            results_rows.append(row)
            logger.info(
                f"  top_n={top_n}: cum={row['cumulative_return']:.1f}x, "
                f"sharpe={row['sharpe']:.2f}, dd={row['max_drawdown']:.1%}, "
                f"SCORE={score:.2f}"
            )

    results_df = pd.DataFrame(results_rows).sort_values("score", ascending=False)
    out_path = RESULTS_DIR / "param_sweep_risk_adjusted.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    logger.info(f"\n{'='*70}")
    logger.info("TOP 10 PARAMETER COMBOS (risk-adjusted score):")
    logger.info(f"{'='*70}")
    for i, row in results_df.head(10).iterrows():
        logger.info(
            f"  fwd={int(row['forward_days'])}d, thresh={row['return_threshold']:.1%}, "
            f"top_n={int(row['top_n'])}: "
            f"cum={row['cumulative_return']:.1f}x, sharpe={row['sharpe']:.2f}, "
            f"dd={row['max_drawdown']:.1%}, SCORE={row['score']:.2f}"
        )

    best = results_df.iloc[0]
    logger.info(f"\nBEST: forward_days={int(best['forward_days'])}, "
                f"threshold={best['return_threshold']:.1%}, top_n={int(best['top_n'])}")
    logger.info(f"Results saved to {out_path}")
    return results_df


if __name__ == "__main__":
    main()
