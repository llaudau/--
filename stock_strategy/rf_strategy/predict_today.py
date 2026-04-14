"""
predict_today.py — Generate today's top stock picks using the best parameters.

Best parameters from param_sweep:
  forward_days=3, return_threshold=3%, max_depth=8, min_samples_leaf=100, top_n=5

Strategy:
  - Train RF on all available data up to today
  - Predict probability each stock gains >3% in next 3 trading days
  - Output top 5 highest-confidence picks

Buy timing : at market close today (or open tomorrow)
Sell timing: at market close 3 trading days later
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# ── Best parameters ──────────────────────────────────────────────────────────
FORWARD_DAYS      = 3      # predict 3-day forward return
RETURN_THRESHOLD  = 0.03   # label=1 if 3-day return > 3%
TOP_N             = 5      # output top 5 picks
RF_PARAMS = dict(
    n_estimators   = 300,
    max_features   = "sqrt",
    max_depth      = 8,       # best from param sweep
    min_samples_leaf = 100,   # best from param sweep
    n_jobs         = -1,
    random_state   = 42,
    class_weight   = "balanced",
)

# ── Paths (same as config.py) ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data_downloader" / "data"
OHLCV_DIR  = DATA_DIR / "ohlcv"
_basics_files = sorted((DATA_DIR / "basics").glob("basics_*.parquet"))
BASICS_PATH = _basics_files[-1] if _basics_files else None
LOOKBACKS  = [5, 10, 20, 60]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Feature engineering (mirrors features.py, FORWARD_DAYS injected) ─────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df = df[df["date"] >= "2022-01-01"]
    if len(df) < 120:
        return pd.DataFrame()

    close  = df["close"]
    ret_1d = close.pct_change()

    for lb in LOOKBACKS:
        df[f"ret_{lb}d"] = close.pct_change(lb)

    df["vol_20d"] = ret_1d.rolling(20).std()
    df["vol_60d"] = ret_1d.rolling(60).std()
    df["volume_ratio_5d"]  = df["volume"] / df["volume"].rolling(5).mean()
    df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()

    for w in [5, 20, 60]:
        df[f"ma{w}_ratio"] = close / close.rolling(w).mean()

    df["high_low_range"] = (df["high"] - df["low"]) / close
    df["upper_shadow"]   = (df["high"] - np.maximum(df["open"], close)) / close
    df["lower_shadow"]   = (np.minimum(df["open"], close) - df["low"]) / close

    # Forward label (used for training rows; not available for latest date)
    df["fwd_ret"] = close.shift(-FORWARD_DAYS) / close - 1
    df["label"]   = (df["fwd_ret"] > RETURN_THRESHOLD).astype("int8")

    keep = (["date", "fwd_ret", "label",
             "turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
            + [f"ret_{lb}d" for lb in LOOKBACKS]
            + ["vol_20d", "vol_60d", "volume_ratio_5d", "volume_ratio_20d",
               "ma5_ratio", "ma20_ratio", "ma60_ratio",
               "high_low_range", "upper_shadow", "lower_shadow"])
    return df[keep]


def load_data():
    """Load all stocks, compute features, merge industry codes."""
    if BASICS_PATH is None:
        raise FileNotFoundError("No basics parquet found in data/basics/")

    basics = pd.read_parquet(BASICS_PATH)
    industry_map = basics[["code", "industry"]].copy()
    industry_cats = industry_map["industry"].astype("category")
    industry_map["industry_code"] = industry_cats.cat.codes

    parquet_files = sorted(OHLCV_DIR.glob("*.parquet"))
    logger.info(f"Loading {len(parquet_files)} stock files...")

    all_dfs = []
    for f in tqdm(parquet_files, desc="Features"):
        code = f.stem
        df = pd.read_parquet(f, columns=["date", "open", "high", "low", "close",
                                          "volume", "turnover_rate", "pe_ttm",
                                          "pb_mrq", "ps_ttm", "pcf_ttm", "is_st"])
        if "is_st" in df.columns and df["is_st"].iloc[-1] == 1:
            continue

        feat = compute_features(df)
        if feat.empty:
            continue
        feat["code"] = code

        float_cols = feat.select_dtypes("float64").columns
        feat[float_cols] = feat[float_cols].astype("float32")
        all_dfs.append(feat)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.merge(industry_map[["code", "industry_code"]], on="code", how="left")
    combined["industry_code"] = combined["industry_code"].fillna(-1).astype("int16")
    return combined


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "code", "fwd_ret", "label"}
    return [c for c in df.columns if c not in exclude]


def main():
    logger.info("=== RF Stock Predictor (Best Params) ===")
    logger.info(f"  forward_days={FORWARD_DAYS}, return_threshold={RETURN_THRESHOLD:.0%}, top_n={TOP_N}")
    logger.info(f"  max_depth={RF_PARAMS['max_depth']}, min_samples_leaf={RF_PARAMS['min_samples_leaf']}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data()
    feature_cols = get_feature_cols(df)

    latest_date = df["date"].max()
    logger.info(f"Latest date in data: {latest_date}")

    # ── Separate training rows vs. today's prediction rows ───────────────────
    # Training rows: must have a valid label (fwd_ret known), so exclude rows
    # where the forward window extends beyond available data (last FORWARD_DAYS dates).
    all_dates = sorted(df["date"].unique())
    safe_train_end = all_dates[-(FORWARD_DAYS + 1)]  # leave last 3 days unlabelled

    train_mask = (df["date"] <= safe_train_end) & df["label"].notna()
    train_df   = df[train_mask].dropna(subset=feature_cols)

    MAX_TRAIN = 1_000_000
    if len(train_df) > MAX_TRAIN:
        train_df = train_df.sample(n=MAX_TRAIN, random_state=42)
        logger.info(f"  Subsampled training to {MAX_TRAIN} rows")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    logger.info(f"Training on {len(X_train)} samples, label dist: {np.bincount(y_train.astype(int))}")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training RF...")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── Predict on latest date ────────────────────────────────────────────────
    today_df = df[df["date"] == latest_date].dropna(subset=feature_cols).copy()
    logger.info(f"Stocks available for prediction on {latest_date}: {len(today_df)}")

    today_df["pred_prob"] = clf.predict_proba(today_df[feature_cols].values)[:, 1]

    # ── Top N picks ───────────────────────────────────────────────────────────
    top_picks = today_df.nlargest(TOP_N, "pred_prob")[["code", "pred_prob"]].reset_index(drop=True)
    top_picks.index += 1  # rank starts at 1

    print("\n" + "=" * 55)
    print(f"  TOP {TOP_N} PREDICTED STOCKS  (signal date: {latest_date})")
    print(f"  Target: >{RETURN_THRESHOLD:.0%} return in {FORWARD_DAYS} trading days")
    print("=" * 55)
    print(f"  {'Rank':<6} {'Code':<12} {'RF Confidence':>14}")
    print("-" * 55)
    for rank, row in top_picks.iterrows():
        print(f"  {rank:<6} {row['code']:<12} {row['pred_prob']:>13.1%}")
    print("=" * 55)
    print(f"\n  BUY  : at market close on {latest_date} (or open next day)")
    sell_info = f"3 trading days after {latest_date}"
    print(f"  SELL : at market close {sell_info}")
    print(f"  SIZE : equal weight — {100/TOP_N:.0f}% per stock\n")

    return top_picks


if __name__ == "__main__":
    main()
