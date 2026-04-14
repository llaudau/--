"""Daily prediction generator — saves top-N picks to data/predictions/ folder.

Run this daily after market close to generate tomorrow's buy list.
Each prediction file contains:
  - Buy date (tomorrow)
  - Sell date (3 trading days later)
  - Top-N stock codes with confidence scores
  - Portfolio instructions

Usage:
    python -m model.RF.generate_daily_prediction
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLEANED_OHLCV_DIR, RAW_BASICS_DIR, POOL_DIR, PREDICTIONS_DIR
from model.features import load_and_build_features
from model.RF.config import RF_PARAMS, FORWARD_DAYS, RETURN_THRESHOLD, TOP_N, MAX_TRAIN_SAMPLES
from model.RF.rf_model import RFModel
from stock_pool.pool_builder import load_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_next_trading_date(from_date: pd.Timestamp, n_days: int = 1) -> str:
    """Estimate next trading date (assumes Mon-Fri, no holidays)."""
    current = from_date
    count = 0
    while count < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            count += 1
    return current.strftime('%Y-%m-%d')


def main():
    logger.info("=== Daily Prediction Generator ===\n")

    # Load stock pool
    pool_df = load_pool(POOL_DIR)
    pool_codes = set(pool_df["code"].tolist())
    logger.info(f"Stock pool: {len(pool_codes)} stocks")

    # Find latest basics file
    basics_files = sorted(RAW_BASICS_DIR.glob("basics_*.parquet"))
    if not basics_files:
        raise FileNotFoundError(f"No basics parquet in {RAW_BASICS_DIR}")
    basics_path = basics_files[-1]

    # Load features
    df, feature_cols, _ = load_and_build_features(
        ohlcv_dir=CLEANED_OHLCV_DIR,
        basics_path=basics_path,
        pool_codes=pool_codes,
        forward_days=FORWARD_DAYS,
        return_threshold=RETURN_THRESHOLD,
    )

    latest_date = df["date"].max()
    logger.info(f"Latest data date: {latest_date.date()}")

    # Training: exclude last FORWARD_DAYS dates to prevent label leakage
    all_dates = sorted(df["date"].unique())
    safe_train_end = all_dates[-(FORWARD_DAYS + 1)]

    train_mask = (df["date"] <= safe_train_end) & df["label"].notna()
    train_df = df[train_mask].dropna(subset=feature_cols)

    if len(train_df) > MAX_TRAIN_SAMPLES:
        train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)
        logger.info(f"Subsampled training to {MAX_TRAIN_SAMPLES} rows")

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    logger.info(f"Training on {len(X_train)} samples...")

    # Train model
    model = RFModel(config={"rf_params": RF_PARAMS,
                             "forward_days": FORWARD_DAYS,
                             "return_threshold": RETURN_THRESHOLD})
    model.train(X_train, y_train)

    # Predict on latest date
    today_df = df[df["date"] == latest_date].dropna(subset=feature_cols).copy()
    logger.info(f"Stocks available on {latest_date.date()}: {len(today_df)}")

    today_df["pred_prob"] = model.predict_proba(today_df[feature_cols]).values
    top_picks = today_df.nlargest(TOP_N, "pred_prob")[["code", "pred_prob"]].reset_index(drop=True)

    # Calculate buy/sell dates
    signal_date = latest_date.date()
    buy_date = get_next_trading_date(latest_date, 1)  # Next trading day
    sell_date = get_next_trading_date(latest_date, FORWARD_DAYS + 1)  # FORWARD_DAYS after buy

    # Save prediction file
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_filename = f"prediction_{signal_date.strftime('%Y%m%d')}.csv"
    pred_path = PREDICTIONS_DIR / pred_filename

    # Add metadata columns
    top_picks["signal_date"] = signal_date
    top_picks["buy_date"] = buy_date
    top_picks["sell_date"] = sell_date
    top_picks["forward_days"] = FORWARD_DAYS
    top_picks["return_threshold"] = RETURN_THRESHOLD
    top_picks["position_size_pct"] = 100 / TOP_N

    top_picks.to_csv(pred_path, index=False)
    logger.info(f"Prediction saved to {pred_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"  DAILY PREDICTION  (signal: {signal_date})")
    print("=" * 70)
    print(f"  Strategy: RF Top-{TOP_N} | fwd={FORWARD_DAYS}d | thresh={RETURN_THRESHOLD:.0%}")
    print(f"  BUY:  {buy_date} (market open)")
    print(f"  SELL: {sell_date} (market close)")
    print("=" * 70)
    print(f"  {'Rank':<6} {'Code':<12} {'Confidence':>12} {'Position':>10}")
    print("-" * 70)
    for i, row in top_picks.iterrows():
        print(f"  {i+1:<6} {row['code']:<12} {row['pred_prob']:>11.1%} {row['position_size_pct']:>9.1f}%")
    print("=" * 70)
    print(f"\n  Portfolio Action:")
    print(f"    - BUY these {TOP_N} stocks on {buy_date}")
    print(f"    - SELL the {TOP_N} stocks you bought on {get_next_trading_date(latest_date - timedelta(days=FORWARD_DAYS), 1)}")
    print(f"    - Hold period: {FORWARD_DAYS} trading days")
    print(f"    - Position size: {100/TOP_N:.1f}% per stock (equal weight)\n")

    return top_picks


if __name__ == "__main__":
    main()
