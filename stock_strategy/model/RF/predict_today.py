"""Generate today's top stock picks using the RF model with best parameters."""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLEANED_OHLCV_DIR, RAW_BASICS_DIR, POOL_DIR, MODELS_DIR
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


def main():
    logger.info("=== RF Stock Predictor ===")
    logger.info(f"  forward_days={FORWARD_DAYS}, return_threshold={RETURN_THRESHOLD:.0%}, top_n={TOP_N}")
    logger.info(f"  max_depth={RF_PARAMS['max_depth']}, min_samples_leaf={RF_PARAMS['min_samples_leaf']}\n")

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
    logger.info(f"Latest date in data: {latest_date.date()}")

    # Training rows: exclude last FORWARD_DAYS dates to prevent label leakage
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

    # Train
    model = RFModel(config={"rf_params": RF_PARAMS,
                             "forward_days": FORWARD_DAYS,
                             "return_threshold": RETURN_THRESHOLD})
    model.train(X_train, y_train)

    # Save trained model
    model_path = MODELS_DIR / "RF" / "rf_model_latest.joblib"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Predict on latest date
    today_df = df[df["date"] == latest_date].dropna(subset=feature_cols).copy()
    logger.info(f"Stocks available on {latest_date.date()}: {len(today_df)}")

    today_df["pred_prob"] = model.predict_proba(today_df[feature_cols]).values
    top_picks = today_df.nlargest(TOP_N, "pred_prob")[["code", "pred_prob"]].reset_index(drop=True)
    top_picks.index += 1

    print("\n" + "=" * 57)
    print(f"  TOP {TOP_N} PREDICTED STOCKS  (signal date: {latest_date.date()})")
    print(f"  Target: >{RETURN_THRESHOLD:.0%} return in {FORWARD_DAYS} trading days")
    print("=" * 57)
    print(f"  {'Rank':<6} {'Code':<12} {'RF Confidence':>14}")
    print("-" * 57)
    for rank, row in top_picks.iterrows():
        print(f"  {rank:<6} {row['code']:<12} {row['pred_prob']:>13.1%}")
    print("=" * 57)
    print(f"\n  BUY  : at market close on {latest_date.date()} (or next open)")
    print(f"  SELL : {FORWARD_DAYS} trading days later")
    print(f"  SIZE : equal weight — {100/TOP_N:.0f}% per stock\n")

    return top_picks


if __name__ == "__main__":
    main()
