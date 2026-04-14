"""View historical predictions and portfolio actions.

Usage:
    python -m model.RF.view_predictions              # Show all predictions
    python -m model.RF.view_predictions --latest     # Show only latest
    python -m model.RF.view_predictions --date 20260413  # Show specific date
"""
import argparse
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PREDICTIONS_DIR


def main():
    parser = argparse.ArgumentParser(description="View prediction history")
    parser.add_argument("--latest", action="store_true", help="Show only latest prediction")
    parser.add_argument("--date", type=str, help="Show prediction for specific date (YYYYMMDD)")
    args = parser.parse_args()

    pred_files = sorted(PREDICTIONS_DIR.glob("prediction_*.csv"))

    if not pred_files:
        print(f"No predictions found in {PREDICTIONS_DIR}")
        return

    if args.latest:
        pred_files = [pred_files[-1]]
    elif args.date:
        pred_files = [f for f in pred_files if args.date in f.name]
        if not pred_files:
            print(f"No prediction found for date {args.date}")
            return

    print("=" * 80)
    print("  PREDICTION HISTORY")
    print("=" * 80)

    for pred_file in pred_files:
        df = pd.read_csv(pred_file, dtype={'code': str})

        signal_date = df['signal_date'].iloc[0]
        buy_date = df['buy_date'].iloc[0]
        sell_date = df['sell_date'].iloc[0]
        fwd_days = int(df['forward_days'].iloc[0])
        thresh = float(df['return_threshold'].iloc[0])

        print(f"\n📅 Signal: {signal_date} | Buy: {buy_date} | Sell: {sell_date}")
        print(f"   Strategy: fwd={fwd_days}d, thresh={thresh:.0%}, top-{len(df)}")
        print("-" * 80)
        print(f"  {'Rank':<6} {'Code':<12} {'Confidence':>12} {'Position':>10}")
        print("-" * 80)

        for i, row in df.iterrows():
            print(f"  {i+1:<6} {row['code']:<12} {row['pred_prob']:>11.1%} {row['position_size_pct']:>9.1f}%")

        print("-" * 80)
        print(f"  Portfolio Action:")
        print(f"    ✅ BUY:  {', '.join(df['code'].tolist())} on {buy_date}")
        print(f"    ❌ SELL: stocks from 3 days ago on {buy_date}")
        print()

    print("=" * 80)
    print(f"Total predictions: {len(pred_files)}")
    print(f"Prediction folder: {PREDICTIONS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
