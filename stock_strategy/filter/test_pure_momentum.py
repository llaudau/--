"""
Test pure momentum score
"""

import pandas as pd
import numpy as np
import os
import glob

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")

TRADING_DATES = [
    '2026-01-27', '2026-01-20', '2026-01-13', '2026-01-06', '2025-12-26',
    '2025-12-19', '2025-12-12', '2025-12-05', '2025-11-28', '2025-11-21',
    '2025-11-14', '2025-11-07', '2025-10-31', '2025-10-24', '2025-10-17',
    '2025-10-10', '2025-09-25', '2025-09-18', '2025-09-11', '2025-09-04'
]


def get_stock_return(code, start_date, days):
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        
        target_date = pd.to_datetime(start_date)
        df_start = df[df['日期'] <= target_date]
        if df_start.empty:
            return None
        start_price = df_start.iloc[-1]['收盘']
        
        df_future = df[df['日期'] > target_date]
        if len(df_future) < days:
            return None
        
        end_price = df_future.iloc[days - 1]['收盘']
        return (end_price - start_price) / start_price
    except:
        return None


def load_ranking(date):
    date_str = date.replace('-', '')
    pattern = os.path.join(REPORT_DIR, f"score_ranking_{date_str}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    try:
        return pd.read_csv(files[0])
    except:
        return None


def test_pure_momentum():
    """Test using raw momentum directly"""
    print("="*60)
    print("Testing PURE Raw Momentum (not normalized)")
    print("="*60)
    
    hold_days = 20
    
    for rank_date in TRADING_DATES[:3]:
        print(f"\n{rank_date}:")
        ranking = load_ranking(rank_date)
        if ranking is None:
            continue
        
        # Sort by raw momentum
        ranking_sorted = ranking.sort_values('momentum', ascending=False)
        
        # Top 50 by momentum
        top50 = ranking_sorted.head(50)
        bottom50 = ranking_sorted.tail(50)
        
        top_returns = []
        for _, row in top50.iterrows():
            ret = get_stock_return(str(row['code']).zfill(6), rank_date, hold_days)
            if ret and -1 < ret < 5:
                top_returns.append(ret)
        
        bottom_returns = []
        for _, row in bottom50.iterrows():
            ret = get_stock_return(str(row['code']).zfill(6), rank_date, hold_days)
            if ret and -1 < ret < 5:
                bottom_returns.append(ret)
        
        if top_returns and bottom_returns:
            print(f"  Top 50 momentum: {np.mean(top_returns)*100:+.2f}%")
            print(f"  Bottom 50 momentum: {np.mean(bottom_returns)*100:+.2f}%")
        
        # Also check by score (current ranking)
        score_top50 = ranking.sort_values('score', ascending=False).head(50)
        score_bottom50 = ranking.sort_values('score', ascending=False).tail(50)
        
        score_top_returns = []
        for _, row in score_top50.iterrows():
            ret = get_stock_return(str(row['code']).zfill(6), rank_date, hold_days)
            if ret and -1 < ret < 5:
                score_top_returns.append(ret)
        
        score_bottom_returns = []
        for _, row in score_bottom50.iterrows():
            ret = get_stock_return(str(row['code']).zfill(6), rank_date, hold_days)
            if ret and -1 < ret < 5:
                score_bottom_returns.append(ret)
        
        if score_top_returns and score_bottom_returns:
            print(f"  Top 50 by SCORE: {np.mean(score_top_returns)*100:+.2f}%")
            print(f"  Bottom 50 by SCORE: {np.mean(score_bottom_returns)*100:+.2f}%")
            print(f"  Top 50 avg momentum: {score_top50['momentum'].mean():.4f}")
            print(f"  Bottom 50 avg momentum: {score_bottom50['momentum'].mean():.4f}")


if __name__ == "__main__":
    test_pure_momentum()
