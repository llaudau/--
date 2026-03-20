"""
Debug quantile analysis
"""

import pandas as pd
import numpy as np
import os
import glob

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")

TRADING_DATES = ['2026-01-20']


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


rank_date = '2026-01-20'
hold_days = 20

ranking = load_ranking(rank_date)
if ranking is None:
    print("No ranking found")
    exit()

top_n = min(500, len(ranking))
ranking = ranking.head(top_n).copy()

returns_data = []
for _, row in ranking.iterrows():
    code = str(row['code']).zfill(6)
    ret = get_stock_return(code, rank_date, hold_days)
    if ret is not None and -1 < ret < 5:
        returns_data.append({'code': code, 'ret': ret, 'score': row['score'], 'momentum': row['momentum']})

df = pd.DataFrame(returns_data)

print(f"Total stocks with returns: {len(df)}")
print(f"\nScore range: {df['score'].min():.2f} to {df['score'].max():.2f}")
print(f"Momentum range: {df['momentum'].min():.4f} to {df['momentum'].max():.4f}")

# Calculate quantile
df['quantile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')

print("\nQuantile analysis:")
for q in range(10):
    q_df = df[df['quantile'] == q]
    if len(q_df) > 0:
        print(f"Q{q+1}: score={q_df['score'].mean():.1f}, ret={q_df['ret'].mean()*100:+.2f}%, momentum={q_df['momentum'].mean():.4f}, n={len(q_df)}")

# Check correlation
corr_score_ret = np.corrcoef(df['score'], df['ret'])[0,1]
corr_momentum_ret = np.corrcoef(df['momentum'], df['ret'])[0,1]
print(f"\nCorrelation(score, ret): {corr_score_ret:+.4f}")
print(f"Correlation(momentum, ret): {corr_momentum_ret:+.4f}")
