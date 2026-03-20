"""
Test different factor inversions and weight combinations
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


def test_raw_factor_weights():
    """Test using RAW factors instead of normalized ones"""
    print("="*60)
    print("Testing RAW Factors (not normalized)")
    print("="*60)
    
    hold_days = 20
    
    configs = [
        ('Raw Momentum Only', {'momentum': 1.0}),
        ('Raw PVT Only', {'pvt': 1.0}),
        ('Raw Momentum+PVT', {'momentum': 0.5, 'pvt': 0.5}),
        ('Raw Vol (inv)', {'volatility': -1.0}),
        ('Raw Vol (inv) + Momentum', {'momentum': 0.5, 'volatility': -0.5}),
        ('Raw Vol+PVT+Momentum', {'momentum': 0.4, 'volatility': -0.3, 'pvt': 0.3}),
    ]
    
    results = []
    
    for config_name, weights in configs:
        quantile_returns = {i: [] for i in range(10)}
        
        for rank_date in TRADING_DATES:
            ranking = load_ranking(rank_date)
            if ranking is None or ranking.empty:
                continue
            
            top_n = min(500, len(ranking))
            ranking = ranking.head(top_n).copy()
            
            score = 0
            for factor, weight in weights.items():
                if factor in ranking.columns:
                    score += weight * ranking[factor]
            ranking['custom_score'] = score
            
            returns_data = []
            for _, row in ranking.iterrows():
                code = str(row['code']).zfill(6)
                ret = get_stock_return(code, rank_date, hold_days)
                if ret is not None and -1 < ret < 5:
                    returns_data.append({'ret': ret, 'score': row['custom_score']})
            
            if len(returns_data) < 50:
                continue
            
            df = pd.DataFrame(returns_data)
            try:
                df['quantile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
                for q in range(10):
                    q_ret = df[df['quantile'] == q]['ret']
                    if len(q_ret) > 0:
                        quantile_returns[q].append(q_ret.mean())
            except:
                continue
        
        if any(quantile_returns.values()):
            means = [np.mean(quantile_returns[q]) * 100 if quantile_returns[q] else 0 for q in range(10)]
            
            monotonicity_score = sum([1 if means[i] < means[i+1] else 0 for i in range(9)])
            
            spread = means[-1] - means[0]
            
            print(f"\n{config_name}:")
            print(f"  Q1: {means[0]:+.2f}%, Q10: {means[9]:+.2f}%, Spread: {spread:+.2f}%")
            print(f"  Monotonicity: {monotonicity_score}/9")
            
            results.append({
                'name': config_name,
                'spread': spread,
                'monotonicity': monotonicity_score,
                'q1': means[0],
                'q10': means[9],
                'all_quantiles': means
            })
    
    print("\n" + "="*60)
    print("Best configs by spread:")
    results.sort(key=lambda x: x['spread'], reverse=True)
    for r in results[:5]:
        print(f"  {r['name']}: spread={r['spread']:+.2f}%, mono={r['monotonicity']}/9")
    
    return results


def test_with_different_hold_periods():
    """Test which hold periods work best"""
    print("\n" + "="*60)
    print("Testing Different Hold Periods (Momentum Only)")
    print("="*60)
    
    hold_periods = [5, 10, 20, 30, 40]
    
    for hold_days in hold_periods:
        quantile_returns = {i: [] for i in range(10)}
        
        for rank_date in TRADING_DATES:
            ranking = load_ranking(rank_date)
            if ranking is None or ranking.empty:
                continue
            
            top_n = min(500, len(ranking))
            ranking = ranking.head(top_n).copy()
            
            returns_data = []
            for _, row in ranking.iterrows():
                code = str(row['code']).zfill(6)
                ret = get_stock_return(code, rank_date, hold_days)
                if ret is not None and -1 < ret < 5:
                    returns_data.append({'ret': ret, 'score': row['momentum']})
            
            if len(returns_data) < 50:
                continue
            
            df = pd.DataFrame(returns_data)
            try:
                df['quantile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
                for q in range(10):
                    q_ret = df[df['quantile'] == q]['ret']
                    if len(q_ret) > 0:
                        quantile_returns[q].append(q_ret.mean())
            except:
                continue
        
        if any(quantile_returns.values()):
            means = [np.mean(quantile_returns[q]) * 100 if quantile_returns[q] else 0 for q in range(10)]
            spread = means[-1] - means[0]
            print(f"  Hold {hold_days}d: Q1={means[0]:+.2f}%, Q10={means[9]:+.2f}%, Spread={spread:+.2f}%")


if __name__ == "__main__":
    test_raw_factor_weights()
    test_with_different_hold_periods()
