"""
Analyze factor predictive power and optimize weights
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")

TRADING_DATES = [
    '2026-01-27', '2026-01-20', '2026-01-13', '2026-01-06', '2025-12-26',
    '2025-12-19', '2025-12-12', '2025-12-05', '2025-11-28', '2025-11-21',
    '2025-11-14', '2025-11-07', '2025-10-31', '2025-10-24', '2025-10-17',
    '2025-10-10', '2025-09-25', '2025-09-18', '2025-09-11', '2025-09-04'
]

FACTORS = ['momentum', 'volatility', 'volume_ratio', 'turnover_stability', 'pvt', 'clv']


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


def analyze_factor_ic():
    """Analyze IC for each individual factor"""
    print("="*60)
    print("Factor-by-Factor IC Analysis (20d hold)")
    print("="*60)
    
    factor_ics = {f: [] for f in FACTORS}
    
    for rank_date in TRADING_DATES:
        print(f"  Processing {rank_date}...")
        ranking = load_ranking(rank_date)
        if ranking is None or ranking.empty:
            continue
        
        top_n = min(200, len(ranking))
        ranking = ranking.head(top_n)
        
        returns = []
        factor_values = {f: [] for f in FACTORS}
        
        for _, row in ranking.iterrows():
            code = str(row['code']).zfill(6)
            ret = get_stock_return(code, rank_date, 20)
            if ret is not None and -1 < ret < 5:
                returns.append(ret)
                for f in FACTORS:
                    factor_values[f].append(row.get(f, 0))
        
        if len(returns) < 20:
            continue
        
        returns = np.array(returns)
        for f in FACTORS:
            fvals = np.array(factor_values[f])
            if len(fvals) == len(returns):
                corr = np.corrcoef(fvals, returns)[0, 1]
                if not np.isnan(corr):
                    factor_ics[f].append(corr)
    
    print("\nFactor ICs:")
    for f in FACTORS:
        if factor_ics[f]:
            mean_ic = np.mean(factor_ics[f])
            print(f"  {f:20s}: {mean_ic:+.4f} ({len(factor_ics[f])} samples)")
    
    return factor_ics


def analyze_raw_factor_quantiles():
    """Analyze returns by raw factor values (not normalized)"""
    print("\n" + "="*60)
    print("Raw Factor Quantile Analysis")
    print("="*60)
    
    hold_days = 20
    
    for factor in FACTORS:
        quantile_returns = {i: [] for i in range(5)}
        
        for rank_date in TRADING_DATES:
            ranking = load_ranking(rank_date)
            if ranking is None or ranking.empty:
                continue
            
            top_n = min(300, len(ranking))
            ranking = ranking.head(top_n).copy()
            
            returns_data = []
            for _, row in ranking.iterrows():
                code = str(row['code']).zfill(6)
                ret = get_stock_return(code, rank_date, hold_days)
                if ret is not None and -1 < ret < 5:
                    returns_data.append({'ret': ret, 'factor': row.get(factor, 0)})
            
            if len(returns_data) < 50:
                continue
            
            df = pd.DataFrame(returns_data)
            try:
                df['quantile'] = pd.qcut(df['factor'], 5, labels=False, duplicates='drop')
                for q in range(5):
                    q_ret = df[df['quantile'] == q]['ret']
                    if len(q_ret) > 0:
                        quantile_returns[q].append(q_ret.mean())
            except:
                continue
        
        if any(quantile_returns.values()):
            print(f"\n{factor}:")
            means = [np.mean(quantile_returns[q]) * 100 if quantile_returns[q] else 0 for q in range(5)]
            for q in range(5):
                print(f"  Q{q+1}: {means[q]:+.2f}%")
            spread = means[-1] - means[0]
            print(f"  Spread (Q5-Q1): {spread:+.2f}%")


def test_different_weights():
    """Test different weight combinations"""
    print("\n" + "="*60)
    print("Testing Different Weight Combinations")
    print("="*60)
    
    weight_configs = [
        {
            'name': 'Low Vol Focus',
            'W1': 0.10, 'W2': 0.40, 'W3': 0.10, 'W4': 0.10, 'W5': 0.15, 'W6': 0.15
        },
        {
            'name': 'Momentum Only',
            'W1': 0.50, 'W2': 0.10, 'W3': 0.10, 'W4': 0.10, 'W5': 0.10, 'W6': 0.10
        },
        {
            'name': 'Inverse Vol + PVT',
            'W1': 0.10, 'W2': 0.25, 'W3': 0.10, 'W4': 0.15, 'W5': 0.25, 'W6': 0.15
        },
        {
            'name': 'Equal Weights',
            'W1': 1/6, 'W2': 1/6, 'W3': 1/6, 'W4': 1/6, 'W5': 1/6, 'W6': 1/6
        },
        {
            'name': 'High PVT+CLV',
            'W1': 0.15, 'W2': 0.15, 'W3': 0.10, 'W4': 0.10, 'W5': 0.25, 'W6': 0.25
        },
        {
            'name': 'Turnover Stability Focus',
            'W1': 0.15, 'W2': 0.15, 'W3': 0.10, 'W4': 0.35, 'W5': 0.10, 'W6': 0.15
        },
    ]
    
    hold_days = 20
    
    results = []
    
    for config in weight_configs:
        print(f"\nTesting: {config['name']}")
        
        quantile_returns = {i: [] for i in range(10)}
        
        for rank_date in TRADING_DATES:
            ranking = load_ranking(rank_date)
            if ranking is None or ranking.empty:
                continue
            
            top_n = min(500, len(ranking))
            ranking = ranking.head(top_n).copy()
            
            score = (
                config['W1'] * ranking['momentum_norm'] +
                config['W2'] * ranking['volatility_norm'] +
                config['W3'] * ranking['volume_ratio_norm'] +
                config['W4'] * ranking['turnover_stability_norm'] +
                config['W5'] * ranking['pvt_norm'] +
                config['W6'] * ranking['clv_norm']
            )
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
            spread = means[-1] - means[0]
            
            print(f"  Q1: {means[0]:+.2f}%, Q10: {means[9]:+.2f}%, Spread: {spread:+.2f}%")
            
            results.append({
                'name': config['name'],
                'spread': spread,
                'q1': means[0],
                'q10': means[9],
                'config': config
            })
    
    print("\n" + "="*60)
    print("Summary - Best Configs by Spread:")
    print("="*60)
    results.sort(key=lambda x: x['spread'], reverse=True)
    for r in results[:5]:
        print(f"  {r['name']}: spread = {r['spread']:+.2f}%")
    
    return results


def main():
    factor_ics = analyze_factor_ic()
    analyze_raw_factor_quantiles()
    results = test_different_weights()
    
    if results:
        best = results[0]
        print(f"\nBest config: {best['name']}")
        print(f"  W1={best['config']['W1']}, W2={best['config']['W2']}, W3={best['config']['W3']}")
        print(f"  W4={best['config']['W4']}, W5={best['config']['W5']}, W6={best['config']['W6']}")


if __name__ == "__main__":
    main()
