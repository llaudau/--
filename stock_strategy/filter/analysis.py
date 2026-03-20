"""
IC, Quantile Returns, and Turnover Analysis - Optimized Version
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

TRADING_DATES = [
    '2026-01-27', '2026-01-20', '2026-01-13', '2026-01-06', '2025-12-26',
    '2025-12-19', '2025-12-12', '2025-12-05', '2025-11-28', '2025-11-21',
    '2025-11-14', '2025-11-07', '2025-10-31', '2025-10-24', '2025-10-17',
    '2025-10-10', '2025-09-25', '2025-09-18', '2025-09-11', '2025-09-04'
]


def get_stock_price_series(code):
    """Get stock price series"""
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        df = df.set_index('日期')
        return df['收盘']
    except:
        return None


def get_return(code, start_date, days):
    """Get return over K days from start_date"""
    prices = get_stock_price_series(code)
    if prices is None:
        return None
    
    try:
        start_date = pd.to_datetime(start_date)
        if start_date not in prices.index:
            prices = prices[prices.index <= start_date]
            if len(prices) == 0:
                return None
            start_price = prices.iloc[-1]
        else:
            start_price = prices.loc[start_date]
        
        future_idx = prices.index[prices.index > start_date]
        if len(future_idx) < days:
            return None
        
        target_date = future_idx[days - 1]
        end_price = prices.loc[target_date]
        
        return (end_price - start_price) / start_price
    except:
        return None


def load_ranking(date):
    """Load ranking file"""
    date_str = date.replace('-', '')
    pattern = os.path.join(REPORT_DIR, f"score_ranking_{date_str}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    try:
        return pd.read_csv(files[0])
    except:
        return None


def spearman_ic(x, y):
    """Calculate Spearman rank IC"""
    n = len(x)
    if n < 3:
        return np.nan
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    x_mean = np.mean(x_rank)
    y_mean = np.mean(y_rank)
    numerator = np.sum((x_rank - x_mean) * (y_rank - y_mean))
    denominator = np.sqrt(np.sum((x_rank - x_mean)**2) * np.sum((y_rank - y_mean)**2))
    if denominator == 0:
        return np.nan
    return numerator / denominator


def analyze_ic_decay():
    """Calculate IC decay"""
    print("="*60)
    print("1. IC Decay Analysis")
    print("="*60)
    
    taus = [5, 10, 15, 20, 25, 30]
    ic_by_tau = {tau: [] for tau in taus}
    
    for rank_date in TRADING_DATES:
        print(f"  Processing {rank_date}...")
        ranking = load_ranking(rank_date)
        if ranking is None or ranking.empty:
            continue
        
        top_n = min(200, len(ranking))
        ranking = ranking.head(top_n)
        
        for tau in taus:
            returns = []
            scores = []
            
            for _, row in ranking.iterrows():
                code = str(row['code']).zfill(6)
                ret = get_return(code, rank_date, tau)
                if ret is not None and -2 < ret < 10:
                    returns.append(ret)
                    scores.append(row['score'])
            
            if len(returns) > 20:
                ic = spearman_ic(np.array(scores), np.array(returns))
                if not np.isnan(ic):
                    ic_by_tau[tau].append(ic)
    
    mean_ics = []
    std_ics = []
    
    for tau in taus:
        if ic_by_tau[tau]:
            mean_ic = np.mean(ic_by_tau[tau])
            std_ic = np.std(ic_by_tau[tau])
            mean_ics.append(mean_ic)
            std_ics.append(std_ic)
            print(f"  τ={tau:2d}: IC = {mean_ic:+.4f} ± {std_ic:.4f} ({len(ic_by_tau[tau])} samples)")
        else:
            mean_ics.append(0)
            std_ics.append(0)
    
    return taus, mean_ics, std_ics


def analyze_quantile_returns():
    """Calculate quantile return spread"""
    print("\n" + "="*60)
    print("2. Quantile Return Analysis (10 groups, 20d hold)")
    print("="*60)
    
    hold_days = 20
    quantile_returns = {i: [] for i in range(10)}
    
    for rank_date in TRADING_DATES:
        print(f"  Processing {rank_date}...")
        ranking = load_ranking(rank_date)
        if ranking is None or ranking.empty:
            continue
        
        top_n = min(300, len(ranking))
        ranking = ranking.head(top_n).copy()
        
        returns_data = []
        for _, row in ranking.iterrows():
            code = str(row['code']).zfill(6)
            ret = get_return(code, rank_date, hold_days)
            if ret is not None and -2 < ret < 10:
                returns_data.append({'code': code, 'ret': ret, 'score': row['score']})
        
        if len(returns_data) < 50:
            continue
        
        returns_df = pd.DataFrame(returns_data)
        returns_df['quantile'] = pd.qcut(returns_df['score'], 10, labels=False, duplicates='drop')
        
        for q in range(10):
            q_returns = returns_df[returns_df['quantile'] == q]['ret']
            if len(q_returns) > 0:
                quantile_returns[q].append(q_returns.mean())
    
    quantiles = list(range(10))
    mean_returns = [np.mean(quantile_returns[q]) * 100 if quantile_returns[q] else 0 for q in quantiles]
    
    print("\n  Decile Returns:")
    for q in range(10):
        print(f"    Q{q+1}: {mean_returns[q]:+.2f}%")
    
    spread = mean_returns[-1] - mean_returns[0]
    print(f"\n  Q10 - Q1 Spread: {spread:+.2f}%")
    
    return quantiles, mean_returns


def analyze_turnover_alpha():
    """Analyze turnover vs alpha"""
    print("\n" + "="*60)
    print("3. Turnover vs Alpha Analysis")
    print("="*60)
    
    intervals = [5, 10, 20, 40]
    hold_days = 20
    
    results = []
    
    for interval in intervals:
        print(f"\n  Testing interval = {interval} days...")
        
        selected_dates = TRADING_DATES[::interval//5]
        
        total_return = 0
        total_rounds = 0
        
        for rank_date in selected_dates:
            ranking = load_ranking(rank_date)
            if ranking is None or ranking.empty:
                continue
            
            top_stocks = ranking.head(int(len(ranking) * 0.05))
            
            if top_stocks.empty:
                continue
            
            returns = []
            for _, row in top_stocks.iterrows():
                code = str(row['code']).zfill(6)
                ret = get_return(code, rank_date, hold_days)
                if ret is not None and -2 < ret < 10:
                    returns.append(ret)
            
            if returns:
                total_return += np.mean(returns)
                total_rounds += 1
        
        if total_rounds > 0:
            avg_return = total_return / total_rounds
            annualized = avg_return * (250 / interval) * 100
            results.append({
                'interval': interval,
                'avg_return': avg_return * 100,
                'annualized': annualized,
                'rounds': total_rounds
            })
            print(f"    Avg Return: {avg_return*100:+.2f}%, Annualized: {annualized:+.2f}%")
    
    intervals_result = [r['interval'] for r in results]
    annualized_returns = [r['annualized'] for r in results]
    
    return intervals_result, annualized_returns


def main():
    """Run all analyses and generate plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    taus, mean_ics, std_ics = analyze_ic_decay()
    
    axes[0].bar(taus, mean_ics, yerr=std_ics, color='steelblue', alpha=0.7, capsize=3)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('τ (days)')
    axes[0].set_ylabel('Rank IC')
    axes[0].set_title('IC Decay: corr(Score_t, Return_{t+τ})')
    axes[0].grid(True, alpha=0.3)
    
    quantiles, mean_returns = analyze_quantile_returns()
    
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', 
             '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
    colors = colors[::-1]
    axes[1].bar(range(10), mean_returns, color=colors)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Quantile (1=Low Score, 10=High Score)')
    axes[1].set_ylabel('Avg Return (%)')
    axes[1].set_title('Quantile Return Spread (20d hold)')
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([f'Q{i+1}' for i in range(10)])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    intervals_result, annualized_returns = analyze_turnover_alpha()
    
    bars = axes[2].bar(range(len(intervals_result)), annualized_returns, color='coral')
    axes[2].set_xlabel('Rebalance Interval (days)')
    axes[2].set_ylabel('Annualized Return (%)')
    axes[2].set_title('Turnover vs Alpha')
    axes[2].set_xticks(range(len(intervals_result)))
    axes[2].set_xticklabels([str(i) for i in intervals_result])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(annualized_returns):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(REPORT_DIR, "analysis_report.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    return {
        'ic': (taus, mean_ics, std_ics),
        'quantile': (quantiles, mean_returns),
        'turnover': (intervals_result, annualized_returns)
    }


if __name__ == "__main__":
    main()
