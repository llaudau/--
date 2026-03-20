"""
Backtest for Rolling K Strategy with Staggered Layers

Strategy:
- On each selected date, buy top X% stocks from ranking
- Divide into 4 layers (equal money each layer)
- Each layer holds for 20 trading days
- Layer entry staggered by 20 days
- Weight within layer is linear proportional to score
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta


REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")

# ==============================================================================
# BACKTEST CONFIGURATION - Adjust these values
# ==============================================================================
BUY_PCT = 0.05       # Buy top 5% stocks (0.05 = 5%, 0.10 = 10%)
N_LAYERS = 4         # Number of layers (staggered entry)
LAYER_HOLD_DAYS = 20 # Days each layer holds
INTERVAL_DAYS = 5     # Days between each buy date

# Trading dates (20 rounds at 5-trading-day intervals, ending 30 days ago)
TRADING_DATES = [
    '2026-01-27', '2026-01-20', '2026-01-13', '2026-01-06', '2025-12-26',
    '2025-12-19', '2025-12-12', '2025-12-05', '2025-11-28', '2025-11-21',
    '2025-11-14', '2025-11-07', '2025-10-31', '2025-10-24', '2025-10-17',
    '2025-10-10', '2025-09-25', '2025-09-18', '2025-09-11', '2025-09-04'
]

# ==============================================================================


def get_stock_price(code, date):
    """Get stock price on a specific date"""
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        
        target_date = pd.to_datetime(date)
        df = df[df['日期'] <= target_date]
        
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        return {
            'close': latest['收盘'],
            'date': latest['日期'].strftime('%Y-%m-%d')
        }
    except Exception:
        return None


def get_future_price(code, start_date, days):
    """Get stock price after K days"""
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        
        target_date = pd.to_datetime(start_date)
        df = df[df['日期'] > target_date]
        
        if len(df) < days:
            return None
        
        future_date = df.iloc[days - 1]
        return {
            'close': future_date['收盘'],
            'date': future_date['日期'].strftime('%Y-%m-%d')
        }
    except Exception:
        return None


def load_ranking(date):
    """Load ranking file for a specific date"""
    date_str = date.replace('-', '')
    pattern = os.path.join(REPORT_DIR, f"score_ranking_{date_str}.csv")
    files = glob.glob(pattern)
    
    if not files:
        pattern = os.path.join(REPORT_DIR, f"mean_return_ranking_{date_str}.csv")
        files = glob.glob(pattern)
    
    if not files:
        print(f"  Warning: No ranking file found for {date}")
        return None
    
    try:
        df = pd.read_csv(files[0])
        return df
    except Exception as e:
        print(f"  Error loading ranking for {date}: {e}")
        return None


def calculate_linear_weights(scores):
    """Calculate linear weights proportional to score"""
    scores = np.array(scores)
    min_score = scores.min()
    weights = scores - min_score + 1e-6
    weights = weights / weights.sum()
    return weights


def run_backtest():
    """Run the backtest with staggered layers"""
    print("="*70)
    print("Rolling K Strategy Backtest - Staggered Layers")
    print("="*70)
    print(f"Configuration:")
    print(f"  BUY_PCT:       {BUY_PCT*100:.0f}% (top {BUY_PCT*100:.0f}% stocks)")
    print(f"  N_LAYERS:      {N_LAYERS} layers")
    print(f"  LAYER_HOLD:   {LAYER_HOLD_DAYS} trading days per layer")
    print(f"  INTERVAL:      {INTERVAL_DAYS} trading days between rounds")
    print(f"  ROUNDS:        {len(TRADING_DATES)} rounds")
    print(f"  Weight:        Linear proportional to score")
    print("="*70)
    
    round_results = []
    total_return = 0
    total_trades = 0
    successful_trades = 0
    
    for i, buy_date in enumerate(TRADING_DATES):
        print(f"\n[Round {i+1}/20] Buy Date: {buy_date}")
        
        ranking = load_ranking(buy_date)
        if ranking is None or ranking.empty:
            print(f"  Skipping - no ranking data")
            continue
        
        n_stocks = max(N_LAYERS, int(len(ranking) * BUY_PCT))
        top_stocks = ranking.head(n_stocks).copy()
        
        stocks_per_layer = n_stocks // N_LAYERS
        
        round_profit = 0
        round_trades = 0
        
        for layer_idx in range(N_LAYERS):
            start_idx = layer_idx * stocks_per_layer
            end_idx = start_idx + stocks_per_layer if layer_idx < N_LAYERS - 1 else n_stocks
            
            layer_stocks = top_stocks.iloc[start_idx:end_idx]
            if layer_stocks.empty:
                continue
            
            layer_entry_date = buy_date
            for _ in range(layer_idx):
                layer_entry_date = get_next_trading_date(layer_entry_date, INTERVAL_DAYS)
            
            scores = layer_stocks['score'].values
            weights = calculate_linear_weights(scores)
            
            layer_money = 1.0 / N_LAYERS
            
            print(f"  Layer {layer_idx+1}: {len(layer_stocks)} stocks, entry {layer_entry_date}, hold {LAYER_HOLD_DAYS}d")
            
            for j, (_, row) in enumerate(layer_stocks.iterrows()):
                code = str(row['code']).zfill(6)
                name = row.get('name', '')
                
                buy_price = get_stock_price(code, layer_entry_date)
                if buy_price is None:
                    continue
                
                sell_price = get_future_price(code, layer_entry_date, LAYER_HOLD_DAYS)
                if sell_price is None:
                    continue
                
                ret = (sell_price['close'] - buy_price['close']) / buy_price['close']
                weighted_ret = ret * weights[j] * layer_money
                round_profit += weighted_ret
                round_trades += 1
                total_trades += 1
                successful_trades += 1
                
                print(f"    {code} {name}: {weights[j]*100:.1f}% @ Buy {buy_price['close']:.2f} -> Sell {sell_price['close']:.2f} ({ret*100:+.2f}%)")
        
        if round_trades > 0:
            total_return += round_profit
            print(f"  Round {i+1} Return: {round_profit*100:+.2f}% ({round_trades} weighted trades)")
            round_results.append({
                'round': i+1,
                'buy_date': buy_date,
                'n_trades': round_trades,
                'return': round_profit
            })
        else:
            print(f"  No successful trades")
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Total Rounds:      {len(round_results)}")
    print(f"Total Trades:      {total_trades}")
    print(f"Total Return:      {total_return*100:+.2f}%")
    print(f"Annualized Return: {total_return * (250 / (len(TRADING_DATES) * INTERVAL_DAYS)) * 100:.2f}%")
    
    print("\n" + "-"*70)
    print("Round-by-Round Results:")
    print("-"*70)
    print(f"{'Round':<6} {'Buy Date':<12} {'Trades':<8} {'Return':<10}")
    print("-"*70)
    for r in round_results:
        print(f"{r['round']:<6} {r['buy_date']:<12} {r['n_trades']:<8} {r['return']*100:+.2f}%")
    
    results_df = pd.DataFrame(round_results)
    results_file = os.path.join(REPORT_DIR, f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    return round_results


def get_next_trading_date(start_date, days):
    """Get the trading date K days after start_date"""
    file_path = os.path.join(DATA_DIR, "000001.csv")
    if not os.path.exists(file_path):
        return start_date
    
    try:
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        
        target_date = pd.to_datetime(start_date)
        df = df[df['日期'] > target_date]
        
        if len(df) <= days:
            return start_date
        
        return df.iloc[days - 1]['日期'].strftime('%Y-%m-%d')
    except Exception:
        return start_date


def run_custom_backtest(buy_pct=0.05, n_layers=4, layer_hold_days=20, interval_days=5):
    """Run backtest with custom parameters"""
    global BUY_PCT, N_LAYERS, LAYER_HOLD_DAYS, INTERVAL_DAYS
    BUY_PCT = buy_pct
    N_LAYERS = n_layers
    LAYER_HOLD_DAYS = layer_hold_days
    INTERVAL_DAYS = interval_days
    return run_backtest()


if __name__ == "__main__":
    run_backtest()
