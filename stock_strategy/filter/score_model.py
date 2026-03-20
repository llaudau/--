import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
STOCK_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_list.csv")

# ==============================================================================
# FILTER CONFIGURATION - Adjust these values as needed
# ==============================================================================
MIN_TURNOVER_AMOUNT = 50_000_000    # Minimum 成交额 (turnover amount) > 50M
MAX_TURNOVER_PCT = 0.15              # Maximum turnover < 15% (0.15 = 15%)
MIN_STOCK_LIFE_DAYS = 90             # Minimum stock life > 90 days (not new stock)

# ==============================================================================
# WINDOW CONFIGURATION - Adjust these values as needed
# ==============================================================================
MOMENTUM_WINDOW_SHORT = 20    # Short momentum window (days)
MOMENTUM_WINDOW_LONG = 40     # Long momentum window (days) - double ruler
VOLATILITY_WINDOW = 15        # Volatility calculation window (days)
VOLUME_WINDOW = 10            # Volume ratio calculation window (days)
TURNOVER_STABILITY_WINDOW = 20 # Turnover stability calculation window (days)
PVT_WINDOW = 10               # Price-volume Trend calculation window (days)
CLV_WINDOW = 1                # Close Location Value window (days, usually 1)

# Winsorization parameters
WINSORIZE_LOWER_PCT = 1   # Lower percentile cap (1%)
WINSORIZE_UPPER_PCT = 99  # Upper percentile cap (99%)

MIN_DATA_DAYS = max(MOMENTUM_WINDOW_LONG, VOLATILITY_WINDOW, VOLUME_WINDOW, 
                    TURNOVER_STABILITY_WINDOW, PVT_WINDOW) + 10

# ==============================================================================
# WEIGHT CONFIGURATION - Adjust these values as needed
# ==============================================================================
# Based on raw factor analysis:
# - Momentum has best spread (+2.22%) - HIGH weight
# - PVT has good spread (+1.57%) - moderate weight  
# - Volatility (inverted) has good but non-monotonic spread
# - Focus on momentum

W1 = 0.70   # Momentum: best spread, focus on this
W2 = 0.10   # Volatility: small weight (inverted)
W3 = 0.05   # Volume Ratio: small
W4 = 0.05   # Turnover Stability: small
W5 = 0.10   # PVT: moderate
W6 = 0.00   # CLV: removed


def load_stock_list():
    if os.path.exists(STOCK_LIST_FILE):
        return pd.read_csv(STOCK_LIST_FILE)
    return None


def calculate_metrics(df, target_date=None):
    """
    Calculate all 6 metrics for a given stock
    
    Parameters:
    - df: DataFrame with columns '日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', 'turnover'
    - target_date: specific date to calculate metrics for (if None, uses last available date)
    
    Returns:
    - dict with metrics or None if insufficient data
    """
    if df is None or len(df) < MIN_DATA_DAYS:
        return None
    
    df = df.sort_values('日期').reset_index(drop=True)
    
    if target_date:
        df = df[df['日期'] <= target_date]
        if len(df) < MIN_DATA_DAYS:
            return None
    
    recent_df = df.tail(MOMENTUM_WINDOW_LONG + 10).copy()
    
    if len(recent_df) < MOMENTUM_WINDOW_LONG:
        return None
    
    # 1. Momentum (Double Ruler: 20d + 40d average)
    # Mom20 = (close_t - close_{t-19}) / close_{t-19}
    # Mom40 = (close_t - close_{t-39}) / close_{t-39}
    close_current = recent_df['收盘'].iloc[-1]
    
    # Short momentum (20d)
    if len(recent_df) >= MOMENTUM_WINDOW_SHORT:
        close_20_ago = recent_df['收盘'].iloc[-MOMENTUM_WINDOW_SHORT]
        momentum_short = (close_current - close_20_ago) / close_20_ago if close_20_ago != 0 else 0
    else:
        momentum_short = 0
    
    # Long momentum (40d)
    if len(recent_df) >= MOMENTUM_WINDOW_LONG:
        close_40_ago = recent_df['收盘'].iloc[-MOMENTUM_WINDOW_LONG]
        momentum_long = (close_current - close_40_ago) / close_40_ago if close_40_ago != 0 else 0
    else:
        momentum_long = 0
    
    # Double ruler: average of short and long momentum
    momentum = (momentum_short + momentum_long) / 2
    
    # 2. Volatility (using VOLATILITY_WINDOW)
    recent_df['daily_return'] = (recent_df['收盘'] - recent_df['开盘']) / recent_df['开盘']
    volatility_df = recent_df.tail(VOLATILITY_WINDOW)
    volatility = volatility_df['daily_return'].std()
    
    # 3. Volume Ratio (using VOLUME_WINDOW)
    volume_df = recent_df.tail(VOLUME_WINDOW)
    avg_volume = volume_df['成交量'].mean()
    current_volume = recent_df['成交量'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # 4. Turnover Stability (using TURNOVER_STABILITY_WINDOW)
    turnover_df = recent_df.tail(TURNOVER_STABILITY_WINDOW)
    turnover_std = turnover_df['turnover'].std()
    turnover_mean = turnover_df['turnover'].mean()
    if turnover_mean > 0:
        cv = turnover_std / turnover_mean
        turnover_stability = 1 / (1 + cv)
    else:
        turnover_stability = 0.5
    
    # 5. Price-volume Trend (PVT) (using PVT_WINDOW)
    recent_df['volume_change'] = recent_df['成交量'].pct_change()
    recent_df['pvt'] = recent_df['daily_return'] * recent_df['volume_change']
    pvt_df = recent_df.tail(PVT_WINDOW)
    pvt = pvt_df['pvt'].sum()
    
    # 6. Close Location Value (CLV) (using CLV_WINDOW, usually 1 day)
    clv_df = recent_df.tail(CLV_WINDOW)
    close = clv_df['收盘'].iloc[-1]
    high = clv_df['最高'].iloc[-1]
    low = clv_df['最低'].iloc[-1]
    if high != low:
        clv = (close - low) / (high - low)
    else:
        clv = 0
    
    # ==============================================================================
    # APPLY FILTERS
    # ==============================================================================
    # Filter 1: 成交额 (turnover amount) > 50,000,000
    latest_turnover_amount = recent_df['成交额'].iloc[-1]
    if latest_turnover_amount < MIN_TURNOVER_AMOUNT:
        return None
    
    # Filter 2: Turnover < 15%
    current_turnover = recent_df['turnover'].iloc[-1]
    if current_turnover > MAX_TURNOVER_PCT:
        return None
    
    # Filter 3: Stock life > 90 days (not new stock)
    stock_life = len(df)
    if stock_life < MIN_STOCK_LIFE_DAYS:
        return None
    
    return {
        'momentum': momentum,
        'momentum_short': momentum_short,
        'momentum_long': momentum_long,
        'volatility': volatility,
        'volume_ratio': volume_ratio,
        'turnover_stability': turnover_stability,
        'pvt': pvt,
        'clv': clv,
        'turnover_amount': latest_turnover_amount,
        'current_turnover': current_turnover,
        'stock_life': stock_life,
        'days': len(recent_df)
    }


def winsorize(series, lower_pct=1, upper_pct=99):
    """Winsorize a series: cap values at lower and upper percentiles"""
    lower = series.quantile(lower_pct / 100)
    upper = series.quantile(upper_pct / 100)
    return series.clip(lower=lower, upper=upper)


def zscore(series):
    """Standardize to mean=0, std=1"""
    mean = series.mean()
    std = series.std()
    if std > 0:
        return (series - mean) / std
    return series - mean


def normalize_metrics(metrics_list):
    """
    1. Invert volatility (lower volatility = higher score)
    2. Winsorize all metrics (1st-99th percentile)
    3. Z-score standardization
    4. Scale to 0-100
    """
    if not metrics_list:
        return []
    
    df = pd.DataFrame(metrics_list)
    
    # List of raw metric columns
    metric_cols = ['momentum', 'volatility', 'volume_ratio', 'turnover_stability', 'pvt', 'clv']
    
    # 1. Invert volatility: lower volatility = higher score
    df['volatility'] = -df['volatility']
    
    # 2. Winsorize each metric (1st and 99th percentile)
    for col in metric_cols:
        df[col] = winsorize(df[col], lower_pct=1, upper_pct=99)
    
    # 3. Z-score standardization
    for col in metric_cols:
        df[col + '_zscore'] = zscore(df[col])
    
    # 4. Scale z-scores to 0-100 (min-max after z-score)
    for col in metric_cols:
        zscore_col = col + '_zscore'
        min_val = df[zscore_col].min()
        max_val = df[zscore_col].max()
        if max_val > min_val:
            df[col + '_norm'] = (df[zscore_col] - min_val) / (max_val - min_val) * 100
        else:
            df[col + '_norm'] = 50
    
    return df


def calculate_scores(target_date=None, w1=W1, w2=W2, w3=W3, w4=W4, w5=W5, w6=W6):
    """
    Calculate scores for all stocks
    
    Parameters:
    - target_date: specific date to calculate scores for (YYYY-MM-DD format)
      if None, uses the latest available date
    - w1-w6: weights for each metric
    
    Returns:
    - DataFrame with scores
    """
    print(f"Loading stock list...")
    stock_list = load_stock_list()
    
    if stock_list is None:
        print("Stock list not found!")
        return None
    
    stock_dict = dict(zip(stock_list['code'].astype(str).str.zfill(6), stock_list['name']))
    
    print(f"Calculating metrics for stocks...")
    
    metrics_list = []
    valid_codes = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for filename in files:
        code = filename.replace('.csv', '')
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            df = pd.read_csv(file_path)
            
            if '日期' not in df.columns:
                continue
            
            metrics = calculate_metrics(df, target_date)
            
            if metrics is None:
                continue
            
            metrics['code'] = code
            metrics['name'] = stock_dict.get(code, '')
            metrics_list.append(metrics)
            valid_codes.append(code)
            
        except Exception as e:
            continue
    
    if not metrics_list:
        print("No valid data found!")
        return None
    
    print(f"Calculated metrics for {len(metrics_list)} stocks")
    
    # Normalize metrics
    df = normalize_metrics(metrics_list)
    
    # Calculate weighted score
    df['score'] = (
        df['momentum_norm'] * w1 +
        df['volatility_norm'] * w2 +
        df['volume_ratio_norm'] * w3 +
        df['turnover_stability_norm'] * w4 +
        df['pvt_norm'] * w5 +
        df['clv_norm'] * w6
    )
    
    # Sort by score
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Select columns for output
    result = df[['rank', 'code', 'name', 
                 'momentum', 'volatility', 'volume_ratio', 
                 'turnover_stability', 'pvt', 'clv', 'score',
                 'momentum_norm', 'volatility_norm', 'volume_ratio_norm',
                 'turnover_stability_norm', 'pvt_norm', 'clv_norm']]
    
    return result


def save_report(result, target_date=None):
    if result is None or len(result) == 0:
        print("No results to save!")
        return
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    if target_date:
        date_str = target_date.replace('-', '')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    
    report_file = os.path.join(REPORT_DIR, f'score_ranking_{date_str}.csv')
    result.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    return report_file


def run_scoring(target_date=None, print_top=20):
    """
    Main function to run the scoring model
    
    Parameters:
    - target_date: specific date (YYYY-MM-DD), if None uses latest available
    - print_top: number of top stocks to display
    """
    print("="*70)
    print("Stock Scoring Model")
    print("="*70)
    print(f"Filter Configuration:")
    print(f"  MIN_TURNOVER_AMOUNT: {MIN_TURNOVER_AMOUNT:,.0f} (>50M)")
    print(f"  MAX_TURNOVER_PCT:    {MAX_TURNOVER_PCT:.1%} (<15%)")
    print(f"  MIN_STOCK_LIFE_DAYS: {MIN_STOCK_LIFE_DAYS} days (>90 days)")
    print("-"*70)
    print(f"Window Configuration:")
    print(f"  MOMENTUM_WINDOW_SHORT: {MOMENTUM_WINDOW_SHORT} days")
    print(f"  MOMENTUM_WINDOW_LONG:  {MOMENTUM_WINDOW_LONG} days (double ruler)")
    print(f"  VOLATILITY_WINDOW:     {VOLATILITY_WINDOW} days")
    print(f"  VOLUME_WINDOW:         {VOLUME_WINDOW} days")
    print(f"  TURNOVER_STABILITY_WINDOW: {TURNOVER_STABILITY_WINDOW} days")
    print(f"  PVT_WINDOW:            {PVT_WINDOW} days")
    print(f"  CLV_WINDOW:            {CLV_WINDOW} days")
    print("-"*70)
    print(f"Normalization:")
    print(f"  Winsorize: {WINSORIZE_LOWER_PCT}%-{WINSORIZE_UPPER_PCT}%")
    print(f"  Z-score standardization applied")
    print(f"  Volatility: INVERTED (lower volatility = higher score)")
    print("-"*70)
    print(f"Weights:")
    print(f"  W1 (Momentum):          {W1:.0%}")
    print(f"  W2 (Volatility):        {W2:.0%}")
    print(f"  W3 (Volume Ratio):      {W3:.0%}")
    print(f"  W4 (Turnover Stability):{W4:.0%}")
    print(f"  W5 (Price-volume Trend):{W5:.0%}")
    print(f"  W6 (Close Location):    {W6:.0%}")
    print("="*70)
    
    if target_date:
        print(f"Target date: {target_date}")
    else:
        print(f"Target date: Latest available")
    
    result = calculate_scores(target_date)
    
    if result is None:
        print("Failed to calculate scores!")
        return None
    
    report_file = save_report(result, target_date)
    
    print(f"\nTotal stocks scored: {len(result)}")
    print(f"Report saved to: {report_file}")
    
    print(f"\n{'='*70}")
    print(f"TOP {print_top} STOCKS BY SCORE")
    print(f"{'='*70}")
    print(result.head(print_top).to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"BOTTOM {print_top} STOCKS BY SCORE")
    print(f"{'='*70}")
    print(result.tail(print_top).to_string(index=False))
    
    return result


def run_scoring_custom_date(target_date, w1=W1, w2=W2, w3=W3, w4=W4, w5=W5, w6=W6):
    """Run scoring for a custom date with custom weights"""
    print("="*70)
    print(f"Stock Scoring Model - Date: {target_date}")
    print(f"Weights: W1={w1}, W2={w2}, W3={w3}, W4={w4}, W5={w5}, W6={w6}")
    print("="*70)
    
    global W1, W2, W3, W4, W5, W6
    W1, W2, W3, W4, W5, W6 = w1, w2, w3, w4, w5, w6
    
    result = calculate_scores(target_date, w1, w2, w3, w4, w5, w6)
    
    if result is None:
        print("Failed to calculate scores!")
        return None
    
    report_file = save_report(result, target_date)
    
    print(f"\nTotal stocks scored: {len(result)}")
    print(f"Report saved to: {report_file}")
    
    print(f"\nTop 10:")
    print(result.head(10)[['rank', 'code', 'name', 'score', 
                          'momentum', 'volatility', 'volume_ratio', 
                          'turnover_stability', 'pvt', 'clv']].to_string(index=False))
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
        run_scoring(target_date=target_date)
    else:
        run_scoring()
