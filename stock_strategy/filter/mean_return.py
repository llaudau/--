import pandas as pd
import numpy as np
import os
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_daily")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
STOCK_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "stock_list.csv")


def load_stock_list():
    if os.path.exists(STOCK_LIST_FILE):
        return pd.read_csv(STOCK_LIST_FILE)
    return None


def calculate_returns(df):
    if df is None or len(df) < 2:
        return None
    
    df = df.sort_values('日期')
    
    df['daily_return'] = (df['收盘'] - df['开盘']) / df['开盘']
    
    avg_return = df['daily_return'].mean()
    
    volatility = df['daily_return'].std()
    
    total_return = (df['收盘'].iloc[-1] - df['开盘'].iloc[0]) / df['开盘'].iloc[0]
    
    return {
        'avg_daily_return': avg_return,
        'volatility': volatility,
        'total_return': total_return,
        'sharpe_ratio': avg_return / volatility if volatility > 0 else 0,
        'days': len(df)
    }


def rank_stocks():
    print("Loading stock list...")
    stock_list = load_stock_list()
    
    if stock_list is None:
        print("Stock list not found!")
        return
    
    stock_dict = dict(zip(stock_list['code'].astype(str).str.zfill(6), stock_list['name']))
    
    print(f"Processing stocks from {DATA_DIR}...")
    
    results = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for filename in files:
        code = filename.replace('.csv', '')
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) < 5:
                continue
            
            returns = calculate_returns(df)
            
            if returns is None:
                continue
            
            name = stock_dict.get(code, '')
            
            results.append({
                'code': code,
                'name': name,
                'avg_daily_return': returns['avg_daily_return'],
                'volatility': returns['volatility'],
                'total_return': returns['total_return'],
                'sharpe_ratio': returns['sharpe_ratio'],
                'days': returns['days']
            })
            
        except Exception as e:
            print(f"Error processing {code}: {e}")
            continue
    
    if not results:
        print("No valid data found!")
        return
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('avg_daily_return', ascending=False)
    
    results_df['rank'] = range(1, len(results_df) + 1)
    
    results_df = results_df[['rank', 'code', 'name', 'avg_daily_return', 'volatility', 'total_return', 'sharpe_ratio', 'days']]
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    today = datetime.now().strftime('%Y%m%d')
    report_file = os.path.join(REPORT_DIR, f'mean_return_ranking_{today}.csv')
    
    results_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"Ranking completed!")
    print(f"Total stocks ranked: {len(results_df)}")
    print(f"Report saved to: {report_file}")
    print(f"\nTop 10 stocks by average daily return:")
    print(results_df.head(10).to_string(index=False))
    print(f"\nBottom 10 stocks by average daily return:")
    print(results_df.tail(10).to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    rank_stocks()
