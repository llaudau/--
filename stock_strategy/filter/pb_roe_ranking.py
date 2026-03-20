import os
import pandas as pd
import baostock as bs
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def is_workday():
    today = datetime.now()
    if today.weekday() >= 5:
        return False
    return True

def get_stock_list():
    cache_file = os.path.join(CACHE_DIR, 'stock_list.csv')
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        if len(df) > 1000:
            print("Loading stock list from cache...")
            return df
    
    print("Fetching A-share stock list from baostock...")
    lg = bs.login()
    
    rs = bs.query_stock_industry()
    stocks = []
    while (rs.error_code == '0') & rs.next():
        row = rs.get_row_data()
        code = row[1]
        name = row[2]
        if code.startswith('sh.6') or code.startswith('sh.0') or code.startswith('sz.0') or code.startswith('sz.3'):
            stocks.append({'code': code, 'name': name})
    
    bs.logout()
    
    df = pd.DataFrame(stocks)
    df.to_csv(cache_file, index=False)
    return df

def get_profit_data(stock_list):
    print("Fetching ROE data from baostock...")
    print("This will take a few minutes...\n")
    
    results = []
    total = len(stock_list)
    
    lg = bs.login()
    print(f"Logged in: {lg.error_msg}\n")
    
    for i, row in stock_list.iterrows():
        code = str(row['code'])
        name = str(row['name'])
        
        try:
            rs = bs.query_profit_data(code=code, year=2024, quarter=4)
            
            if rs.error_code != '0':
                time.sleep(0.2)
                continue
            
            while (rs.error_code == '0') & rs.next():
                data = rs.get_row_data()
                if data and len(data) > 3 and data[3]:
                    try:
                        roe = float(data[3])
                        results.append({
                            'code': code,
                            'name': name,
                            'roe': roe
                        })
                    except Exception as e:
                        print(f"Parse error {code}: {e}")
                        
        except Exception as e:
            print(f"Error {code}: {e}")
        
        if (i + 1) % 200 == 0:
            print(f"Progress: {i+1}/{total} - Got {len(results)} records")
        
        time.sleep(0.1)
    
    bs.logout()
    
    print(f"\nTotal records fetched: {len(results)}")
    
    return pd.DataFrame(results)

def calculate_rankings(profit_df):
    print("Calculating rankings...")
    
    profit_df = profit_df[profit_df['roe'].notna() & (profit_df['roe'] > 0)]
    profit_df = profit_df.sort_values('roe', ascending=False)
    profit_df['rank'] = range(1, len(profit_df) + 1)
    
    profit_df['roe'] = (profit_df['roe'] * 100).round(2)
    profit_df = profit_df.rename(columns={'roe': 'roe_percent'})
    
    return profit_df[['rank', 'code', 'name', 'roe_percent']]

def main():
    if not is_workday():
        print("Today is not a workday. Skipping...")
        return
    
    today = datetime.now().strftime('%Y%m%d')
    report_dir = os.path.join(os.path.dirname(__file__), 'report', 'PB-ROE')
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  ROE Ranking Report for A-Share Main Board")
    print(f"  Date: {today}")
    print(f"{'='*60}\n")
    
    stock_list = get_stock_list()
    print(f"Total A-share main board stocks: {len(stock_list)}")
    
    # Use all stocks
    profit_df = get_profit_data(stock_list)
    print(f"Fetched {len(profit_df)} records")
    
    if profit_df.empty:
        print("Failed to get financial data.")
        return
    
    result_df = calculate_rankings(profit_df)
    
    csv_path = os.path.join(report_dir, f'PB-ROE_{today}.csv')
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nReport saved to: {csv_path}")
    
    print(f"\n{'='*60}")
    print("Top 30 Stocks by ROE (Higher is Better)")
    print(f"{'='*60}")
    print(result_df.head(30).to_string(index=False))

if __name__ == '__main__':
    main()
