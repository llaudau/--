import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "cache")
STOCK_LIST_FILE = os.path.join(DATA_DIR, "stock_list.csv")
METADATA_FILE = os.path.join(DATA_DIR, "download_meta.json")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_daily")

MIN_DELAY = 0.05
MAX_DELAY = 0.15
MAX_WORKERS = 20


def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_download_date": None, "stocks": {}}


def save_metadata(meta):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_today_str():
    return datetime.now().strftime("%Y-%m-%d")


def get_main_board_stocks():
    try:
        # --- Shanghai (600/601/603/605) ---
        sh_df = ak.stock_info_sh_name_code()
        sh_df = sh_df.rename(columns={
            "证券代码": "code",
            "证券简称": "name"
        })

        # --- Shenzhen (000/001/002) ---
        sz_df = ak.stock_info_sz_name_code()
        sz_df = sz_df.rename(columns={
            "A股代码": "code",
            "A股简称": "name"
        })

        # Combine markets
        df = pd.concat([sh_df[["code", "name"]],
                        sz_df[["code", "name"]]],
                       ignore_index=True)

        # Clean format
        df["code"] = df["code"].astype(str).str.strip().str.zfill(6)

        # --- Keep only MAIN BOARD ---
        pattern = r"^(600|601|603|605|000|001|002)\d{3}$"
        df = df[df["code"].str.match(pattern)]

        df = df.sort_values("code").reset_index(drop=True)

        print(f"Total main board stocks: {len(df)}")
        print("Shanghai count:", (df["code"].str.startswith("6")).sum())
        print("Shenzhen count:", (df["code"].str.startswith(("000","001","002"))).sum())

        return df

    except Exception as e:
        print("Error:", e)
        return None

def download_stock_daily(symbol, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            if df is not None and not df.empty:
                df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高', 
                                       'low': '最低', 'close': '收盘', 'volume': '成交量',
                                       'amount': '成交额'})
                return df
        except Exception:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 1 + random.uniform(0, 0.5)
                time.sleep(wait_time)
    return None


def get_previous_trading_day(date_str):
    try:
        df = ak.tool_trade_date_hist_sina()
        trade_dates = list(pd.to_datetime(df["trade_date"]).sort_values())
        
        target = pd.to_datetime(date_str)
        prev_dates = [d for d in trade_dates if d < target]
        
        if len(prev_dates) > 0:
            return prev_dates[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def download_single_stock(code, start_date, end_date):
    code_str = str(code).zfill(6)
    prefix = "sh" if code_str.startswith("6") else "sz"
    full_code = prefix + code_str
    
    file_path = os.path.join(STOCK_DATA_DIR, f"{code_str}.csv")
    
    current_start = start_date
    need_download = True
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                dates = pd.to_datetime(existing_df["日期"], errors="coerce")
                latest_date = dates.max()
                if pd.notna(latest_date):
                    latest_str = latest_date.strftime("%Y-%m-%d")
                    if latest_str >= end_date:
                        return code_str, 0, "skipped"
                    current_start = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            pass
    
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    
    df = download_stock_daily(full_code, current_start, end_date)
    
    if df is not None and not df.empty:
        df["code"] = code_str
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=["日期"], keep="last")
                df = df.sort_values("日期")
            except Exception:
                pass
        
        df.to_csv(file_path, index=False, encoding="utf-8")
        return code_str, len(df), "success"
    
    return code_str, 0, "failed"


def download_new_data():
    meta = load_metadata()
    today_str = get_today_str()
    
    last_date = meta.get("last_download_date")
    
    if last_date is None:
        print("No previous download found. Starting initial download...")
        return download_all_data()
    
    if last_date == today_str:
        print(f"Data already downloaded for today: {today_str}")
        return True
    
    new_start_date = get_previous_trading_day(last_date)
    if new_start_date is None:
        print("Could not determine start date, downloading all data")
        return download_all_data()
    
    print(f"Checking for new data from {new_start_date} to {today_str}...")
    
    stocks = get_main_board_stocks()
    if stocks is None or stocks.empty:
        print("Failed to get stock list")
        return False
    
    os.makedirs(STOCK_DATA_DIR, exist_ok=True)
    
    total_stocks = len(stocks)
    stock_codes = stocks["code"].astype(str).str.zfill(6).tolist()
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_stock, code, new_start_date, today_str): code 
                   for code in stock_codes}
        
        completed = 0
        for future in as_completed(futures):
            code_str, count, status = future.result()
            completed += 1
            
            if status == "success":
                success_count += 1
            elif status == "failed":
                failed_count += 1
            else:
                skipped_count += 1
            
            percent = completed / total_stocks * 100
            if completed % 100 == 0 or completed == total_stocks:
                print(f"Progress: {completed}/{total_stocks} ({percent:.1f}%) | Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    
    meta["last_download_date"] = today_str
    save_metadata(meta)
    print(f"Completed: {success_count} success, {failed_count} failed, {skipped_count} skipped")
    
    return True


def download_all_data():
    meta = load_metadata()
    today_str = get_today_str()
    start_date_str = "20200101"
    
    stocks = get_main_board_stocks()
    if stocks is None or stocks.empty:
        print("Failed to get stock list")
        return False
    
    os.makedirs(STOCK_DATA_DIR, exist_ok=True)
    
    total_stocks = len(stocks)
    stock_codes = stocks["code"].astype(str).str.zfill(6).tolist()
    
    print(f"Starting download of {total_stocks} stocks with {MAX_WORKERS} workers...")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_stock, code, start_date_str, today_str): code 
                   for code in stock_codes}
        
        completed = 0
        for future in as_completed(futures):
            code_str, count, status = future.result()
            completed += 1
            
            if status == "success":
                success_count += 1
            elif status == "failed":
                failed_count += 1
            else:
                skipped_count += 1
            
            percent = completed / total_stocks * 100
            if completed % 50 == 0 or completed == total_stocks:
                print(f"Progress: {completed}/{total_stocks} ({percent:.1f}%) | Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    
    meta["last_download_date"] = today_str
    save_metadata(meta)
    print(f"Download completed: {success_count} success, {failed_count} failed, {skipped_count} skipped")
    return True


def update_stock_list():
    stocks = get_main_board_stocks()
    if stocks is not None and not stocks.empty:
        os.makedirs(DATA_DIR, exist_ok=True)
        stocks.to_csv(STOCK_LIST_FILE, index=False, encoding="utf-8")
        print(f"Updated stock list: {len(stocks)} stocks")
        return True
    return False


def check_and_download():
    print(f"Starting data check at {get_today_str()}")
    
    update_stock_list()
    
    download_new_data()
    
    print("Data update completed!")


if __name__ == "__main__":
    check_and_download()
