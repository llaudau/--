# A-Share Mainboard Stock Data Downloader

Downloads daily OHLCV + PE/PB/PS data for all Chinese A-share mainboard stocks (~3200 stocks).

## Data Sources

| Source | What it provides | Protocol |
|--------|-----------------|----------|
| **baostock** | Daily OHLCV + PE(TTM) + PB(MRQ) + PS(TTM) + turnover | TCP socket (no HTTP proxy issues) |
| **akshare** | Stock list, market cap, industry, listing date | HTTP (finance.eastmoney.com) |

## Setup

```bash
cd data_downloader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Daily use: incremental OHLCV + PE/PB update
python download_all.py

# Download stock basics (market cap, industry) - run occasionally
python download_all.py --basics

# Both OHLCV and basics
python download_all.py --all

# Run individual downloaders directly
python download_daily_ohlcv.py
python download_basics.py
```

### First run

The first run downloads full history (from 2010-01-01) for all ~3200 mainboard stocks. This takes a few hours. The download is crash-safe -- if interrupted, rerun and it picks up where it left off.

### Daily updates

Subsequent runs only fetch new data since the last download date. Typically completes in ~10 minutes.

## Data Layout

```
data/
  ohlcv/          # one parquet per stock
    000001.parquet    # columns: date, open, high, low, close, volume, amount,
    000002.parquet    #          turnover_rate, pe_ttm, pb_mrq, ps_ttm, pcf_ttm, is_st
    ...
  basics/         # dated snapshots
    basics_20260409.parquet  # columns: code, total_shares, float_shares,
                             #          total_market_cap, float_market_cap, industry, listing_date
  meta/
    download_log.json   # tracks last download date per stock
    download.log        # execution log
```

## Stock Coverage

Mainboard stocks only (SH + SZ, ~3200 stocks):
- SZ: 000xxx, 001xxx, 002xxx, 003xxx
- SH: 600xxx, 601xxx, 603xxx, 605xxx

Excludes: ChiNext (300xxx), STAR Market (688xxx), BSE (8xxxxx/4xxxxx).

## Reading Data

```python
import pandas as pd

# Read one stock
df = pd.read_parquet("data/ohlcv/000001.parquet")
print(df.tail())

# Read all stocks into one DataFrame
from pathlib import Path
dfs = []
for f in Path("data/ohlcv").glob("*.parquet"):
    d = pd.read_parquet(f)
    d["code"] = f.stem
    dfs.append(d)
all_data = pd.concat(dfs, ignore_index=True)

# Read basics
basics = pd.read_parquet("data/basics/basics_20260409.parquet")
```

## Notes

- Forward-adjusted prices (前复权) by default
- Proxy env vars (`http_proxy`, `all_proxy` etc.) are automatically cleared since baostock uses TCP and akshare targets domestic Chinese APIs
- baostock requires no registration or API token
