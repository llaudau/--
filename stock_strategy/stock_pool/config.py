"""Stock pool configuration — filter thresholds."""

MIN_TRADING_DAYS = 120        # require at least 120 non-suspended trading days
MIN_AVG_DAILY_AMOUNT = 5e6    # 5M CNY average daily turnover (last 60 days)
IPO_BUFFER_DAYS = 60          # exclude first 60 trading days after IPO
MAX_SUSPENSION_RATIO = 0.10   # >10% suspended days = exclude
