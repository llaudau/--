"""
Generate all historical score rankings
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_model import run_scoring

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")

TRADING_DATES = [
    '2026-01-27', '2026-01-20', '2026-01-13', '2026-01-06', '2025-12-26',
    '2025-12-19', '2025-12-12', '2025-12-05', '2025-11-28', '2025-11-21',
    '2025-11-14', '2025-11-07', '2025-10-31', '2025-10-24', '2025-10-17',
    '2025-10-10', '2025-09-25', '2025-09-18', '2025-09-11', '2025-09-04'
]

if __name__ == "__main__":
    for date in TRADING_DATES:
        print(f"\n{'='*60}")
        print(f"Generating ranking for {date}")
        print(f"{'='*60}")
        run_scoring(target_date=date)
    
    print("\nDone!")
