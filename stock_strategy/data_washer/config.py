"""Data washer configuration — thresholds for validators."""

# OHLC: A-share has 10% daily limit (ST has 5%), but forward-adjusted prices
# can show larger moves at ex-dividend dates. Flag anything beyond this.
PRICE_SPIKE_THRESHOLD = 0.22

# Fundamental value bounds (beyond = likely data error, clip to bound)
PE_RANGE = (-10000, 100000)
PB_RANGE = (-1000, 10000)
PS_RANGE = (-10000, 100000)
PCF_RANGE = (-10000, 100000)
