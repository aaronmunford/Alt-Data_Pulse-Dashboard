"""
Fetches consensus and historical estimates from LSEG/Refinitiv Workspace.
Writes the raw output to data/lseg_raw_estimates.csv for inspection.
"""

import refinitiv.data as rd
import pandas as pd
from datetime import datetime

# 1. Connect to your running LSEG Desktop App
try:
    rd.open_session()
    print("✅ Connected to LSEG Workspace")
except Exception as e:
    print(f"❌ Could not connect to LSEG: {e}")
    print("Make sure the LSEG Workspace desktop app is running!")
    exit()

# 2. Define Tickers and Fields
# SBUX.O = Starbucks (NASDAQ), MCD = McDonald's, CMG = Chipotle
tickers = ["SBUX.O", "MCD", "CMG"]

# TR.RevenueActValue = Actual Revenue (History)
# TR.RevenueMean = Wall St. Consensus Mean
# TR.RevenueSmartEst = Wall St. SmartEstimate (more accurate, weights top analysts)
fields = [
    "TR.RevenueActValue",       # Historical Actuals
    "TR.RevenueMean",           # Current Consensus
    "TR.RevenueSmartEst",       # Smart Consensus
    "TR.EPSMean",               # EPS Consensus
    "TR.OriginalAnnouncementDate" # When the earnings come out
]

# 3. Fetch the Data
print(f"Fetching estimates for: {tickers}...")
df = rd.get_data(
    universe=tickers,
    fields=fields,
    parameters={
        'SDate': '0',       # Current Period
        'EDate': '-8',      # Go back 8 quarters (2 years)
        'Frq': 'FQ',        # Fiscal Quarterly
        'Curn': 'USD'
    }
)

# 4. Clean and Rename for your Pipeline
# LSEG returns a wide format; we want a clean lookup table
print("\nRaw Data Sample:")
print(df.head())

# Save raw output so we can inspect column names exactly
df.to_csv("data/lseg_raw_estimates.csv", index=False)
print("Saved to data/lseg_raw_estimates.csv")
