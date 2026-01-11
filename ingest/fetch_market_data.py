"""
Fetches market proxy data used for modeling:
- Stock prices via yfinance
- US retail sales (food services) via FRED
Outputs data/market_data_stocks.csv and data/market_data_retail_sales.csv.
"""

import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import datetime


def _extract_price_table(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if "Adj Close" in level0:
            return raw["Adj Close"]
        if "Close" in level0:
            return raw["Close"]
        if "Adj Close" in level1:
            return raw.xs("Adj Close", level=1, axis=1)
        if "Close" in level1:
            return raw.xs("Close", level=1, axis=1)
    else:
        if "Adj Close" in raw.columns:
            return raw[["Adj Close"]]
        if "Close" in raw.columns:
            return raw[["Close"]]
    raise KeyError("Adj Close or Close column not found in yfinance output.")


# 1. Fetch Stock Data (SBUX, CMG, MCD, DPZ, TGT)
tickers = ["SBUX", "CMG", "MCD", "DPZ", "TGT"]
print(f"Fetching stock data for: {tickers}")
raw_stocks = yf.download(
    tickers,
    start="2023-01-01",
    end=datetime.date.today(),
    auto_adjust=False,
    group_by="column",
)
stocks = _extract_price_table(raw_stocks)

# Save to CSV
stocks.to_csv("data/market_data_stocks.csv")
print("Saved stock prices to data/market_data_stocks.csv")

# 2. Fetch Macro data (US Retail Sales - Food Services)
# Symbol: RSFSXMV (Retail Sales: Food Services and Drinking places)
# Pull extra history so YoY is available for 2023 quarters.
print("Fetching US Retail Sales data")
start = datetime.datetime(2018, 1, 1)
end = datetime.date.today()
retail_sales = web.DataReader("RSFSXMV", "fred", start, end) # fred: federal reserve economic data

# Save to CSV
retail_sales.to_csv("data/market_data_retail_sales.csv")
print("Saved retail sales data to data/market_data_retail_sales.csv")
