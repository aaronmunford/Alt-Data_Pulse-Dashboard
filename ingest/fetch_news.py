"""
Refresh the LSEG news cache on-demand.

This script is meant for manual runs or cron jobs to keep
data/news_cache.json fresh without blocking the dashboard.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path so imports work when run from ingest/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.news_client import LSEGNewsClient
from ingest.ticker_mapping import supported_tickers


def main() -> None:
    client = LSEGNewsClient()

    # Pull news headlines for all supported tickers and update the cache.
    tickers = supported_tickers()
    if not tickers:
        print("No tickers configured for LSEG news.")
        return

    print(f"Fetching news for {len(tickers)} tickers: {', '.join(tickers)}")

    results = client.refresh_cache(tickers=tickers, count=15)

    if results:
        print(f"Refreshed news for {len(results)} tickers:")
        for ticker, headlines in results.items():
            print(f"  {ticker}: {len(headlines)} headlines")
    else:
        print("No news fetched. Check LSEG connection.")
        if client.last_error:
            print(f"Last error: {client.last_error}")


if __name__ == "__main__":
    main()
