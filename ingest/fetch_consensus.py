"""
Fetches consensus and historical estimates from LSEG/Refinitiv Workspace.
Writes the raw output to data/lseg_raw_estimates.csv for inspection.
"""

"""
Refresh the LSEG consensus cache on-demand.

This script is meant for manual runs or cron jobs to keep
data/consensus_cache.json fresh without blocking the dashboard.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path so imports work when run from ingest/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.lseg_client import LSEGConsensusClient
from ingest.ticker_mapping import supported_tickers


def main() -> None:
    client = LSEGConsensusClient()

    # Pull consensus for all supported tickers and update the cache.
    tickers = supported_tickers()
    if not tickers:
        print("No tickers configured for LSEG consensus.")
        return

    results = client.refresh_cache(tickers=tickers)
    print(f"Refreshed consensus for {len(results)} tickers.")


if __name__ == "__main__":
    main()
