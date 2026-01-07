"""
Fetches clean Advan traffic data from BigQuery.
Writes data/clean_traffic_daily.csv for the predictor and dashboard.
"""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in {ENV_PATH} or your shell environment.")
    return value

PROJECT_ID = require_env("GCP_PROJECT_ID")
DATASET_ID = require_env("BQ_DATASET_ID")

# Query new Advan table
query = f"""
SELECT
    brand,
    date,
    total_visits,
    store_count
FROM `{PROJECT_ID}.{DATASET_ID}.clean_traffic_daily`
ORDER BY brand, date
"""

print("Fetching Advan traffic data from BigQuery...")

# Download directly to CSV
df = pd.read_gbq(query, project_id=PROJECT_ID)
df.to_csv("data/clean_traffic_daily.csv", index=False)
print(f"Saved {len(df)} rows to data/clean_traffic_daily.csv")
