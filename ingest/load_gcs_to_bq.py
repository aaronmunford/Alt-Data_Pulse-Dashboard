"""
Loads raw Consumer Edge and Advan files from GCS into BigQuery tables.
"""

import os
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv

# Confgiuration
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in {ENV_PATH} or your shell environment.")
    return value

PROJECT_ID = require_env("GCP_PROJECT_ID")
DATASET_ID = require_env("BQ_DATASET_ID")
BUCKET_NAME = require_env("GCS_BUCKET_NAME")

client = bigquery.Client(project=PROJECT_ID)

def load_gcs_to_table(table_name, gcs_uri, schema=None):
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        source_format = bigquery.SourceFormat.CSV,
        skip_leading_rows=1, #Skip header
        autodetect=True, # Let BigQuery guess the types
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE # overwrite table on each run
    )
    
    print(f"Loading {gcs_uri} into {table_ref}...")
    load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    load_job.result() # Waits for the job to complete
    
    # check results
    table = client.get_table(table_ref)
    print(f"Loaded {table.num_rows} rows into {table_ref}.")
    
# 1. load ConsumerEdge
load_gcs_to_table("raw_consumer_edge_daily", f"gs://{BUCKET_NAME}/raw_consumer_edge/*.csv")

# 2. load Advan (We'll load it into a staging table first because of the JSON)
# Note: for the JSON array, we might need a specific schema or just load as string first
load_gcs_to_table("raw_advan_weekly", f"gs://{BUCKET_NAME}/raw_advan/*.json")
