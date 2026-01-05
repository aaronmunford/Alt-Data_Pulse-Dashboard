"""
Quantitative Signal Platform - Data Pipeline
=============================================
This pipeline handles the complete ETL process:
1. Ingest: Download data from Dewey (Consumer Edge + Advan)
2. Transform: Clean, harmonize fiscal quarters, aggregate
3. Load: Push to BigQuery for dashboard consumption
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import storage, bigquery
import deweydatapy as ddp

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ID = "altdatapulsedashboard"
DATASET_ID = "alternative_data"
BUCKET_NAME = "raw-data-2025"
TEMP_DIR = "temp_download"

# Dewey API
API_KEY = os.environ.get("DEWEY_API_KEY", "akv1_VK9ZTbm-gPSQAYFamgi50onevnEVA1AmKDq")
CONSUMER_EDGE_URL = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_rxm7vzotfttqn4ppf"
ADVAN_URL = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_bpyousrmfggrfubk"

# Target QSR brands to track
QSR_BRANDS = [
    "STARBUCKS", "CHIPOTLE", "MCDONALD'S", "DOMINO'S", "DUNKIN",
    "TACO BELL", "CHICK-FIL-A", "WENDY'S", "BURGER KING", "SUBWAY"
]

# Batch settings for large file processing
BATCH_SIZE = 10

# =============================================================================
# FISCAL CALENDAR (QSR companies have weird fiscal years)
# =============================================================================
FISCAL_CALENDARS = {
    "SBUX": {  # Starbucks: Fiscal year ends first Sunday of October
        "Q1": {"start_month": 10, "start_day": 1, "end_month": 12, "end_day": 31},
        "Q2": {"start_month": 1, "start_day": 1, "end_month": 3, "end_day": 31},
        "Q3": {"start_month": 4, "start_day": 1, "end_month": 6, "end_day": 30},
        "Q4": {"start_month": 7, "start_day": 1, "end_month": 9, "end_day": 30},
    },
    "CMG": {  # Chipotle: Standard calendar year
        "Q1": {"start_month": 1, "start_day": 1, "end_month": 3, "end_day": 31},
        "Q2": {"start_month": 4, "start_day": 1, "end_month": 6, "end_day": 30},
        "Q3": {"start_month": 7, "start_day": 1, "end_month": 9, "end_day": 30},
        "Q4": {"start_month": 10, "start_day": 1, "end_month": 12, "end_day": 31},
    },
    "MCD": {  # McDonald's: Standard calendar year
        "Q1": {"start_month": 1, "start_day": 1, "end_month": 3, "end_day": 31},
        "Q2": {"start_month": 4, "start_day": 1, "end_month": 6, "end_day": 30},
        "Q3": {"start_month": 7, "start_day": 1, "end_month": 9, "end_day": 30},
        "Q4": {"start_month": 10, "start_day": 1, "end_month": 12, "end_day": 31},
    },
}


def get_fiscal_quarter(date: datetime, ticker: str = "CMG") -> tuple:
    """
    Convert a calendar date to fiscal quarter for a given ticker.
    Returns (fiscal_year, fiscal_quarter) tuple.
    """
    calendar = FISCAL_CALENDARS.get(ticker, FISCAL_CALENDARS["CMG"])

    for quarter, bounds in calendar.items():
        start = datetime(date.year, bounds["start_month"], bounds["start_day"])
        end = datetime(date.year, bounds["end_month"], bounds["end_day"])

        # Handle fiscal years that span calendar years (like Starbucks)
        if bounds["start_month"] > bounds["end_month"]:
            if date.month >= bounds["start_month"]:
                end = datetime(date.year + 1, bounds["end_month"], bounds["end_day"])
            else:
                start = datetime(date.year - 1, bounds["start_month"], bounds["start_day"])

        if start <= date <= end:
            fiscal_year = date.year if quarter != "Q1" or bounds["start_month"] <= 3 else date.year + 1
            return (fiscal_year, quarter)

    # Default fallback
    q = (date.month - 1) // 3 + 1
    return (date.year, f"Q{q}")


# =============================================================================
# DATA INGESTION
# =============================================================================
class DeweyIngester:
    """Handles batch download from Dewey Data API."""

    def __init__(self):
        self.storage_client = storage.Client(project=PROJECT_ID)
        self.bucket = self.storage_client.bucket(BUCKET_NAME)
        os.makedirs(TEMP_DIR, exist_ok=True)

    def upload_to_gcs(self, local_path: str, destination: str):
        """Upload file to Google Cloud Storage."""
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(local_path)
        print(f"  Uploaded: {destination}")

    def ingest_consumer_edge(self, start_date: str = "2023-01-01"):
        """Download Consumer Edge spend data and upload to GCS."""
        print("\n=== Ingesting Consumer Edge Data ===")

        files_df = ddp.get_file_list(
            api_key=API_KEY,
            product_path=CONSUMER_EDGE_URL,
            start_date=start_date,
            print_info=True
        )

        if files_df is None or files_df.empty:
            print("No Consumer Edge files found.")
            return

        total_files = len(files_df)
        print(f"Found {total_files} files to process.")

        for start_idx in range(0, total_files, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_files)
            batch_df = files_df.iloc[start_idx:end_idx]

            print(f"\nBatch {start_idx+1}-{end_idx} of {total_files}")
            ddp.download_files(batch_df, TEMP_DIR)

            for file_path in glob.glob(f"{TEMP_DIR}/*"):
                filename = os.path.basename(file_path)
                self.upload_to_gcs(file_path, f"consumer_edge_raw/{filename}")
                os.remove(file_path)

        print("Consumer Edge ingestion complete.")

    def ingest_advan(self, start_date: str = "2024-01-01"):
        """Download Advan foot traffic data and upload to GCS."""
        print("\n=== Ingesting Advan Foot Traffic Data ===")

        files_df = ddp.get_file_list(
            api_key=API_KEY,
            product_path=ADVAN_URL,
            start_date=start_date,
            print_info=True
        )

        if files_df is None or files_df.empty:
            print("No Advan files found.")
            return

        total_files = len(files_df)
        print(f"Found {total_files} files to process.")

        for start_idx in range(0, total_files, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_files)
            batch_df = files_df.iloc[start_idx:end_idx]

            print(f"\nBatch {start_idx+1}-{end_idx} of {total_files}")
            ddp.download_files(batch_df, TEMP_DIR)

            for file_path in glob.glob(f"{TEMP_DIR}/*"):
                filename = os.path.basename(file_path)
                self.upload_to_gcs(file_path, f"advan_raw/{filename}")
                os.remove(file_path)

        print("Advan ingestion complete.")


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================
class DataTransformer:
    """Transform raw data into analytics-ready tables."""

    def __init__(self):
        self.bq_client = bigquery.Client(project=PROJECT_ID)

    def process_consumer_edge_local(self, raw_dir: str = "data/raw/consumer_edge") -> pd.DataFrame:
        """Process Consumer Edge files from local storage."""
        print("\n=== Processing Consumer Edge Data ===")

        all_files = glob.glob(f"{raw_dir}/*.csv.gz") + glob.glob(f"{raw_dir}/*.csv")

        if not all_files:
            print(f"No files found in {raw_dir}")
            return pd.DataFrame()

        dfs = []
        for f in all_files:
            try:
                df = pd.read_csv(f, compression='gzip' if f.endswith('.gz') else None)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df):,} rows from {len(all_files)} files")

        # Standardize column names
        df.columns = df.columns.str.upper()

        # Filter to QSR brands
        df['BRAND_NAME_UPPER'] = df['BRAND_NAME'].str.upper()
        qsr_pattern = '|'.join(QSR_BRANDS)
        df = df[df['BRAND_NAME_UPPER'].str.contains(qsr_pattern, na=False)]
        print(f"Filtered to {len(df):,} QSR rows")

        # Parse dates
        df['TRANS_DATE'] = pd.to_datetime(df['TRANS_DATE'], errors='coerce')
        df = df.dropna(subset=['TRANS_DATE'])

        # Calculate ticket size (spend per transaction)
        df['TICKET_SIZE'] = df['SPEND_AMOUNT'] / df['TRANS_COUNT'].replace(0, np.nan)

        # Add fiscal quarter mapping
        df['FISCAL_QUARTER'] = df['TRANS_DATE'].apply(
            lambda d: f"{get_fiscal_quarter(d)[0]}_{get_fiscal_quarter(d)[1]}"
        )

        # Aggregate to daily brand level
        daily_agg = df.groupby(['BRAND_NAME', 'TRANS_DATE']).agg({
            'SPEND_AMOUNT': 'sum',
            'TRANS_COUNT': 'sum',
            'TICKET_SIZE': 'mean'
        }).reset_index()

        daily_agg.columns = ['brand', 'date', 'spend', 'transactions', 'avg_ticket_size']
        print(f"Aggregated to {len(daily_agg):,} daily brand records")

        return daily_agg

    def create_clean_tables_bq(self):
        """Run transformation SQL in BigQuery."""
        print("\n=== Creating Clean Tables in BigQuery ===")

        # Create clean Consumer Edge table
        consumer_edge_sql = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.clean_spend_daily` AS
        SELECT
            UPPER(BRAND_NAME) as brand,
            DATE(TRANS_DATE) as date,
            SUM(SPEND_AMOUNT) as spend,
            SUM(TRANS_COUNT) as transactions,
            SAFE_DIVIDE(SUM(SPEND_AMOUNT), SUM(TRANS_COUNT)) as avg_ticket_size,
            -- YoY comparison
            LAG(SUM(SPEND_AMOUNT), 365) OVER (
                PARTITION BY UPPER(BRAND_NAME) ORDER BY DATE(TRANS_DATE)
            ) as spend_ly,
            LAG(SUM(TRANS_COUNT), 365) OVER (
                PARTITION BY UPPER(BRAND_NAME) ORDER BY DATE(TRANS_DATE)
            ) as transactions_ly
        FROM `{PROJECT_ID}.{DATASET_ID}.raw_consumer_edge`
        WHERE UPPER(BRAND_NAME) IN UNNEST(@qsr_brands)
        GROUP BY 1, 2
        ORDER BY brand, date
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("qsr_brands", "STRING", QSR_BRANDS)
            ]
        )

        try:
            self.bq_client.query(consumer_edge_sql, job_config=job_config).result()
            print("Created: clean_spend_daily")
        except Exception as e:
            print(f"Error creating clean_spend_daily: {e}")

        # Create quarterly aggregation
        quarterly_sql = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.clean_spend_quarterly` AS
        SELECT
            brand,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(QUARTER FROM date) as quarter,
            CONCAT(CAST(EXTRACT(YEAR FROM date) AS STRING), '_Q',
                   CAST(EXTRACT(QUARTER FROM date) AS STRING)) as fiscal_quarter,
            SUM(spend) as total_spend,
            SUM(transactions) as total_transactions,
            AVG(avg_ticket_size) as avg_ticket_size,
            COUNT(DISTINCT date) as days_in_quarter
        FROM `{PROJECT_ID}.{DATASET_ID}.clean_spend_daily`
        GROUP BY 1, 2, 3, 4
        ORDER BY brand, year, quarter
        """

        try:
            self.bq_client.query(quarterly_sql).result()
            print("Created: clean_spend_quarterly")
        except Exception as e:
            print(f"Error creating clean_spend_quarterly: {e}")


# =============================================================================
# LOAD TO BIGQUERY
# =============================================================================
class BigQueryLoader:
    """Load processed data to BigQuery."""

    def __init__(self):
        self.client = bigquery.Client(project=PROJECT_ID)

    def ensure_dataset_exists(self):
        """Create dataset if it doesn't exist."""
        dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
        try:
            self.client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            self.client.create_dataset(dataset)
            print(f"Created dataset: {dataset_ref}")

    def load_dataframe(self, df: pd.DataFrame, table_name: str):
        """Load a pandas DataFrame to BigQuery."""
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )

        job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()

        table = self.client.get_table(table_ref)
        print(f"Loaded {table.num_rows:,} rows to {table_ref}")

    def load_from_gcs(self, table_name: str, gcs_pattern: str, source_format: str = "CSV"):
        """Load data from GCS to BigQuery."""
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

        job_config = bigquery.LoadJobConfig(
            source_format=getattr(bigquery.SourceFormat, source_format),
            skip_leading_rows=1 if source_format == "CSV" else 0,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )

        uri = f"gs://{BUCKET_NAME}/{gcs_pattern}"
        print(f"Loading {uri} -> {table_ref}")

        job = self.client.load_table_from_uri(uri, table_ref, job_config=job_config)
        job.result()

        table = self.client.get_table(table_ref)
        print(f"Loaded {table.num_rows:,} rows")


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================
def run_full_pipeline(
    ingest_consumer_edge: bool = True,
    ingest_advan: bool = False,  # Set to True when ready for large download
    process_local: bool = True,
    start_date: str = "2023-01-01"
):
    """
    Run the complete data pipeline.

    Args:
        ingest_consumer_edge: Download Consumer Edge data from Dewey
        ingest_advan: Download Advan data from Dewey (large dataset!)
        process_local: Process files from local storage
        start_date: Start date for data download
    """
    print("=" * 60)
    print("QUANTITATIVE SIGNAL PLATFORM - DATA PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Initialize components
    loader = BigQueryLoader()
    loader.ensure_dataset_exists()

    # Step 1: Ingest from Dewey
    if ingest_consumer_edge or ingest_advan:
        ingester = DeweyIngester()

        if ingest_consumer_edge:
            ingester.ingest_consumer_edge(start_date)

        if ingest_advan:
            ingester.ingest_advan(start_date)

    # Step 2: Transform local data
    if process_local:
        transformer = DataTransformer()

        # Process Consumer Edge from local files
        df_spend = transformer.process_consumer_edge_local()

        if not df_spend.empty:
            loader.load_dataframe(df_spend, "clean_spend_daily")

    print("\n" + "=" * 60)
    print(f"Pipeline completed: {datetime.now()}")
    print("=" * 60)


def run_local_only():
    """Quick run: just process local files without downloading."""
    print("Processing local Consumer Edge data...")

    loader = BigQueryLoader()
    loader.ensure_dataset_exists()

    transformer = DataTransformer()
    df = transformer.process_consumer_edge_local()

    if not df.empty:
        # Save locally for testing
        df.to_csv("data/clean_spend_daily.csv", index=False)
        print(f"Saved to data/clean_spend_daily.csv")

        # Also load to BigQuery
        loader.load_dataframe(df, "clean_spend_daily")

        return df

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the data pipeline")
    parser.add_argument("--full", action="store_true", help="Run full pipeline with Dewey download")
    parser.add_argument("--local", action="store_true", help="Process local files only")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date for downloads")

    args = parser.parse_args()

    if args.full:
        run_full_pipeline(start_date=args.start_date)
    elif args.local:
        run_local_only()
    else:
        print("Usage:")
        print("  python pipeline.py --local    # Process local files only")
        print("  python pipeline.py --full     # Full pipeline with Dewey download")