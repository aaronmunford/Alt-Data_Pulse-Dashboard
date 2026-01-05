from google.cloud import bigquery

# Confgiuration
PROJECT_ID = "AltDataPulseDashboard"
DATASET_ID = "alternative_data"
BUCKET_NAME = "raw-data-2025"

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
load_gcs_to_table("raw_consumer_edge_daily", "gs://{BUCKET_NAME}/raw_consumer_edge/*.csv")

# 2. load Advan (We'll load it into a staging table first because of the JSON)
# Note: for the JSON array, we might need a specific schema or just load as string first
load_gcs_to_table("raw_advan_weekly", f"gs://{BUCKET_NAME}/raw_advan/*.json")