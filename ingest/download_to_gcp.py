# Download 10 files, upload them to GCP, delete them from local storage, and then repeat

import os
import deweydatapy as ddp
from google.cloud import storage
import glob
import pandas as pd
import shutil


# ---------------------------------------------------------
# GCP CONFIGURATION
# ---------------------------------------------------------
PROJECT_ID = "AltDataPulseDashboard" # <--- Paste your Project ID (not name) here
BUCKET_NAME = "raw-data-2025"  # <--- UPDATE THIS

# Dewey settings
API_KEY = "akv1_VK9ZTbm-gPSQAYFamgi50onevnEVA1AmKDq"
ADVAN_PATH = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_bpyousrmfggrfubk"

# BATCH SETTINGS (Crucial for 1TB data)
BATCH_SIZE = 10  # Files to process at once. Keeps disk usage low (~150MB).

# Setup GCP Client (It will look for your credentials)
# Run 'gcloud auth application-default login' in terminal first!
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

TEMP_DIR = "temp_download"
os.makedirs(TEMP_DIR, exist_ok=True)

def upload_to_gcs(local_path, destination_blob_name):
    """Uploads a file to the bucket."""
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{BUCKET_NAME}/{destination_blob_name}")
    
# ---------------------------------------------------------
# 1. GET FILE LIST 
# ---------------------------------------------------------
print("--- Fetching File List from Dewey (2024-Present) ---")

# This gets the metadata first so we know what to download
files_df = ddp.get_file_list(
    api_key=API_KEY,
    product_path=ADVAN_PATH,
    start_date="2024-01-01", 
    print_info=True
)

if files_df is None or files_df.empty:
    print("❌ Error: No files found! Check your API Key and Product Path.")
    exit()

total_files = len(files_df)
print(f"Found {total_files} files to download.")

# ---------------------------------------------------------
# 2. BATCH DOWNLOAD & UPLOAD LOOP
# ---------------------------------------------------------
print("\n--- Starting batch processing ---")

# Iterate through the dataframe in chunks
for start_idx in range(0, total_files, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, total_files)
    batch_df = files_df.iloc[start_idx:end_idx]

    print(f"\n--- Processing Batch {start_idx}-{end_idx} / {total_files} ---")
    
    # A. DOWNLOAD (fix: Removing 'output_dir' keyword, passing as 2nd arg)
    print(" Downloading....")
    ddp.download_files(batch_df, TEMP_DIR)


    # Upload and delete
    downloaded_files = glob.glob(f"{TEMP_DIR}/*")

    for file_path in downloaded_files:
        filename = os.path.basename(file_path)
    
        # Upload to GCS
        upload_to_gcs(file_path, f"advan_raw/{filename}")

        # Delete local copy immediately
        os.remove(file_path)

    print("Batch cleared from Local Disk.")

print("\n✅ Success! All files are now in your Google Cloud Bucket.")