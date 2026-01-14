"""
App Engagement Data Fetcher
============================
Downloads Daily App Engagement data from Dewey (Similarweb) and processes it
for QSR brand analysis. Outputs clean CSV for use in the revenue predictor.

Metrics available:
- Daily installs
- Sessions per user
- Active minutes per user  
- Daily Active Users (DAU)

Note: Android apps only. iOS data not available through Similarweb/Dewey.
"""

import os
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import deweydatapy as ddp
from dotenv import load_dotenv

# Load environment
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in {ENV_PATH} or your shell environment.")
    return value


# Configuration
API_KEY = require_env("DEWEY_API_KEY")
TEMP_DIR = "temp_download"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Dewey Similarweb Daily App Engagement product URL
SIMILARWEB_APP_URL = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_rs7gf69zok6hzikhe"

# QSR App Package Names (Android)
# These are the package names for QSR apps on Google Play Store
APP_TO_BRAND = {
    "com.starbucks.mobilecard": "STARBUCKS",
    "com.mcdonalds.app": "MCDONALD'S",
    "com.mcdonalds.mcdordering": "MCDONALD'S",  # Alternate package
    "com.chipotle.ordering": "CHIPOTLE",
    "com.dominospizza": "DOMINO'S",
    "com.dominospizza.usa": "DOMINO'S",  # Alternate package
    "com.dunkinbrands.otgo": "DUNKIN",
    "com.yum.tacobell": "TACO BELL",
    "com.wendys.nutritiontool": "WENDY'S",
    "com.emn8.mobilem8.nativeapp.bk": "BURGER KING",
    "com.subway.mobile.subwayapp03": "SUBWAY",
}

# Reverse mapping for lookups
BRAND_TO_APPS = {}
for app, brand in APP_TO_BRAND.items():
    if brand not in BRAND_TO_APPS:
        BRAND_TO_APPS[brand] = []
    BRAND_TO_APPS[brand].append(app)


def get_available_app_products():
    """
    List available Similarweb products in Dewey to help identify the correct URL.
    Run this if unsure of the product path.
    """
    print("Fetching available Dewey products...")
    # This would require browsing Dewey's API or web interface
    # The exact product path depends on your subscription
    print("Check your Dewey dashboard for the Similarweb Daily App Engagement product URL")
    print("Expected format: https://api.deweydata.io/api/v1/external/data/prj_XXXXX__fldr_XXXXX")


def ingest_app_data(
    start_date: str = "2023-01-01",
    product_url: str = None,
) -> pd.DataFrame:
    """
    Download and process Similarweb Daily App Engagement data from Dewey.
    
    Args:
        start_date: Start date for data download (YYYY-MM-DD)
        product_url: Override the default Dewey product URL
        
    Returns:
        DataFrame with processed app engagement data
    """
    url = product_url or SIMILARWEB_APP_URL
    
    print(f"\n{'='*60}")
    print("SIMILARWEB APP ENGAGEMENT DATA INGESTION")
    print(f"{'='*60}")
    print(f"Started: {datetime.now()}")
    print(f"Product URL: {url}")
    print(f"Start Date: {start_date}")
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Get file list from Dewey
    print("\nFetching file list from Dewey...")
    try:
        files_df = ddp.get_file_list(
            apikey=API_KEY,
            product_path=url,
            start_date=start_date,
            print_info=True
        )
    except Exception as e:
        print(f"Error fetching file list: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your Dewey API key is correct")
        print("2. Check that you have access to Similarweb Daily App Engagement")
        print("3. Update SIMILARWEB_APP_URL with the correct product path from Dewey")
        return pd.DataFrame()
    
    if files_df is None or files_df.empty:
        print("No files found. Check your Dewey subscription and product URL.")
        return pd.DataFrame()
    
    total_files = len(files_df)
    print(f"Found {total_files} files to process.")
    
    # Download files in batches
    BATCH_SIZE = 10
    all_dfs = []
    
    for start_idx in range(0, total_files, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_files)
        batch_df = files_df.iloc[start_idx:end_idx]
        
        print(f"\nDownloading batch {start_idx+1}-{end_idx} of {total_files}...")
        ddp.download_files(batch_df, TEMP_DIR)
        
        # Process downloaded files
        for file_path in glob.glob(f"{TEMP_DIR}/*"):
            try:
                df = pd.read_csv(
                    file_path,
                    compression='gzip' if file_path.endswith('.gz') else None
                )
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            finally:
                os.remove(file_path)  # Clean up
    
    if not all_dfs:
        print("No data extracted from files.")
        return pd.DataFrame()
    
    # Combine all data
    raw_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nLoaded {len(raw_df):,} total rows")
    
    # Process and filter to QSR apps
    df = process_app_data(raw_df)
    
    # Save to CSV
    output_path = DATA_DIR / "clean_app_daily.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Brands: {df['brand'].nunique()}")
    
    return df


def process_app_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Similarweb app data into clean format.
    
    Actual Similarweb column names from Dewey:
    - APP (app package name)
    - APPS_DAILY_ACTIVE_USERS_ANDROID (DAU)
    - APPS_DOWNLOADS (daily downloads)
    - APPS_UNIQUE_INSTALLS (unique installs)
    - APPS_SESSIONS_PER_USER
    - APPS_AVG_TOTAL_USAGE_TIME_PER_USER
    - APPS_AVG_USAGE_TIME_PER_SESSION
    - COUNTRY
    - DATE
    """
    df = raw_df.copy()
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Map Similarweb column names to our schema
    column_mapping = {
        'app': 'app_package',
        'apps_daily_active_users_android': 'dau',
        'apps_downloads': 'downloads',
        'apps_unique_installs': 'installs',
        'apps_sessions_per_user': 'sessions_per_user',
        'apps_avg_total_usage_time_per_user': 'active_minutes_per_user',
        'apps_avg_usage_time_per_session': 'avg_session_time',
        'country': 'country',
        'date': 'date',
    }
    df = df.rename(columns=column_mapping)
    
    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Filter to US data only (or worldwide 'WW' if US not available)
    if 'country' in df.columns:
        # Prefer US, fallback to WW (worldwide)
        if 'US' in df['country'].values:
            df = df[df['country'] == 'US']
            print("Filtered to US market data")
        elif 'WW' in df['country'].values:
            df = df[df['country'] == 'WW']
            print("Using worldwide (WW) data")
    
    # Filter to QSR apps only
    if 'app_package' in df.columns:
        qsr_apps = list(APP_TO_BRAND.keys())
        df = df[df['app_package'].isin(qsr_apps)]
        df['brand'] = df['app_package'].map(APP_TO_BRAND)
    else:
        print("Warning: Could not find app package column. Available columns:", df.columns.tolist())
        return pd.DataFrame()
    
    print(f"Filtered to {len(df):,} QSR app rows")
    
    # Select and rename final columns
    final_columns = {
        'date': 'date',
        'brand': 'brand',
        'app_package': 'app_package',
    }
    
    # Add metrics if available
    metric_cols = ['installs', 'dau', 'sessions_per_user', 'active_minutes_per_user']
    for col in metric_cols:
        if col in df.columns:
            final_columns[col] = col
    
    df = df[list(final_columns.keys())].rename(columns=final_columns)
    
    # Aggregate by brand and date (in case of multiple app packages per brand)
    agg_dict = {}
    if 'installs' in df.columns:
        agg_dict['installs'] = 'sum'
    if 'dau' in df.columns:
        agg_dict['dau'] = 'sum'
    if 'sessions_per_user' in df.columns:
        agg_dict['sessions_per_user'] = 'mean'
    if 'active_minutes_per_user' in df.columns:
        agg_dict['active_minutes_per_user'] = 'mean'
    
    if agg_dict:
        df = df.groupby(['date', 'brand']).agg(agg_dict).reset_index()
    
    # Sort by date and brand
    df = df.sort_values(['brand', 'date']).reset_index(drop=True)
    
    # Add rolling averages for smoothing
    if 'dau' in df.columns:
        df['dau_7d_avg'] = df.groupby('brand')['dau'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
    if 'installs' in df.columns:
        df['installs_7d_avg'] = df.groupby('brand')['installs'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
    
    return df


def load_app_data(path: str = None) -> pd.DataFrame:
    """
    Load processed app data from CSV.
    
    Args:
        path: Optional path to CSV file. Defaults to data/clean_app_daily.csv
        
    Returns:
        DataFrame with app engagement data
    """
    if path is None:
        path = DATA_DIR / "clean_app_daily.csv"
    
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print(f"App data not found at {path}. Run ingest_app_data() first.")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Similarweb App Engagement data from Dewey")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date for download (YYYY-MM-DD)")
    parser.add_argument("--product-url", help="Override the Dewey product URL")
    parser.add_argument("--list-products", action="store_true", help="Show help for finding product URL")
    
    args = parser.parse_args()
    
    if args.list_products:
        get_available_app_products()
    else:
        ingest_app_data(
            start_date=args.start_date,
            product_url=args.product_url,
        )
