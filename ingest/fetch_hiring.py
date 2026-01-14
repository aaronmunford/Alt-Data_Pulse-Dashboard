"""
Hiring Data Fetcher
====================
Downloads workforce/hiring data from WRDS Revelio Labs (primary) or Dewey WageScape (fallback).
Provides headcount, hiring velocity, and attrition metrics for QSR brands.

Data Sources:
- Primary: Revelio Labs via WRDS (requires NYU/institutional WRDS access)
- Fallback: WageScape via Dewey (job postings data)

Metrics provided:
- headcount: Total employees at month end
- inflows: New hires during month
- outflows: Separations during month
- hiring_velocity: MoM % change in inflows
- attrition_rate: outflows / headcount * 100
- net_hiring: inflows - outflows

Usage:
    # Fetch from WRDS (default)
    python ingest/fetch_hiring.py --start-date 2020-01-01

    # Use Dewey fallback
    python ingest/fetch_hiring.py --source dewey --start-date 2023-01-01

    # Upload to GCS after fetch
    python ingest/fetch_hiring.py --upload-gcs
"""

import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from dotenv import load_dotenv

# Load environment
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

# Configuration
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEMP_DIR = "temp_download"
OUTPUT_FILE = "clean_hiring_monthly.csv"

# Revelio Labs company name mapping for QSR brands
BRAND_TO_REVELIO_COMPANY: Dict[str, List[str]] = {
    "STARBUCKS": ["Starbucks", "Starbucks Corporation", "Starbucks Coffee Company"],
    "MCDONALD'S": ["McDonald's", "McDonald's Corporation", "McDonalds"],
    "CHIPOTLE": ["Chipotle", "Chipotle Mexican Grill", "Chipotle Mexican Grill Inc"],
    "DOMINO'S": ["Domino's Pizza", "Domino's Pizza Inc", "Dominos Pizza"],
    "DUNKIN": ["Dunkin' Brands", "Dunkin' Brands Group", "Inspire Brands"],
    "TACO BELL": ["Taco Bell", "Yum! Brands"],
    "WENDY'S": ["Wendy's", "The Wendy's Company", "Wendy's International"],
    "BURGER KING": ["Burger King", "Restaurant Brands International"],
    "SUBWAY": ["Subway", "Subway IP LLC", "Doctor's Associates"],
}

# Reverse mapping for Revelio company -> brand
REVELIO_TO_BRAND: Dict[str, str] = {}
for brand, companies in BRAND_TO_REVELIO_COMPANY.items():
    for company in companies:
        REVELIO_TO_BRAND[company.lower()] = brand


def get_env(name: str, required: bool = True) -> Optional[str]:
    """Get environment variable with optional requirement check."""
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Missing {name}. Set it in {ENV_PATH} or your shell environment.")
    return value


# ============================================================================
# WRDS Revelio Labs Integration (Primary Source)
# ============================================================================

def test_wrds_connection() -> bool:
    """Test WRDS connection and print available Revelio tables."""
    try:
        import wrds
        print("Connecting to WRDS...")
        db = wrds.Connection()

        print("\nConnection successful!")
        print(f"Username: {db.username}")

        # List available libraries
        print("\nSearching for Revelio Labs tables...")
        libs = db.list_libraries()
        revelio_libs = [lib for lib in libs if 'revelio' in lib.lower()]

        if revelio_libs:
            print(f"Found Revelio libraries: {revelio_libs}")
            for lib in revelio_libs:
                tables = db.list_tables(library=lib)
                print(f"\nTables in {lib}:")
                for t in tables[:20]:  # Show first 20 tables
                    print(f"  - {t}")
        else:
            print("No Revelio libraries found. Checking common names...")
            # Try common library names
            for lib_name in ['revelio', 'revelio_labs', 'revl']:
                try:
                    tables = db.list_tables(library=lib_name)
                    print(f"\nTables in {lib_name}: {tables}")
                except Exception:
                    pass

        db.close()
        return True

    except ImportError:
        print("ERROR: wrds package not installed. Run: pip install wrds")
        return False
    except Exception as e:
        print(f"ERROR connecting to WRDS: {e}")
        print("\nFirst-time setup instructions:")
        print("1. Run: python -c \"import wrds; wrds.Connection()\"")
        print("2. Enter your WRDS username (your NYU NetID)")
        print("3. Enter your WRDS password")
        print("4. Credentials will be saved to ~/.pgpass")
        return False


def fetch_from_wrds(
    start_date: str = "2020-01-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Fetch hiring data from WRDS Revelio Labs.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (default: today)

    Returns:
        DataFrame with hiring metrics by brand and month
    """
    try:
        import wrds
    except ImportError:
        print("ERROR: wrds package not installed. Run: pip install wrds")
        return pd.DataFrame()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print("REVELIO LABS HIRING DATA (via WRDS)")
    print(f"{'='*60}")
    print(f"Started: {datetime.now()}")
    print(f"Date range: {start_date} to {end_date}")

    try:
        db = wrds.Connection()
        print(f"Connected as: {db.username}")
    except Exception as e:
        print(f"WRDS connection failed: {e}")
        print("Run test_wrds_connection() for setup instructions.")
        return pd.DataFrame()

    # Build company name filter for SQL query
    all_companies = []
    for companies in BRAND_TO_REVELIO_COMPANY.values():
        all_companies.extend(companies)
    company_list = ", ".join([f"'{c}'" for c in all_companies])

    # Try different Revelio table structures
    # The exact schema depends on WRDS subscription level
    queries_to_try = [
        # Revelio Workforce Dynamics (aggregated monthly data)
        f"""
        SELECT
            company_name,
            date,
            headcount,
            inflows,
            outflows
        FROM revelio.company_monthly
        WHERE LOWER(company_name) IN ({company_list.lower()})
        AND date >= '{start_date}'
        AND date <= '{end_date}'
        ORDER BY company_name, date
        """,

        # Alternative: Revelio company headcount table
        f"""
        SELECT
            company_name,
            month_date as date,
            total_headcount as headcount,
            net_inflows as inflows,
            net_outflows as outflows
        FROM revelio_labs.headcount_monthly
        WHERE company_name ILIKE ANY (ARRAY[{company_list}])
        AND month_date >= '{start_date}'
        AND month_date <= '{end_date}'
        ORDER BY company_name, month_date
        """,

        # Alternative: revelio_workforce schema
        f"""
        SELECT
            company,
            yearmonth as date,
            employees as headcount,
            hires as inflows,
            separations as outflows
        FROM revelio_workforce.company_stats
        WHERE company ILIKE ANY (ARRAY[{company_list}])
        AND yearmonth >= '{start_date}'
        AND yearmonth <= '{end_date}'
        ORDER BY company, yearmonth
        """,
    ]

    df = pd.DataFrame()
    for i, query in enumerate(queries_to_try):
        try:
            print(f"\nTrying query variant {i+1}...")
            df = db.raw_sql(query)
            if not df.empty:
                print(f"Success! Retrieved {len(df):,} rows")
                break
        except Exception as e:
            print(f"Query {i+1} failed: {str(e)[:100]}")
            continue

    db.close()

    if df.empty:
        print("\nNo data retrieved from WRDS Revelio tables.")
        print("This could mean:")
        print("1. Your institution doesn't have Revelio Labs access")
        print("2. The table names have changed")
        print("3. The company name matching needs adjustment")
        print("\nRun with --test-connection to explore available tables.")
        return pd.DataFrame()

    # Process the data
    df = process_revelio_data(df)
    return df


def process_revelio_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Process raw Revelio data into standard schema."""
    df = raw_df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Map company names to brands
    company_col = None
    for col in ['company_name', 'company', 'name']:
        if col in df.columns:
            company_col = col
            break

    if company_col is None:
        print(f"Warning: Could not find company column. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    # Map to canonical brand names
    df['brand'] = df[company_col].str.lower().map(REVELIO_TO_BRAND)
    df = df.dropna(subset=['brand'])

    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Ensure we have the required columns
    required = ['date', 'brand', 'headcount', 'inflows', 'outflows']
    for col in required:
        if col not in df.columns:
            if col in ['inflows', 'outflows']:
                df[col] = 0  # Default to 0 if not available
            elif col == 'headcount':
                print(f"Warning: Missing {col} column")
                return pd.DataFrame()

    # Convert to numeric
    for col in ['headcount', 'inflows', 'outflows']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Aggregate by brand and month (in case of multiple company matches)
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df = df.groupby(['month', 'brand']).agg({
        'headcount': 'sum',
        'inflows': 'sum',
        'outflows': 'sum',
    }).reset_index()
    df = df.rename(columns={'month': 'date'})

    # Calculate derived metrics
    df = calculate_hiring_metrics(df)

    # Sort by brand and date
    df = df.sort_values(['brand', 'date']).reset_index(drop=True)

    print(f"\nProcessed {len(df):,} rows")
    print(f"Brands: {df['brand'].unique().tolist()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


# ============================================================================
# Dewey LinkUp Integration (Primary Source)
# ============================================================================

# LinkUp Ticker Analytics product URL (job postings aggregated by ticker)
LINKUP_TICKER_URL = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_h6fv8exa4deqzb7tu"

# Ticker to brand mapping for LinkUp Ticker Analytics
TICKER_TO_BRAND: Dict[str, str] = {
    "SBUX": "STARBUCKS",
    "MCD": "MCDONALD'S",
    "CMG": "CHIPOTLE",
    "DPZ": "DOMINO'S",
    "DNKN": "DUNKIN",
    "YUM": "TACO BELL",  # Yum! Brands owns Taco Bell
    "WEN": "WENDY'S",
    "QSR": "BURGER KING",  # Restaurant Brands International
}

# QSR tickers to filter for
QSR_TICKERS = ["SBUX", "MCD", "CMG", "DPZ", "DNKN", "YUM", "WEN", "QSR"]

# Legacy WageScape mapping (fallback)
BRAND_TO_WAGESCAPE: Dict[str, List[str]] = {
    "STARBUCKS": ["starbucks"],
    "MCDONALD'S": ["mcdonalds", "mcdonald's"],
    "CHIPOTLE": ["chipotle"],
    "DOMINO'S": ["dominos", "domino's pizza"],
    "DUNKIN": ["dunkin"],
}


def fetch_from_dewey(
    start_date: str = "2023-01-01",
    product_url: str = None,
) -> pd.DataFrame:
    """
    Fetch job postings data from Dewey LinkUp Ticker Analytics.

    Note: LinkUp provides job postings aggregated by ticker, not actual headcount.
    We use job postings as a proxy for hiring intent.

    Args:
        start_date: Start date (YYYY-MM-DD)
        product_url: Override the default Dewey product URL

    Returns:
        DataFrame with hiring metrics by brand and month
    """
    try:
        import deweydatapy as ddp
    except ImportError:
        print("ERROR: deweydatapy package not installed.")
        return pd.DataFrame()

    api_key = get_env("DEWEY_API_KEY", required=False)
    if not api_key:
        print("DEWEY_API_KEY not set. Cannot use Dewey.")
        return pd.DataFrame()

    url = product_url or LINKUP_TICKER_URL

    print(f"\n{'='*60}")
    print("LINKUP TICKER ANALYTICS (via Dewey)")
    print(f"{'='*60}")
    print(f"Started: {datetime.now()}")
    print(f"Product URL: {url}")
    print(f"Start Date: {start_date}")
    print(f"Target tickers: {QSR_TICKERS}")
    print("\nNote: Using job postings as proxy for hiring activity.")

    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Get file list from Dewey
    print("\nFetching file list from Dewey...")
    try:
        files_df = ddp.get_file_list(
            apikey=api_key,
            product_path=url,
            start_date=start_date,
            print_info=True
        )
    except Exception as e:
        print(f"Error fetching file list: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your Dewey API key is correct")
        print("2. Check that you have access to LinkUp Ticker Analytics")
        print("3. Update LINKUP_TICKER_URL with the correct product path")
        return pd.DataFrame()

    if files_df is None or files_df.empty:
        print("No LinkUp files found.")
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
                # Detect file format and read appropriately
                if file_path.endswith('.parquet') or file_path.endswith('.snappy.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.gz'):
                    df = pd.read_csv(file_path, compression='gzip')
                else:
                    df = pd.read_csv(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            finally:
                os.remove(file_path)

    if not all_dfs:
        print("No data extracted from WageScape files.")
        return pd.DataFrame()

    # Combine and process
    raw_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nLoaded {len(raw_df):,} total rows")

    # Try LinkUp format first, fall back to WageScape format
    df = process_linkup_data(raw_df)
    return df


def process_linkup_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process LinkUp Ticker Analytics data into hiring metrics.

    LinkUp Ticker Analytics provides job postings aggregated by ticker symbol.
    We map tickers to brands and aggregate metrics.
    """
    df = raw_df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    print(f"Available columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")

    # Find ticker column
    ticker_col = None
    for col in ['ticker', 'symbol', 'ticker_symbol', 'stock_ticker']:
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        print("Could not find ticker column in LinkUp data.")
        print("Falling back to company-based matching...")
        return process_wagescape_data(raw_df)

    # Clean ticker format - LinkUp uses "TICKER | EXCHANGE | COUNTRY" format
    # Extract just the ticker symbol (first part before |)
    df['ticker_clean'] = df[ticker_col].str.split('|').str[0].str.strip().str.upper()

    # Filter to QSR tickers (check both cleaned and original)
    qsr_mask = df['ticker_clean'].isin(QSR_TICKERS)

    # Also check if any QSR ticker is contained in the original ticker string
    if not qsr_mask.any():
        print("Trying partial ticker matching...")
        for ticker in QSR_TICKERS:
            qsr_mask = qsr_mask | df[ticker_col].str.contains(ticker, case=False, na=False)

    df = df[qsr_mask]

    if df.empty:
        print(f"No QSR tickers found. Looking for: {QSR_TICKERS}")
        print(f"Sample tickers in data: {raw_df[ticker_col].head(20).tolist()}")
        return pd.DataFrame()

    # Map ticker to brand using cleaned ticker
    df['brand'] = df['ticker_clean'].map(TICKER_TO_BRAND)

    # If still no brands, try matching from original ticker column
    if df['brand'].isna().all():
        def extract_brand(ticker_str):
            if pd.isna(ticker_str):
                return None
            ticker_str = ticker_str.upper()
            for qsr_ticker, brand in TICKER_TO_BRAND.items():
                if qsr_ticker in ticker_str:
                    return brand
            return None
        df['brand'] = df[ticker_col].apply(extract_brand)

    df = df.dropna(subset=['brand'])

    print(f"Found {len(df):,} rows for QSR tickers")

    # Find date column - LinkUp uses 'day' column
    date_col = None
    for col in ['day', 'date', 'as_of_date', 'observation_date', 'period_date', 'month']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print("Could not find date column in LinkUp data.")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['date'])

    # Find job postings metrics
    # LinkUp Ticker Analytics columns:
    # - unique_active_job_count: Number of unique active job listings
    # - created_job_count: New jobs posted
    # - deleted_job_count: Jobs removed/filled
    # - active_duration: Average duration jobs stay open
    job_count_col = None
    for col in ['unique_active_job_count', 'active_job_count', 'total_jobs', 'job_openings',
                'active_listings', 'job_count', 'openings', 'active_jobs', 'total_active']:
        if col in df.columns:
            job_count_col = col
            break

    new_jobs_col = None
    for col in ['created_job_count', 'new_jobs', 'jobs_added', 'new_postings', 'added', 'new_listings']:
        if col in df.columns:
            new_jobs_col = col
            break

    removed_jobs_col = None
    for col in ['deleted_job_count', 'removed_jobs', 'jobs_removed', 'closed_postings', 'removed', 'closed']:
        if col in df.columns:
            removed_jobs_col = col
            break

    # Aggregate to monthly by brand
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    agg_dict = {}
    if job_count_col:
        agg_dict[job_count_col] = 'mean'  # Average active jobs during month
    if new_jobs_col:
        agg_dict[new_jobs_col] = 'sum'  # Total new postings
    if removed_jobs_col:
        agg_dict[removed_jobs_col] = 'sum'  # Total removed/filled

    if not agg_dict:
        # If no specific columns found, count rows as job postings
        print("No specific job count columns found, counting rows as postings")
        monthly = df.groupby(['month', 'brand']).size().reset_index(name='job_postings')
    else:
        monthly = df.groupby(['month', 'brand']).agg(agg_dict).reset_index()

    monthly = monthly.rename(columns={'month': 'date'})

    # Standardize column names for output schema
    # Convert to float to avoid Decimal type issues
    if job_count_col and job_count_col in monthly.columns:
        monthly['headcount'] = pd.to_numeric(monthly[job_count_col], errors='coerce').astype(float).fillna(0)
    else:
        monthly['headcount'] = 0.0

    if new_jobs_col and new_jobs_col in monthly.columns:
        monthly['inflows'] = pd.to_numeric(monthly[new_jobs_col], errors='coerce').astype(float).fillna(0)
    elif 'job_postings' in monthly.columns:
        monthly['inflows'] = pd.to_numeric(monthly['job_postings'], errors='coerce').astype(float).fillna(0)
    else:
        monthly['inflows'] = 0.0

    if removed_jobs_col and removed_jobs_col in monthly.columns:
        monthly['outflows'] = pd.to_numeric(monthly[removed_jobs_col], errors='coerce').astype(float).fillna(0)
    else:
        monthly['outflows'] = 0.0

    # Calculate derived metrics
    monthly = calculate_hiring_metrics(monthly)

    # Add data source flag
    monthly['data_source'] = 'linkup_ticker'

    # Sort
    monthly = monthly.sort_values(['brand', 'date']).reset_index(drop=True)

    print(f"\nProcessed {len(monthly):,} rows")
    print(f"Brands: {monthly['brand'].unique().tolist()}")
    print(f"Date range: {monthly['date'].min()} to {monthly['date'].max()}")

    return monthly


def process_wagescape_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process WageScape job postings into hiring metrics (fallback).

    Note: Since WageScape has job postings (not actual headcount),
    we use postings as proxy metrics:
    - job_postings -> proxy for hiring demand
    - posting_velocity -> proxy for hiring acceleration
    """
    df = raw_df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    print(f"Available columns: {df.columns.tolist()}")

    # Map company names to brands (WageScape schema varies)
    company_col = None
    for col in ['company', 'employer', 'company_name', 'employer_name']:
        if col in df.columns:
            company_col = col
            break

    if company_col is None:
        print("Could not find company column in WageScape data.")
        return pd.DataFrame()

    # Filter to QSR brands
    qsr_patterns = []
    for brand, patterns in BRAND_TO_WAGESCAPE.items():
        qsr_patterns.extend(patterns)

    # Case-insensitive matching
    df[company_col] = df[company_col].str.lower()

    # Map to brands
    def map_to_brand(company: str) -> Optional[str]:
        if pd.isna(company):
            return None
        company = company.lower()
        for brand, patterns in BRAND_TO_WAGESCAPE.items():
            for pattern in patterns:
                if pattern in company:
                    return brand
        return None

    df['brand'] = df[company_col].apply(map_to_brand)
    df = df.dropna(subset=['brand'])

    if df.empty:
        print("No QSR brands found in WageScape data.")
        return pd.DataFrame()

    # Parse date
    date_col = None
    for col in ['date', 'posting_date', 'post_date', 'created_date']:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        print("Could not find date column in WageScape data.")
        return pd.DataFrame()

    # Aggregate job postings by brand and month
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    monthly = df.groupby(['month', 'brand']).size().reset_index(name='job_postings')
    monthly = monthly.rename(columns={'month': 'date'})

    # Create proxy metrics from job postings
    # job_postings serves as inflows proxy
    monthly['headcount'] = 0  # Not available from WageScape
    monthly['inflows'] = monthly['job_postings']  # Use postings as hiring proxy
    monthly['outflows'] = 0  # Not available

    # Calculate derived metrics
    monthly = calculate_hiring_metrics(monthly)

    # Add flag indicating this is proxy data
    monthly['data_source'] = 'wagescape_proxy'

    # Sort
    monthly = monthly.sort_values(['brand', 'date']).reset_index(drop=True)

    print(f"\nProcessed {len(monthly):,} rows")
    print(f"Brands: {monthly['brand'].unique().tolist()}")
    print(f"Date range: {monthly['date'].min()} to {monthly['date'].max()}")

    return monthly


# ============================================================================
# Common Processing Functions
# ============================================================================

def calculate_hiring_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived hiring metrics."""
    df = df.copy()

    # Convert numeric columns to float to avoid Decimal issues
    for col in ['headcount', 'inflows', 'outflows']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float).fillna(0)

    # Net hiring
    df['net_hiring'] = df['inflows'] - df['outflows']

    # Attrition rate (outflows / headcount)
    df['attrition_rate'] = 0.0
    mask = df['headcount'] > 0
    df.loc[mask, 'attrition_rate'] = (
        df.loc[mask, 'outflows'] / df.loc[mask, 'headcount'] * 100
    )

    # Hiring velocity (MoM % change in inflows)
    df['hiring_velocity'] = df.groupby('brand')['inflows'].pct_change() * 100

    # Headcount YoY % change
    df['headcount_yoy_pct'] = df.groupby('brand')['headcount'].pct_change(periods=12) * 100

    # Fill NaN values from calculations
    df['hiring_velocity'] = df['hiring_velocity'].fillna(0)
    df['headcount_yoy_pct'] = df['headcount_yoy_pct'].fillna(0)

    return df


def save_hiring_data(df: pd.DataFrame, upload_gcs: bool = False) -> str:
    """Save hiring data to CSV and optionally upload to GCS."""
    output_path = DATA_DIR / OUTPUT_FILE
    DATA_DIR.mkdir(exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total rows: {len(df):,}")

    if upload_gcs:
        upload_to_gcs(output_path)

    return str(output_path)


def upload_to_gcs(local_path: Path) -> bool:
    """Upload hiring data to Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed.")
        return False

    bucket_name = get_env("GCS_BUCKET_NAME", required=False)
    if not bucket_name:
        print("GCS_BUCKET_NAME not set. Skipping GCS upload.")
        return False

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blob_name = f"hiring_data/{OUTPUT_FILE}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))

        print(f"Uploaded to gs://{bucket_name}/{blob_name}")
        return True
    except Exception as e:
        print(f"GCS upload failed: {e}")
        return False


def load_hiring_data(path: str = None) -> pd.DataFrame:
    """
    Load processed hiring data from CSV.

    Args:
        path: Optional path to CSV file. Defaults to data/clean_hiring_monthly.csv

    Returns:
        DataFrame with hiring data
    """
    if path is None:
        path = DATA_DIR / OUTPUT_FILE

    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print(f"Hiring data not found at {path}.")
        print("Run fetch_hiring.py to download data first.")
        return pd.DataFrame()


def ingest_hiring_data(
    source: str = "dewey",
    start_date: str = "2020-01-01",
    end_date: str = None,
    upload_gcs: bool = False,
    product_url: str = None,
) -> pd.DataFrame:
    """
    Main entry point for hiring data ingestion.

    Args:
        source: Data source - "dewey" (default, LinkUp) or "wrds" (Revelio Labs)
        start_date: Start date for data fetch
        end_date: End date (default: today)
        upload_gcs: Whether to upload to GCS after fetch
        product_url: Override Dewey product URL

    Returns:
        DataFrame with hiring data
    """
    df = pd.DataFrame()

    if source == "wrds":
        df = fetch_from_wrds(start_date, end_date)

        # Fallback to Dewey if WRDS fails
        if df.empty:
            print("\n" + "="*60)
            print("WRDS fetch failed. Trying Dewey LinkUp fallback...")
            print("="*60)
            df = fetch_from_dewey(start_date, product_url)

    elif source == "dewey":
        df = fetch_from_dewey(start_date, product_url)

    else:
        print(f"Unknown source: {source}. Use 'wrds' or 'dewey'.")
        return pd.DataFrame()

    if not df.empty:
        save_hiring_data(df, upload_gcs)

    return df


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch hiring data from WRDS Revelio Labs or Dewey WageScape"
    )
    parser.add_argument(
        "--source",
        choices=["wrds", "dewey"],
        default="dewey",
        help="Data source: dewey (LinkUp Ticker Analytics, default) or wrds (Revelio Labs)"
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date for download (YYYY-MM-DD). Default: 2020-01-01"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for download (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="Upload to Google Cloud Storage after fetch"
    )
    parser.add_argument(
        "--product-url",
        help="Override Dewey product URL"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test WRDS connection and list available tables"
    )

    args = parser.parse_args()

    if args.test_connection:
        test_wrds_connection()
    else:
        ingest_hiring_data(
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            upload_gcs=args.upload_gcs,
            product_url=args.product_url,
        )
