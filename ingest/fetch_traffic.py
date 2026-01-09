"""
Fetches Advan traffic data from BigQuery and writes data/clean_traffic_daily.csv.

If Chipotle is missing from the clean table, we attempt to rebuild the dataset
directly from the raw Advan table using a QSR brand allowlist.
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

CLEAN_TABLE = f"{PROJECT_ID}.{DATASET_ID}.clean_traffic_daily"
RAW_TABLE = f"{PROJECT_ID}.{DATASET_ID}.raw_advan_weekly"

QSR_BRAND_PATTERNS = [
    "STARBUCKS",
    "MCDONALD'S",
    "CHIPOTLE",
    "CHIPOTLE MEXICAN GRILL",
    "DOMINO'S",
    "DOMINO'S PIZZA",
    "DUNKIN'",
    "DUNKIN' DONUTS",
    "TACO BELL",
    "WENDY'S",
    "BURGER KING",
    "SUBWAY",
    "CHICK-FIL-A",
]

RAW_BRAND_COL_ENV = "ADVAN_RAW_BRAND_COLUMN"
RAW_DATE_COL_ENV = "ADVAN_RAW_DATE_COLUMN"
RAW_VISITS_COL_ENV = "ADVAN_RAW_VISITS_COLUMN"
RAW_STORE_COL_ENV = "ADVAN_RAW_STORE_COLUMN"
RAW_LOCATION_ID_COL_ENV = "ADVAN_RAW_LOCATION_ID_COLUMN"
RAW_JSON_COL_ENV = "ADVAN_RAW_JSON_COLUMN"

def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    lower_cols = [col.lower() for col in columns]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col, col_lower in zip(columns, lower_cols):
            if candidate_lower in col_lower:
                return col
    return None

def _find_payload_column(columns: list[str]) -> str | None:
    for name in columns:
        lower = name.lower()
        if lower in {"json", "payload", "raw_json", "data"}:
            return name
        if lower.startswith("string_field_"):
            return name
    return None

def _fetch_clean_table() -> pd.DataFrame:
    query = f"""
    SELECT
        brand,
        date,
        total_visits,
        store_count
    FROM `{CLEAN_TABLE}`
    ORDER BY brand, date
    """
    return pd.read_gbq(query, project_id=PROJECT_ID)

def _get_raw_table_columns() -> list[str]:
    query = f"""
    SELECT column_name
    FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'raw_advan_weekly'
    """
    schema_df = pd.read_gbq(query, project_id=PROJECT_ID)
    return schema_df["column_name"].tolist()

def _build_brand_filter(brand_expr: str) -> str:
    """
    Build SQL WHERE clause for QSR brand filtering.
    Uses exact match OR prefix match (for location variants like "STARBUCKS #1234").
    Avoids false positives from substring matching.
    """
    conditions = []
    for pattern in QSR_BRAND_PATTERNS:
        # Escape single quotes for SQL by doubling them
        escaped_pattern = pattern.replace("'", "''")
        # Exact match OR starts with pattern followed by space
        conditions.append(f"({brand_expr} = '{escaped_pattern}' OR {brand_expr} LIKE '{escaped_pattern} %')")
    return "(" + " OR ".join(conditions) + ")"

def _json_value_expr(payload_col: str, keys: list[str]) -> str:
    parts = [f"JSON_VALUE({payload_col}, '$.{key}')" for key in keys]
    return "COALESCE(" + ", ".join(parts) + ")"

def _json_date_expr(payload_col: str, keys: list[str]) -> str:
    parts = [
        f"DATE(SAFE_CAST(JSON_VALUE({payload_col}, '$.{key}') AS TIMESTAMP))"
        for key in keys
    ]
    return "COALESCE(" + ", ".join(parts) + ")"

def _build_raw_query_from_json(payload_col: str) -> str:
    brand_keys = [
        "brands",
        "brand",
        "brand_name",
        "chain_name",
        "merchant_name",
        "company_name",
        "parent_company",
        "location_name",
    ]
    date_keys = [
        "date",
        "date_range_start",
        "week_start",
        "start_date",
        "period_start",
        "week_start_date",
    ]
    visits_keys = [
        "total_visits",
        "raw_visit_counts",
        "raw_visits",
        "visit_count",
        "visits",
        "traffic",
        "total_traffic",
    ]
    store_count_keys = [
        "store_count",
        "location_count",
        "num_locations",
        "store_cnt",
    ]
    location_keys = [
        "location_id",
        "store_id",
        "place_id",
        "safegraph_place_id",
        "placekey",
        "poi_id",
    ]

    brand_expr = f"UPPER({_json_value_expr(payload_col, brand_keys)})"
    date_expr = _json_date_expr(payload_col, date_keys)
    visits_expr = f"SAFE_CAST({_json_value_expr(payload_col, visits_keys)} AS FLOAT64)"
    store_count_expr = f"SAFE_CAST({_json_value_expr(payload_col, store_count_keys)} AS INT64)"
    location_expr = _json_value_expr(payload_col, location_keys)

    store_expr = (
        f"COALESCE(MAX({store_count_expr}), COUNT(DISTINCT {location_expr})) as store_count"
    )
    brand_filter = _build_brand_filter(brand_expr)

    return f"""
    SELECT
        {brand_expr} as brand,
        {date_expr} as date,
        SUM({visits_expr}) as total_visits,
        {store_expr}
    FROM `{RAW_TABLE}`
    WHERE {brand_filter}
      AND {brand_expr} IS NOT NULL
      AND {date_expr} IS NOT NULL
      AND {visits_expr} IS NOT NULL
    GROUP BY brand, date
    ORDER BY brand, date
    """

def _build_raw_query(columns: list[str]) -> str:
    json_override = os.getenv(RAW_JSON_COL_ENV)
    if json_override:
        if json_override not in columns:
            raise RuntimeError(
                f"Configured JSON column '{json_override}' not found in raw_advan_weekly columns: {columns}"
            )
        return _build_raw_query_from_json(json_override)

    brand_override = os.getenv(RAW_BRAND_COL_ENV)
    date_override = os.getenv(RAW_DATE_COL_ENV)
    visits_override = os.getenv(RAW_VISITS_COL_ENV)
    store_override = os.getenv(RAW_STORE_COL_ENV)
    location_override = os.getenv(RAW_LOCATION_ID_COL_ENV)

    for override in [brand_override, date_override, visits_override, store_override, location_override]:
        if override and override not in columns:
            raise RuntimeError(
                f"Configured column '{override}' not found in raw_advan_weekly columns: {columns}"
            )

    brand_col = brand_override or _find_column(
        columns,
        [
            "brands",
            "brand",
            "brand_name",
            "merchant_name",
            "chain_name",
            "company_name",
            "parent_company",
        ],
    )
    location_name_col = _find_column(columns, ["location_name"])
    date_col = date_override or _find_column(
        columns,
        ["date_range_start", "week_start", "start_date", "date", "period_start", "week_start_date"],
    )
    visits_col = visits_override or _find_column(
        columns,
        ["total_visits", "raw_visit_counts", "raw_visits", "visit_count", "visits", "traffic"],
    )
    store_col = store_override or _find_column(
        columns,
        ["store_count", "location_count", "num_locations", "store_cnt"],
    )
    location_id_col = location_override or _find_column(
        columns,
        ["store_id", "location_id", "place_id", "safegraph_place_id", "placekey"],
    )

    if not brand_col or not date_col or not visits_col:
        payload_col = _find_payload_column(columns)
        if payload_col:
            return _build_raw_query_from_json(payload_col)
        raise RuntimeError(
            "raw_advan_weekly is missing expected columns. "
            f"Columns found: {columns}"
        )

    # Build store count expression (fallback to distinct location id if needed).
    if store_col:
        store_expr = f"MAX({store_col}) as store_count"
    elif location_id_col:
        store_expr = f"COUNT(DISTINCT {location_id_col}) as store_count"
    else:
        store_expr = "NULL as store_count"

    json_brand_expr = f"NULLIF(NULLIF(TO_JSON_STRING({brand_col}), 'null'), '[]')"
    if location_name_col and brand_col and location_name_col != brand_col:
        brand_expr = f"UPPER(COALESCE({location_name_col}, {json_brand_expr}))"
    else:
        brand_expr = f"UPPER({json_brand_expr})"
    brand_filter = _build_brand_filter(brand_expr)

    return f"""
    SELECT
        {brand_expr} as brand,
        DATE({date_col}) as date,
        SUM({visits_col}) as total_visits,
        {store_expr}
    FROM `{RAW_TABLE}`
    WHERE {brand_filter}
      AND {brand_expr} IS NOT NULL
      AND DATE({date_col}) IS NOT NULL
      AND {visits_col} IS NOT NULL
    GROUP BY brand, date
    ORDER BY brand, date
    """

print("Fetching Advan traffic data from BigQuery...")
df = _fetch_clean_table()

print(f"Fetched {len(df)} rows from clean_traffic_daily")
print(f"Brands: {sorted(df['brand'].unique().tolist())}")

output_path = Path(__file__).resolve().parents[1] / "data" / "clean_traffic_daily.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
