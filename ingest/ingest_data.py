import duckdb
import pandas as pd
import json 
import glob
import os
import deweydatapy as ddp

# Configuration
DB_PATH = 'data/warehouse.duckdb'

# Update these paths if your files are in different folders
CE_PATTERN = 'data/raw/consumer_edge/*.csv'   # Finds all ConsumerEdge CSVs
ADVAN_PATTERN = 'data/raw/advan/*.csv'        # Finds all Advan CSVs

# Setup the connection (creates the file if it doesn't exist)
con = duckdb.connect(DB_PATH)

# API Key 
apikey = "akv1_VK9ZTbm-gPSQAYFamgi50onevnEVA1AmKDq"

# product path
data_url = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_rxm7vzotfttqn4ppf"

# Setup the connection 
con = duckdb.connect('data/warehouse.duckdb')

# 1. Load ConsumerEdge
print("Loading ConsumerEdge data...")

# Load CSV into Pandas first to handle any preprocessing first
consumer_edge_df = pd.read_csv('data/consumer_edge_sample.csv')

# Ensure date is clean
consumer_edge_df['date'] = pd.to_datetime(consumer_edge_df['period'], errors='coerce')

print(consumer_edge_df.head())

