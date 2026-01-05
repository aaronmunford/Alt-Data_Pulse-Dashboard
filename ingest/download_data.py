import os
from pathlib import Path
import deweypy.auth as ddp_auth
import deweypy.download as ddp_download

# Fetches the data files

# Configuration
API_KEY = os.environ.get(
    "DEWEY_API_KEY", "akv1_VK9ZTbm-gPSQAYFamgi50onevnEVA1AmKDq"
).strip()

# PRODUCT PATHS
CONSUMER_EDGE_PATH = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_rxm7vzotfttqn4ppf"
ADVAN_PATH = "https://api.deweydata.io/api/v1/external/data/prj_tceetydh__fldr_bpyousrmfggrfubk" 

ROOT_DOWNLOAD_DIR = Path("data/raw")
ROOT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

ddp_auth.set_api_key(API_KEY)
ddp_download.set_download_directory(ROOT_DOWNLOAD_DIR)

def _to_dataset_id(value: str) -> str:
    value = value.strip().rstrip("/")
    if "api.deweydata.io" in value:
        return value.split("/")[-1]
    return value


#. 1. Download ConsumerEdge data
print("Downloading ConsumerEdge data...")
# Downloads all available files to the folder
ddp_download.run_speedy_download(
    _to_dataset_id(CONSUMER_EDGE_PATH),
    folder_name="consumer_edge",
    skip_existing=True,
)
print("ConsumerEdge data download complete.")

#. 2. Download Advan data
print("Downloading Advan data...")
# Downloads all available files to the folder
ddp_download.run_speedy_download(
    _to_dataset_id(ADVAN_PATH),
    folder_name="advan",
    skip_existing=True,
)
print("Advan data download complete.")
