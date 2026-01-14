
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

try:
    import refinitiv.data as rd
    from ingest.ticker_mapping import brand_to_ticker
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def inspect_lseg_news():
    app_key = os.getenv("LSEG_APP_KEY")
    if not app_key:
        print("Error: LSEG_APP_KEY not found in environment.")
        return

    print(f"Attempting to connect to LSEG with key: {app_key[:5]}...")
    
    try:
        rd.open_session(app_key=app_key)
        print("✓ Connected to LSEG.")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return

    ticker = "SBUX.O"
    print(f"\nFetching headlines for {ticker}...")
    
    try:
        # Fetch raw dataframe
        response = rd.news.get_headlines(
            query=f"R:{ticker}",
            count=5
        )
        
        if response is None or response.empty:
            print("No headlines found.")
            return

        print("\n--- Columns Found ---")
        print(response.columns.tolist())
        
        print("\n--- First Row Raw Data ---")
        first_row = response.iloc[0]
        for col in response.columns:
            val = first_row[col]
            print(f"{col}: {val} (Type: {type(val)})")

        # Test Story Fetching for the first item
        story_id = first_row.get('storyId') or first_row.get('id')
        if story_id:
            print(f"\n--- Testing Story Fetch for ID: {story_id} ---")
            story = rd.news.get_story(story_id)
            if story:
                print("✓ Story content retrieved (First 100 chars):")
                print(str(story)[:100])
            else:
                print("✗ Story content returned None/Empty.")
        else:
            print("✗ No storyId found in first row.")

    except Exception as e:
        print(f"Error fetching data: {e}")
    finally:
        rd.close_session()
        print("\nSession closed.")

if __name__ == "__main__":
    inspect_lseg_news()
