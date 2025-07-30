#!/usr/bin/env python3

import sys
import os
import asyncio
import pandas as pd

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from jobspy import scrape_jobs
    print("✅ jobspy import successful")
except ImportError as e:
    print(f"❌ jobspy import failed: {e}")
    sys.exit(1)

async def test_search():
    """Test basic job search functionality"""
    print("🔍 Testing job search...")
    
    try:
        # Test basic search parameters
        search_params = {
            "site_name": ["indeed"],
            "search_term": "python developer",
            "location": "USA",
            "distance": 50,
            "results_wanted": 5,
            "hours_old": 168,
            "country_indeed": "USA",
            "verbose": 2
        }
        
        print(f"Search parameters: {search_params}")
        
        # Call jobspy directly
        jobs_df = scrape_jobs(**search_params)
        
        if jobs_df is not None and not jobs_df.empty:
            print(f"✅ Search successful! Found {len(jobs_df)} jobs")
            print(f"Columns: {list(jobs_df.columns)}")
            print("\nFirst job:")
            print(jobs_df.iloc[0][['title', 'company', 'location']].to_dict())
            return True
        else:
            print(f"❌ Search returned no results. Type: {type(jobs_df)}")
            if jobs_df is not None:
                print(f"DataFrame shape: {jobs_df.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Search failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_search())
    sys.exit(0 if result else 1)