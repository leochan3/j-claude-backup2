#!/usr/bin/env python3
"""
JobSpy Comparison Test

This script runs the exact same parameters as your Jupyter notebook
to help debug differences between direct JobSpy calls and the web API.
"""

import pandas as pd
from jobspy import scrape_jobs
import json
from datetime import datetime

def test_direct_jobspy():
    """Test JobSpy with exact Jupyter notebook parameters"""
    print("🔍 Testing JobSpy with your exact Jupyter parameters...")
    print("=" * 60)
    
    # Exact parameters from your Jupyter notebook
    params = {
        "site_name": ["indeed"],
        "search_term": "Product Manager Uber",  # Combined term for direct comparison
        "location": "USA",
        "results_wanted": 1000,
        "hours_old": 10000,
        "country_indeed": 'USA',
    }
    
    print(f"📋 Parameters: {json.dumps(params, indent=2)}")
    print("⏳ Running JobSpy...")
    
    try:
        # Call JobSpy directly
        start_time = datetime.now()
        jobs_df = scrape_jobs(**params)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        if jobs_df is not None and not jobs_df.empty:
            print(f"✅ SUCCESS: Found {len(jobs_df)} jobs in {duration:.1f} seconds")
            print(f"📊 DataFrame columns: {list(jobs_df.columns)}")
            print(f"📈 DataFrame shape: {jobs_df.shape}")
            
            # Show first few jobs
            print("\n📋 First 3 jobs:")
            print("-" * 40)
            for i, job in jobs_df.head(3).iterrows():
                print(f"{i+1}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
                print(f"   Location: {job.get('location', 'N/A')}")
                print(f"   Site: {job.get('site', 'N/A')}")
                print()
            
            # Save results for comparison
            output_file = "jobspy_direct_results.csv"
            jobs_df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")
            
            return jobs_df
        else:
            print("❌ No jobs found!")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def test_api_call():
    """Test the API with same parameters"""
    print("\n🌐 Testing Web API with same parameters...")
    print("=" * 60)
    
    try:
        import requests
        
        api_data = {
            "site_name": ["indeed"],
            "search_term": "Product Manager",  # Separate search term
            "company_filter": "Uber",          # Company filter
            "location": "USA",
            "results_wanted": 1000,
            "hours_old": 10000,
            "country_indeed": "USA"
        }
        
        print(f"📋 API Parameters: {json.dumps(api_data, indent=2)}")
        print("⏳ Calling API...")
        
        start_time = datetime.now()
        response = requests.post(
            "http://localhost:8000/search-jobs",
            json=api_data,
            timeout=300  # 5 minute timeout for large requests
        )
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API SUCCESS: Found {result['job_count']} jobs in {duration:.1f} seconds")
            
            if result['jobs']:
                print("\n📋 First 3 API jobs:")
                print("-" * 40)
                for i, job in enumerate(result['jobs'][:3]):
                    print(f"{i+1}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
                    print(f"   Location: {job.get('location', 'N/A')}")
                    print(f"   Site: {job.get('site', 'N/A')}")
                    print()
            
            # Save API results
            with open('api_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 API results saved to: api_results.json")
            
            return result
        else:
            print(f"❌ API ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ API ERROR: {str(e)}")
        print("Make sure the backend server is running: cd backend && python main.py")
        return None

def compare_results(direct_df, api_result):
    """Compare results from direct JobSpy vs API"""
    print("\n🔍 COMPARISON ANALYSIS")
    print("=" * 60)
    
    if direct_df is None or api_result is None:
        print("❌ Cannot compare - one or both tests failed")
        return
    
    direct_count = len(direct_df)
    api_count = api_result['job_count']
    
    print(f"📊 Direct JobSpy: {direct_count} jobs")
    print(f"📊 Web API: {api_count} jobs")
    print(f"📊 Difference: {abs(direct_count - api_count)} jobs")
    
    if direct_count == api_count:
        print("✅ Same number of results - Good!")
    else:
        print(f"⚠️  Different results - investigating...")
        
        # Check if job titles match
        if api_result['jobs']:
            direct_titles = set(direct_df['title'].dropna().str.lower())
            api_titles = set(job['title'].lower() for job in api_result['jobs'] if job.get('title'))
            
            common_titles = direct_titles.intersection(api_titles)
            print(f"📋 Common job titles: {len(common_titles)}")
            print(f"📋 Only in direct: {len(direct_titles - api_titles)}")
            print(f"📋 Only in API: {len(api_titles - direct_titles)}")

def main():
    """Main test function"""
    print("🧪 JobSpy Comparison Test")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test direct JobSpy
    direct_result = test_direct_jobspy()
    
    # Test API (only if backend might be running)
    api_result = test_api_call()
    
    # Compare results
    compare_results(direct_result, api_result)
    
    print("\n🎯 TEST COMPLETE")
    print("=" * 60)
    print("💡 If results differ:")
    print("   1. Check the backend console for debug output")
    print("   2. Compare jobspy_direct_results.csv with api_results.json")
    print("   3. Verify all parameters are being passed correctly")

if __name__ == "__main__":
    main() 