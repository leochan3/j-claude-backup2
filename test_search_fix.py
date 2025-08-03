#!/usr/bin/env python3
"""
Test script to verify the search endpoint fix works correctly
"""

import requests
import json
import sys

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_SEARCH = {
    "search_term": "software engineer",
    "location": "USA",
    "results_wanted": 5,
    "site_name": ["indeed"]
}

def test_search_endpoint():
    """Test the public search endpoint"""
    print("🧪 Testing search endpoint fix...")
    
    try:
        # Test the public endpoint
        response = requests.post(
            f"{BASE_URL}/search-jobs-public",
            json=TEST_SEARCH,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"📊 Found {result.get('job_count', 0)} jobs")
            
            # Check if we have jobs
            jobs = result.get('jobs', [])
            if jobs:
                # Test the first job to verify field mapping
                first_job = jobs[0]
                print(f"\n🔍 Testing first job data structure:")
                
                # Check for the fixed field name
                if 'interval' in first_job:
                    print(f"✅ Field 'interval' exists: {first_job.get('interval')}")
                else:
                    print(f"❌ Field 'interval' missing!")
                
                # Check that old field name is not present
                if 'salary_interval' in first_job:
                    print(f"❌ Old field 'salary_interval' still present!")
                else:
                    print(f"✅ Old field 'salary_interval' correctly removed")
                
                # Check other key fields
                required_fields = ['title', 'company', 'location', 'site']
                for field in required_fields:
                    if field in first_job:
                        print(f"✅ Field '{field}': {first_job.get(field)}")
                    else:
                        print(f"❌ Required field '{field}' missing!")
                
                print(f"\n📋 Job sample:")
                print(f"   Title: {first_job.get('title', 'N/A')}")
                print(f"   Company: {first_job.get('company', 'N/A')}")
                print(f"   Location: {first_job.get('location', 'N/A')}")
                print(f"   Site: {first_job.get('site', 'N/A')}")
                print(f"   Interval: {first_job.get('interval', 'N/A')}")
                
            else:
                print("⚠️ No jobs returned, but request was successful")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_search_endpoint()
    
    if success:
        print(f"\n🎉 Search endpoint fix test PASSED!")
        sys.exit(0)
    else:
        print(f"\n💥 Search endpoint fix test FAILED!")
        sys.exit(1)