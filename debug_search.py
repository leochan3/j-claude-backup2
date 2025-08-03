#!/usr/bin/env python3
"""
Debug the search conversion logic
"""

import sys
import os
sys.path.append('./backend')

from backend.main import convert_job_search_to_local_search, JobSearchRequest
from backend.models import ScrapedJobSearchRequest

# Test the conversion
print("🔧 Testing search request conversion...")

# Create a test request
test_request = JobSearchRequest(
    search_term="engineer",
    location="USA",
    results_wanted=5,
    site_name=["indeed"]
)

print(f"Original request: {test_request}")

# Convert it
job_titles = ["engineer"]
companies = []
locations = ["USA"]

local_request = convert_job_search_to_local_search(
    test_request, job_titles, companies, locations
)

print(f"Converted request: {local_request}")
print(f"Converted model dump: {local_request.model_dump()}")