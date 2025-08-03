#!/usr/bin/env python3
"""
Test with correct database path
"""

import sys
import os

# Change to backend directory to match database path
os.chdir('./backend')
sys.path.append('.')

from database import get_db, ScrapedJob
from job_scraper import job_scraper

# Test database query
db = next(get_db())

print("🔍 Testing with correct database path...")
print(f"Current directory: {os.getcwd()}")
print(f"Database path from DATABASE_URL: {os.environ.get('DATABASE_URL', 'sqlite:///../jobsearch.db')}")

# Test query
query = db.query(ScrapedJob)
total_count = query.count()
print(f"Total jobs found: {total_count}")

if total_count > 0:
    # Test search function
    print("\nTesting search_local_jobs function...")
    jobs, count = job_scraper.search_local_jobs(
        db=db,
        search_term="engineer",
        company_names=None,
        locations=["USA"],
        job_types=None,
        is_remote=None,
        min_salary=None,
        max_salary=None,
        max_experience_years=None,
        sites=["indeed"],
        days_old=500,  # Use a larger time window
        limit=3,
        offset=0
    )
    print(f"Search results: {len(jobs)} jobs found (total: {count})")
    
    for i, job in enumerate(jobs):
        print(f"{i+1}. {job.title} at {job.company}")

db.close()