#!/usr/bin/env python3
"""
Test boolean query issue
"""

import sys
import os
sys.path.append('./backend')

from backend.database import get_db, ScrapedJob

# Test database query
db = next(get_db())

print("🔍 Testing boolean query...")

# Test different ways to query is_active
query1 = db.query(ScrapedJob).filter(ScrapedJob.is_active == True)
print(f"ScrapedJob.is_active == True: {query1.count()}")

query2 = db.query(ScrapedJob).filter(ScrapedJob.is_active.is_(True))
print(f"ScrapedJob.is_active.is_(True): {query2.count()}")

query3 = db.query(ScrapedJob).filter(ScrapedJob.is_active == 1)
print(f"ScrapedJob.is_active == 1: {query3.count()}")

query4 = db.query(ScrapedJob)
print(f"All jobs (no filter): {query4.count()}")

# Check some raw data
print("\nSample is_active values:")
sample_jobs = db.query(ScrapedJob.id, ScrapedJob.is_active, ScrapedJob.title).limit(5).all()
for job in sample_jobs:
    print(f"ID: {job.id}, is_active: {job.is_active} (type: {type(job.is_active)}), title: {job.title[:50]}")

db.close()