#!/usr/bin/env python3
"""
Test database query directly
"""

import sys
import os
sys.path.append('./backend')
from datetime import datetime, timedelta

from backend.database import get_db, ScrapedJob
from sqlalchemy import or_

# Test database query
db = next(get_db())

print("🔍 Testing database query...")

# Base query
query = db.query(ScrapedJob).filter(ScrapedJob.is_active == True)
print(f"Total active jobs: {query.count()}")

# Add search term filter
search_term = "engineer"
if search_term:
    search_filter = or_(
        ScrapedJob.title.ilike(f"%{search_term}%"),
        ScrapedJob.description.ilike(f"%{search_term}%"),
        ScrapedJob.company.ilike(f"%{search_term}%")
    )
    query = query.filter(search_filter)
    
print(f"Jobs matching '{search_term}': {query.count()}")

# Add date filter
days_old = 416
cutoff_date = datetime.now() - timedelta(days=days_old)
print(f"Cutoff date: {cutoff_date}")

date_query = query.filter(
    or_(
        ScrapedJob.date_posted >= cutoff_date,
        ScrapedJob.date_posted.is_(None)  # Include jobs without date_posted
    )
)

print(f"Jobs after date filter: {date_query.count()}")

# Add location filter
locations = ["USA"]
if locations:
    location_filter = or_(*[
        ScrapedJob.location.ilike(f"%{location}%") 
        for location in locations
    ])
    date_query = date_query.filter(location_filter)

print(f"Jobs after location filter: {date_query.count()}")

# Get first few results
results = date_query.limit(3).all()
for job in results:
    print(f"- {job.title} at {job.company} in {job.location}")

db.close()