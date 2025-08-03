#!/usr/bin/env python3
"""
Debug search steps
"""

import sys
import os
from datetime import datetime, timedelta

# Change to backend directory to match database path
os.chdir('./backend')
sys.path.append('.')

from database import get_db, ScrapedJob
from sqlalchemy import or_

# Test database query
db = next(get_db())

print("🔍 Debugging search steps...")

# Step 1: Base query
query = db.query(ScrapedJob).filter(ScrapedJob.is_active == True)
print(f"Step 1 - Base query (is_active=True): {query.count()}")

# Check what values is_active actually has
is_active_values = db.execute("SELECT DISTINCT is_active FROM scraped_jobs").fetchall()
print(f"Distinct is_active values: {is_active_values}")

# Try without the is_active filter
query_no_active = db.query(ScrapedJob)
print(f"Without is_active filter: {query_no_active.count()}")

# Try different is_active comparisons
query_1 = db.query(ScrapedJob).filter(ScrapedJob.is_active == 1)
print(f"is_active == 1: {query_1.count()}")

query_not_false = db.query(ScrapedJob).filter(ScrapedJob.is_active != False)
print(f"is_active != False: {query_not_false.count()}")

query_not_null = db.query(ScrapedJob).filter(ScrapedJob.is_active.isnot(None))
print(f"is_active is not NULL: {query_not_null.count()}")

# If we get results, continue with other filters
if query_1.count() > 0:
    print("\nContinuing with is_active == 1...")
    query = db.query(ScrapedJob).filter(ScrapedJob.is_active == 1)
    
    # Step 2: Search term filter
    search_term = "engineer"
    if search_term:
        search_filter = or_(
            ScrapedJob.title.ilike(f"%{search_term}%"),
            ScrapedJob.description.ilike(f"%{search_term}%"),
            ScrapedJob.company.ilike(f"%{search_term}%")
        )
        query = query.filter(search_filter)
        print(f"Step 2 - With search term '{search_term}': {query.count()}")
    
    # Step 3: Date filter
    days_old = 500
    cutoff_date = datetime.now() - timedelta(days=days_old)
    print(f"Cutoff date: {cutoff_date}")
    
    query = query.filter(
        or_(
            ScrapedJob.date_posted >= cutoff_date,
            ScrapedJob.date_posted.is_(None)
        )
    )
    print(f"Step 3 - With date filter: {query.count()}")
    
    # Step 4: Location filter
    locations = ["USA"]
    if locations:
        location_filter = or_(*[
            ScrapedJob.location.ilike(f"%{location}%") 
            for location in locations
        ])
        query = query.filter(location_filter)
        print(f"Step 4 - With location filter: {query.count()}")
    
    # Get a sample result
    if query.count() > 0:
        sample = query.first()
        print(f"\nSample result:")
        print(f"  Title: {sample.title}")
        print(f"  Company: {sample.company}")
        print(f"  Location: {sample.location}")
        print(f"  is_active: {sample.is_active}")
        print(f"  date_posted: {sample.date_posted}")

db.close()