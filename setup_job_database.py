#!/usr/bin/env python3
"""
Setup Job Database Script

This script helps you get started with the local job database by:
1. Creating database tables
2. Adding popular companies to scrape
3. Running an initial scraping session
4. Showing how to search the local database
"""

import asyncio
import sys
import os
from datetime import datetime

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import create_tables, get_db, TargetCompany
from job_scraper import job_scraper
from models import BulkScrapingRequest


# Popular tech companies to scrape initially
POPULAR_COMPANIES = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Tesla",
    "Uber", "Airbnb", "Spotify", "Adobe", "Salesforce", "Twitter", "LinkedIn",
    "Stripe", "Coinbase", "Palantir", "Snowflake", "Databricks", "OpenAI"
]

# Popular search terms for tech jobs
SEARCH_TERMS = [
    "software engineer",
    "senior software engineer", 
    "data scientist",
    "product manager",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "machine learning engineer",
    "devops engineer"
]


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🚀 JobSpy Local Database Setup")
    print("=" * 60)
    print("This script will help you set up your local job database")
    print("for faster searches without hitting external APIs every time.")
    print()


def create_database_tables():
    """Create all database tables."""
    print("📊 Creating database tables...")
    try:
        create_tables()
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False


def add_target_companies():
    """Add popular companies to the target companies table."""
    print(f"\n🏢 Adding {len(POPULAR_COMPANIES)} popular companies...")
    
    db = next(get_db())
    added_count = 0
    
    try:
        for company_name in POPULAR_COMPANIES:
            # Check if company already exists
            existing = db.query(TargetCompany).filter(
                TargetCompany.name.ilike(f"%{company_name}%")
            ).first()
            
            if not existing:
                target_company = TargetCompany(
                    name=company_name,
                    display_name=company_name,
                    preferred_sites=["indeed"],
                    search_terms=SEARCH_TERMS[:5],  # Use first 5 search terms
                    location_filters=["USA", "Remote"]
                )
                db.add(target_company)
                added_count += 1
            else:
                print(f"  ⚠️  {company_name} already exists, skipping...")
        
        db.commit()
        print(f"✅ Added {added_count} new companies to database!")
        return True
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error adding companies: {e}")
        return False
    finally:
        db.close()


async def run_initial_scraping():
    """Run initial scraping for a few companies."""
    print("\n🔍 Running initial job scraping...")
    print("This will scrape jobs for the first 5 companies (this may take 5-10 minutes)")
    
    # Ask user if they want to proceed
    response = input("Do you want to start scraping now? (y/n): ").lower().strip()
    if response != 'y':
        print("⏭️  Skipping initial scraping. You can run it later using the API.")
        return True
    
    db = next(get_db())
    
    try:
        # Create scraping request for first 5 companies
        sample_companies = POPULAR_COMPANIES[:5]
        
        scraping_request = BulkScrapingRequest(
            company_names=sample_companies,
            search_terms=["software engineer", "data scientist", "product manager"],
            sites=["indeed"],
            locations=["USA"],
            results_per_company=50,  # Reduced for initial setup
            hours_old=720  # Last 30 days
        )
        
        print(f"🚀 Starting scraping for: {', '.join(sample_companies)}")
        print("This will take several minutes...")
        
        # Run the scraping
        scraping_run = await job_scraper.bulk_scrape_companies(scraping_request, db)
        
        print(f"\n🎉 Initial scraping completed!")
        print(f"📊 Results:")
        print(f"   • Jobs found: {scraping_run.total_jobs_found}")
        print(f"   • New jobs added: {scraping_run.new_jobs_added}")
        print(f"   • Duplicates skipped: {scraping_run.duplicate_jobs_skipped}")
        print(f"   • Duration: {scraping_run.duration_seconds} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return False
    finally:
        db.close()


def show_usage_examples():
    """Show examples of how to use the new system."""
    print("\n" + "=" * 60)
    print("🎯 How to Use Your New Local Job Database")
    print("=" * 60)
    
    print("\n1. 🔍 SEARCH LOCAL JOBS (Faster - No API calls)")
    print("   Endpoint: POST /search-jobs-local-public")
    print("   Example:")
    print("""
   curl -X POST "http://localhost:8000/search-jobs-local-public" \\
     -H "Content-Type: application/json" \\
     -d '{
       "search_term": "software engineer",
       "company_names": ["Google", "Microsoft"],
       "locations": ["USA"],
       "days_old": 30,
       "limit": 20
     }'
   """)
    
    print("\n2. 🏢 MANAGE TARGET COMPANIES")
    print("   Add Company: POST /admin/target-companies")
    print("   List Companies: GET /admin/target-companies")
    
    print("\n3. 🤖 BULK SCRAPE MORE COMPANIES")
    print("   Endpoint: POST /admin/scrape-bulk")
    print("   Example:")
    print("""
   curl -X POST "http://localhost:8000/admin/scrape-bulk" \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer YOUR_TOKEN" \\
     -d '{
       "company_names": ["Netflix", "Uber", "Airbnb"],
       "search_terms": ["software engineer", "data scientist"],
       "sites": ["indeed"],
       "locations": ["USA"],
       "results_per_company": 100
     }'
   """)
    
    print("\n4. 📊 VIEW DATABASE STATS")
    print("   Endpoint: GET /admin/database-stats")
    
    print("\n5. 🔄 SET UP REGULAR SCRAPING")
    print("   • Use cron job or task scheduler")
    print("   • Run bulk scraping daily/weekly")
    print("   • Keep database fresh with new jobs")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete! Your JobSpy local database is ready.")
    print("🌐 Start your server: python backend/main.py")
    print("📖 API Docs: http://localhost:8000/docs")
    print("=" * 60)


async def main():
    """Main setup function."""
    print_banner()
    
    # Step 1: Create database tables
    if not create_database_tables():
        print("❌ Setup failed at database creation step.")
        return
    
    # Step 2: Add target companies
    if not add_target_companies():
        print("❌ Setup failed at adding companies step.")
        return
    
    # Step 3: Run initial scraping (optional)
    if not await run_initial_scraping():
        print("⚠️  Initial scraping failed, but you can try again later.")
    
    # Step 4: Show usage examples
    show_usage_examples()


if __name__ == "__main__":
    # Run the setup
    asyncio.run(main())
    
    print("\n🎉 JobSpy Local Database setup is complete!")
    print("You can now search jobs much faster using your local database.")