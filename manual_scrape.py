#!/usr/bin/env python3
"""
Manual Job Scraping Script

Use this script to quickly scrape jobs for specific companies
without using the web interface or API calls.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import get_db
from job_scraper import job_scraper
from models import BulkScrapingRequest


async def scrape_companies(company_names, search_terms=None, locations=None, results_per_company=100):
    """Scrape jobs for specified companies."""
    
    if not search_terms:
        search_terms = []  # Empty list means use comprehensive default terms
    
    if not locations:
        locations = ["USA", "Remote"]
    
    print(f"🚀 Starting scraping for {len(company_names)} companies...")
    print(f"📍 Companies: {', '.join(company_names)}")
    search_terms_display = ', '.join(search_terms) if search_terms else "COMPREHENSIVE (6 common job types)"
    print(f"🔍 Search terms: {search_terms_display}")
    print(f"📍 Locations: {', '.join(locations)}")
    print(f"📊 Results per company: {results_per_company}")
    print()
    
    # Create scraping request
    request = BulkScrapingRequest(
        company_names=company_names,
        search_terms=search_terms,
        sites=["indeed"],  # Using Indeed for reliability
        locations=locations,
        results_per_company=results_per_company,
        hours_old=720  # Last 30 days
    )
    
    # Get database session
    db = next(get_db())
    
    try:
        # Run the scraping
        scraping_run = await job_scraper.bulk_scrape_companies(request, db)
        
        # Show results
        print("\n" + "="*50)
        print("🎉 SCRAPING COMPLETED!")
        print("="*50)
        print(f"✅ Status: {scraping_run.status}")
        print(f"📊 Total jobs found: {scraping_run.total_jobs_found}")
        print(f"➕ New jobs added: {scraping_run.new_jobs_added}")
        print(f"🔄 Duplicates skipped: {scraping_run.duplicate_jobs_skipped}")
        print(f"⏱️  Duration: {scraping_run.duration_seconds} seconds")
        print(f"🆔 Run ID: {scraping_run.id}")
        
        if scraping_run.error_message:
            print(f"⚠️  Errors: {scraping_run.error_message}")
        
        print("\n🔍 You can now search these jobs using:")
        print("   • Web interface: http://localhost:8000")
        print("   • API: POST /search-jobs-local-public")
        print("   • Test search script: python test_search.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return False
    finally:
        db.close()


def main():
    """Main function with predefined company lists."""
    print("🤖 Manual Job Scraping Script")
    print("="*40)
    
    # Predefined company sets
    company_sets = {
        "1": {
            "name": "Big Tech (FAANG+)",
            "companies": ["Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix", "Tesla"]
        },
        "2": {
            "name": "Hot Startups",
            "companies": ["OpenAI", "Stripe", "Coinbase", "Databricks", "Snowflake", "Palantir", "Uber"]
        },
        "3": {
            "name": "Enterprise Software",
            "companies": ["Salesforce", "Adobe", "Oracle", "SAP", "ServiceNow", "Workday", "Atlassian"]
        },
        "4": {
            "name": "Financial Tech",
            "companies": ["JPMorgan", "Goldman Sachs", "Robinhood", "Plaid", "Square", "PayPal", "Visa"]
        },
        "5": {
            "name": "Custom (Enter your own)"
        }
    }
    
    print("\nChoose a company set to scrape:")
    for key, value in company_sets.items():
        print(f"  {key}. {value['name']}")
        if 'companies' in value:
            print(f"     {', '.join(value['companies'][:3])}{'...' if len(value['companies']) > 3 else ''}")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice in company_sets and 'companies' in company_sets[choice]:
        companies = company_sets[choice]['companies']
        print(f"\n✅ Selected: {company_sets[choice]['name']}")
    elif choice == "5":
        companies_input = input("\nEnter company names (comma-separated): ").strip()
        companies = [c.strip() for c in companies_input.split(',') if c.strip()]
        if not companies:
            print("❌ No companies entered. Exiting.")
            return
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    # Ask for search terms
    use_default_terms = input(f"\nUse default search terms (software engineer, data scientist, product manager)? (y/n): ").lower().strip()
    
    if use_default_terms == 'y':
        search_terms = None  # Will use defaults
    else:
        terms_input = input("Enter search terms (comma-separated): ").strip()
        search_terms = [t.strip() for t in terms_input.split(',') if t.strip()]
        if not search_terms:
            search_terms = None
    
    # Ask for results per company
    try:
        results_input = input(f"\nResults per company (default 100): ").strip()
        results_per_company = int(results_input) if results_input else 100
    except ValueError:
        results_per_company = 100
    
    print(f"\n🚀 Starting scraping...")
    
    # Run the scraping
    success = asyncio.run(scrape_companies(
        company_names=companies,
        search_terms=search_terms,
        results_per_company=results_per_company
    ))
    
    if success:
        print("\n✅ Scraping completed successfully!")
    else:
        print("\n❌ Scraping failed. Check the error messages above.")


if __name__ == "__main__":
    main()