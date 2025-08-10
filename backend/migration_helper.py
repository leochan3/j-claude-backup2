#!/usr/bin/env python3
"""
Migration Helper Script

Provides easy commands for common migration scenarios:
1. Local scraping and migration to production
2. Backup operations
3. Data analysis between environments
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

def scrape_and_migrate():
    """Scrape jobs locally and migrate to production"""
    print("🚀 Starting local scraping and migration process...")
    
    companies = input("Enter companies to scrape (comma-separated): ").strip()
    if not companies:
        print("❌ No companies specified")
        return
    
    companies_list = [c.strip() for c in companies.split(',') if c.strip()]
    
    print(f"\n📋 Will scrape {len(companies_list)} companies:")
    for i, company in enumerate(companies_list, 1):
        print(f"  {i}. {company}")
    
    confirm = input("\n❓ Start scraping? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled by user")
        return
    
    # Start local scraping via API
    import requests
    import json
    
    scrape_data = {
        "company_names": companies_list,
        "search_terms": ["all"],  # Use comprehensive search
        "sites": ["indeed"],
        "locations": ["USA", "Remote"],
        "results_per_company": 1000,
        "hours_old": 72,
        "comprehensive_terms": [
            "tech", "analyst", "manager", "product", "engineer", "market",
            "finance", "business", "associate", "senior", "director",
            "president", "lead", "data", "science", "software", "cloud",
            "developer", "staff", "program", "quality", "security", "specialist"
        ]
    }
    
    try:
        print("🔄 Starting scraping...")
        response = requests.post("http://localhost:8000/scrape-bulk-public", 
                               json=scrape_data, 
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            run_id = data.get('id')
            print(f"✅ Scraping started! Run ID: {run_id}")
            
            # Poll for completion
            import time
            while True:
                time.sleep(5)
                progress_response = requests.get(f"http://localhost:8000/admin/scraping-runs/{run_id}/progress")
                
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    status = progress_data.get('status')
                    progress = progress_data.get('progress', {})
                    
                    if status == 'completed':
                        jobs_found = progress_data.get('total_jobs_found', 0)
                        new_jobs = progress_data.get('new_jobs_added', 0)
                        print(f"🎉 Scraping completed! Found {jobs_found} jobs, {new_jobs} new")
                        break
                    elif status == 'failed':
                        print(f"❌ Scraping failed")
                        return
                    else:
                        current_company = progress.get('current_company', 'Unknown')
                        completed = progress.get('completed_companies', 0)
                        total = progress.get('total_companies', 0)
                        print(f"⏳ In progress: {current_company} ({completed}/{total})")
                else:
                    print(f"❌ Failed to get progress: {progress_response.status_code}")
                    return
        else:
            print(f"❌ Failed to start scraping: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return
    
    # Now migrate to production
    print(f"\n🔄 Starting migration to production...")
    
    postgres_url = os.getenv("DATABASE_URL")
    if not postgres_url:
        postgres_url = input("Enter PostgreSQL URL (or set DATABASE_URL env var): ").strip()
        if not postgres_url:
            print("❌ No PostgreSQL URL provided")
            return
    
    # Run migration
    import subprocess
    cmd = [
        "python", "migrate_sqlite_to_postgres.py",
        "--sqlite-path", "../jobsearch.db",
        "--postgres-url", postgres_url,
        "--batch-size", "500"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("🎉 Migration completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e.stderr}")
        return

def dry_run_migration():
    """Run a dry-run migration to see what would be migrated"""
    postgres_url = os.getenv("DATABASE_URL")
    if not postgres_url:
        postgres_url = input("Enter PostgreSQL URL: ").strip()
        if not postgres_url:
            print("❌ No PostgreSQL URL provided")
            return
    
    import subprocess
    cmd = [
        "python", "migrate_sqlite_to_postgres.py",
        "--sqlite-path", "../jobsearch.db", 
        "--postgres-url", postgres_url,
        "--dry-run"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dry run failed: {e.stderr}")

def analyze_local_data():
    """Analyze local SQLite data"""
    from database import get_db, ScrapedJob, TargetCompany, ScrapingRun
    from sqlalchemy import func
    
    # Use SQLite database
    os.environ["DATABASE_URL"] = "sqlite:///../jobsearch.db"
    
    db = next(get_db())
    try:
        print("📊 Local Database Analysis")
        print("=" * 50)
        
        # Basic counts
        total_jobs = db.query(ScrapedJob).count()
        active_jobs = db.query(ScrapedJob).filter(ScrapedJob.is_active == True).count()
        companies = db.query(TargetCompany).count()
        runs = db.query(ScrapingRun).count()
        
        print(f"Total Jobs: {total_jobs:,}")
        print(f"Active Jobs: {active_jobs:,}")
        print(f"Companies: {companies:,}")
        print(f"Scraping Runs: {runs:,}")
        
        # Recent activity
        recent_jobs = db.query(func.count(ScrapedJob.id)).filter(
            ScrapedJob.date_scraped >= func.date('now', '-7 days')
        ).scalar()
        print(f"Jobs from last 7 days: {recent_jobs:,}")
        
        # Top companies by job count
        print(f"\n🏢 Top Companies by Job Count:")
        top_companies = db.query(
            ScrapedJob.company, 
            func.count(ScrapedJob.id).label('job_count')
        ).filter(
            ScrapedJob.is_active == True
        ).group_by(
            ScrapedJob.company
        ).order_by(
            func.count(ScrapedJob.id).desc()
        ).limit(10).all()
        
        for company, count in top_companies:
            print(f"  {company}: {count:,}")
        
        # Sites breakdown
        print(f"\n🌐 Jobs by Site:")
        site_counts = db.query(
            ScrapedJob.site,
            func.count(ScrapedJob.id).label('job_count')
        ).filter(
            ScrapedJob.is_active == True
        ).group_by(
            ScrapedJob.site
        ).order_by(
            func.count(ScrapedJob.id).desc()
        ).all()
        
        for site, count in site_counts:
            print(f"  {site}: {count:,}")
            
    finally:
        db.close()

def backup_local_db():
    """Create a backup of the local database"""
    import shutil
    
    source = "../jobsearch.db"
    if not os.path.exists(source):
        print(f"❌ Local database not found: {source}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"../jobsearch_backup_{timestamp}.db"
    
    try:
        shutil.copy2(source, backup_path)
        size_mb = os.path.getsize(backup_path) / (1024 * 1024)
        print(f"✅ Backup created: {backup_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Backup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Migration Helper")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape and migrate command
    subparsers.add_parser('scrape-migrate', help='Scrape jobs locally and migrate to production')
    
    # Dry run command
    subparsers.add_parser('dry-run', help='Simulate migration without changes')
    
    # Analysis command
    subparsers.add_parser('analyze', help='Analyze local database')
    
    # Backup command  
    subparsers.add_parser('backup', help='Backup local database')
    
    args = parser.parse_args()
    
    if args.command == 'scrape-migrate':
        scrape_and_migrate()
    elif args.command == 'dry-run':
        dry_run_migration()
    elif args.command == 'analyze':
        analyze_local_data()
    elif args.command == 'backup':
        backup_local_db()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()