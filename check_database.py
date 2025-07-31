#!/usr/bin/env python3
"""
Database Checker Script

This script provides a comprehensive view of what's in your JobSpy database.
"""

import sys
import os
from datetime import datetime, timedelta
from collections import Counter

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import get_db, TargetCompany, ScrapedJob, ScrapingRun
from sqlalchemy import func, desc


def print_banner():
    """Print header banner."""
    print("=" * 60)
    print("🗄️  JOBSPY DATABASE CHECKER")
    print("=" * 60)


def check_database_stats():
    """Show overall database statistics."""
    print("\n📊 DATABASE STATISTICS")
    print("-" * 30)
    
    db = next(get_db())
    
    try:
        # Basic counts
        total_jobs = db.query(ScrapedJob).filter(ScrapedJob.is_active == True).count()
        total_companies = db.query(TargetCompany).filter(TargetCompany.is_active == True).count()
        total_runs = db.query(ScrapingRun).count()
        
        # Jobs in last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        jobs_last_30 = db.query(ScrapedJob).filter(
            ScrapedJob.is_active == True,
            ScrapedJob.date_scraped >= thirty_days_ago
        ).count()
        
        # Jobs in last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        jobs_last_7 = db.query(ScrapedJob).filter(
            ScrapedJob.is_active == True,
            ScrapedJob.date_scraped >= seven_days_ago
        ).count()
        
        # Successful runs
        successful_runs = db.query(ScrapingRun).filter(ScrapingRun.status == "completed").count()
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        print(f"📈 Total Jobs: {total_jobs:,}")
        print(f"🏢 Target Companies: {total_companies}")
        print(f"🤖 Scraping Runs: {total_runs} ({successful_runs} successful, {success_rate:.1f}%)")
        print(f"📅 Jobs (Last 30 days): {jobs_last_30:,}")
        print(f"📅 Jobs (Last 7 days): {jobs_last_7:,}")
        
        if total_jobs == 0:
            print("\n⚠️  DATABASE IS EMPTY!")
            print("   Run 'python setup_job_database.py' to get started")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False
    finally:
        db.close()


def show_top_companies():
    """Show companies with most jobs."""
    print("\n🏆 TOP COMPANIES BY JOB COUNT")
    print("-" * 35)
    
    db = next(get_db())
    
    try:
        top_companies = db.query(
            ScrapedJob.company,
            func.count(ScrapedJob.id).label('job_count')
        ).filter(
            ScrapedJob.is_active == True
        ).group_by(ScrapedJob.company).order_by(
            desc(func.count(ScrapedJob.id))
        ).limit(15).all()
        
        if top_companies:
            for i, (company, count) in enumerate(top_companies, 1):
                print(f"{i:2d}. {company:<25} {count:>5,} jobs")
        else:
            print("No companies found.")
            
    except Exception as e:
        print(f"❌ Error getting top companies: {e}")
    finally:
        db.close()


def show_target_companies():
    """Show configured target companies."""
    print("\n🎯 TARGET COMPANIES")
    print("-" * 25)
    
    db = next(get_db())
    
    try:
        companies = db.query(TargetCompany).filter(
            TargetCompany.is_active == True
        ).order_by(TargetCompany.name).all()
        
        if companies:
            for company in companies:
                print(f"\n🏢 {company.display_name or company.name}")
                print(f"   📍 Locations: {', '.join(company.location_filters)}")
                print(f"   🔍 Search Terms: {', '.join(company.search_terms[:3])}{'...' if len(company.search_terms) > 3 else ''}")
                print(f"   🌐 Sites: {', '.join(company.preferred_sites)}")
                print(f"   📊 Total Jobs: {company.total_jobs_found}")
                
                if company.last_scraped:
                    last_scraped = company.last_scraped.strftime("%Y-%m-%d %H:%M")
                    print(f"   🕒 Last Scraped: {last_scraped}")
                else:
                    print(f"   🕒 Last Scraped: Never")
        else:
            print("No target companies configured.")
            print("Add companies using: POST /admin/target-companies")
            
    except Exception as e:
        print(f"❌ Error getting target companies: {e}")
    finally:
        db.close()


def show_recent_jobs(limit=10):
    """Show most recently scraped jobs."""
    print(f"\n🔍 RECENT JOBS (Last {limit})")
    print("-" * 25)
    
    db = next(get_db())
    
    try:
        recent_jobs = db.query(ScrapedJob).filter(
            ScrapedJob.is_active == True
        ).order_by(
            desc(ScrapedJob.date_scraped)
        ).limit(limit).all()
        
        if recent_jobs:
            for i, job in enumerate(recent_jobs, 1):
                print(f"\n{i:2d}. {job.title}")
                print(f"    🏢 {job.company}")
                print(f"    📍 {job.location or 'Location not specified'}")
                print(f"    🌐 {job.site}")
                
                if job.min_amount or job.max_amount:
                    salary_str = ""
                    if job.min_amount:
                        salary_str += f"${job.min_amount:,.0f}"
                    if job.min_amount and job.max_amount:
                        salary_str += f" - ${job.max_amount:,.0f}"
                    elif job.max_amount:
                        salary_str += f"${job.max_amount:,.0f}"
                    if job.salary_interval:
                        salary_str += f" / {job.salary_interval}"
                    print(f"    💰 {salary_str}")
                
                if job.min_experience_years:
                    print(f"    🎯 {job.min_experience_years}+ years experience")
                
                scraped_date = job.date_scraped.strftime("%Y-%m-%d %H:%M")
                print(f"    📅 Scraped: {scraped_date}")
        else:
            print("No jobs found.")
            
    except Exception as e:
        print(f"❌ Error getting recent jobs: {e}")
    finally:
        db.close()


def show_scraping_runs():
    """Show recent scraping runs."""
    print("\n🤖 RECENT SCRAPING RUNS")
    print("-" * 30)
    
    db = next(get_db())
    
    try:
        runs = db.query(ScrapingRun).order_by(
            desc(ScrapingRun.started_at)
        ).limit(10).all()
        
        if runs:
            for i, run in enumerate(runs, 1):
                status_emoji = "✅" if run.status == "completed" else "❌" if run.status == "failed" else "🔄"
                start_time = run.started_at.strftime("%Y-%m-%d %H:%M")
                
                print(f"\n{i:2d}. {status_emoji} {run.run_type.title()} Run")
                print(f"    📅 Started: {start_time}")
                print(f"    📊 Jobs Found: {run.total_jobs_found}")
                print(f"    ➕ New Jobs: {run.new_jobs_added}")
                print(f"    🔄 Duplicates: {run.duplicate_jobs_skipped}")
                
                if run.duration_seconds:
                    duration = f"{run.duration_seconds} seconds"
                    if run.duration_seconds > 60:
                        duration += f" ({run.duration_seconds // 60}m {run.duration_seconds % 60}s)"
                    print(f"    ⏱️  Duration: {duration}")
                
                if run.error_message:
                    print(f"    ❌ Error: {run.error_message[:100]}...")
        else:
            print("No scraping runs found.")
            
    except Exception as e:
        print(f"❌ Error getting scraping runs: {e}")
    finally:
        db.close()


def show_job_sites_breakdown():
    """Show breakdown by job sites."""
    print("\n🌐 JOBS BY SITE")
    print("-" * 20)
    
    db = next(get_db())
    
    try:
        sites = db.query(
            ScrapedJob.site,
            func.count(ScrapedJob.id).label('count')
        ).filter(
            ScrapedJob.is_active == True
        ).group_by(ScrapedJob.site).order_by(
            desc(func.count(ScrapedJob.id))
        ).all()
        
        if sites:
            for site, count in sites:
                print(f"  {site.title():<15} {count:>5,} jobs")
        else:
            print("No site data found.")
            
    except Exception as e:
        print(f"❌ Error getting site breakdown: {e}")
    finally:
        db.close()


def quick_search(search_term=None):
    """Perform a quick search in the database."""
    if not search_term:
        search_term = input("\n🔍 Enter search term (or press Enter to skip): ").strip()
    
    if not search_term:
        return
    
    print(f"\n🔍 SEARCHING FOR: '{search_term}'")
    print("-" * 30)
    
    db = next(get_db())
    
    try:
        from sqlalchemy import or_
        jobs = db.query(ScrapedJob).filter(
            ScrapedJob.is_active == True,
            or_(
                ScrapedJob.title.ilike(f"%{search_term}%"),
                ScrapedJob.company.ilike(f"%{search_term}%"),
                ScrapedJob.description.ilike(f"%{search_term}%")
            )
        ).order_by(desc(ScrapedJob.date_scraped)).limit(5).all()
        
        if jobs:
            print(f"Found {len(jobs)} matching jobs (showing first 5):")
            for i, job in enumerate(jobs, 1):
                print(f"\n{i}. {job.title}")
                print(f"   🏢 {job.company}")
                print(f"   📍 {job.location or 'Remote/Not specified'}")
                if job.min_amount or job.max_amount:
                    salary = f"${job.min_amount:,.0f}" if job.min_amount else ""
                    if job.max_amount:
                        salary += f" - ${job.max_amount:,.0f}" if salary else f"${job.max_amount:,.0f}"
                    print(f"   💰 {salary}")
        else:
            print(f"No jobs found matching '{search_term}'")
            print("Try broader search terms or scrape more companies.")
            
    except Exception as e:
        print(f"❌ Search error: {e}")
    finally:
        db.close()


def main():
    """Main function."""
    print_banner()
    
    # Check if database has data
    has_data = check_database_stats()
    
    if has_data:
        show_top_companies()
        show_target_companies()
        show_recent_jobs(5)
        show_scraping_runs()
        show_job_sites_breakdown()
        quick_search()
    
    print("\n" + "=" * 60)
    print("✅ Database check complete!")
    
    if has_data:
        print("\n💡 Next steps:")
        print("   • Search jobs: python manual_scrape.py")
        print("   • Web interface: open database_viewer.html")
        print("   • API docs: http://localhost:8000/docs")
    else:
        print("\n💡 To get started:")
        print("   • Run setup: python setup_job_database.py")
        print("   • Or scrape manually: python manual_scrape.py")
    
    print("=" * 60)


if __name__ == "__main__":
    main()