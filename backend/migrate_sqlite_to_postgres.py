#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script

This script migrates job data from local SQLite database to production PostgreSQL.
It handles deduplication and preserves all job metadata, analytics, and relationships.

Usage:
    python migrate_sqlite_to_postgres.py [--dry-run] [--batch-size=1000]
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, text, select, and_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from tqdm import tqdm

# Import our database models
from database import (
    Base, User, UserPreference, TargetCompany, ScrapedJob, 
    ScrapingRun, create_job_hash, DATABASE_URL
)


class DatabaseMigrator:
    """Handles migration from SQLite to PostgreSQL"""
    
    def __init__(self, sqlite_path: str, postgres_url: str = None, batch_size: int = 1000):
        self.batch_size = batch_size
        
        # SQLite connection
        self.sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
        self.SqliteSession = sessionmaker(bind=self.sqlite_engine)
        
        # PostgreSQL connection (optional for stats-only operations)
        if postgres_url:
            if postgres_url.startswith("postgresql://"):
                postgres_url = postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)
            self.postgres_engine = create_engine(postgres_url)
            self.PostgresSession = sessionmaker(bind=self.postgres_engine)
            print(f"🔗 PostgreSQL target: {postgres_url[:50]}...")
        else:
            self.postgres_engine = None
            self.PostgresSession = None
        
        print(f"🔗 SQLite source: {sqlite_path}")
    
    def verify_connections(self) -> bool:
        """Test both database connections"""
        try:
            # Test SQLite
            with self.sqlite_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM scraped_jobs")).scalar()
                print(f"✅ SQLite connected - {result} jobs found")
            
            # Test PostgreSQL
            with self.postgres_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM scraped_jobs")).scalar()
                print(f"✅ PostgreSQL connected - {result} existing jobs")
            
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def get_local_stats(self) -> Dict[str, Any]:
        """Get statistics about local SQLite database only"""
        stats = {}
        
        sqlite_session = self.SqliteSession()
        
        try:
            # Count records in SQLite
            stats['sqlite_jobs'] = sqlite_session.query(ScrapedJob).count()
            stats['sqlite_companies'] = sqlite_session.query(TargetCompany).count()
            stats['sqlite_scraping_runs'] = sqlite_session.query(ScrapingRun).count()
            stats['sqlite_users'] = sqlite_session.query(User).count()
            
        finally:
            sqlite_session.close()
        
        return stats

    def get_migration_stats(self) -> Dict[str, Any]:
        """Get statistics about what needs to be migrated"""
        if not self.PostgresSession:
            raise ValueError("PostgreSQL connection required for migration stats")
            
        stats = {}
        
        sqlite_session = self.SqliteSession()
        postgres_session = self.PostgresSession()
        
        try:
            # Count records in SQLite
            stats['sqlite_jobs'] = sqlite_session.query(ScrapedJob).count()
            stats['sqlite_companies'] = sqlite_session.query(TargetCompany).count()
            stats['sqlite_scraping_runs'] = sqlite_session.query(ScrapingRun).count()
            stats['sqlite_users'] = sqlite_session.query(User).count()
            
            # Count existing records in PostgreSQL
            stats['postgres_jobs'] = postgres_session.query(ScrapedJob).count()
            stats['postgres_companies'] = postgres_session.query(TargetCompany).count()
            stats['postgres_scraping_runs'] = postgres_session.query(ScrapingRun).count()
            stats['postgres_users'] = postgres_session.query(User).count()
            
            # Calculate what's new
            stats['new_jobs_estimate'] = max(0, stats['sqlite_jobs'] - stats['postgres_jobs'])
            
        finally:
            sqlite_session.close()
            postgres_session.close()
        
        return stats
    
    def migrate_users(self, dry_run: bool = False) -> int:
        """Migrate users and preferences"""
        sqlite_session = self.SqliteSession()
        postgres_session = self.PostgresSession()
        migrated_count = 0
        
        try:
            users = sqlite_session.query(User).all()
            print(f"📋 Migrating {len(users)} users...")
            
            for user in tqdm(users, desc="Users"):
                if dry_run:
                    print(f"  [DRY-RUN] Would migrate user: {user.username}")
                    continue
                
                # Check if user already exists
                existing = postgres_session.query(User).filter(
                    User.id == user.id
                ).first()
                
                if existing:
                    print(f"  ⚠️ User {user.username} already exists, skipping")
                    continue
                
                try:
                    # Create new user
                    new_user = User(
                        id=user.id,
                        username=user.username,
                        email=user.email,
                        hashed_password=user.hashed_password,
                        full_name=user.full_name,
                        is_active=user.is_active,
                        created_at=user.created_at,
                        updated_at=user.updated_at
                    )
                    postgres_session.add(new_user)
                    
                    # Migrate preferences if they exist
                    if user.preferences:
                        prefs = user.preferences
                        new_prefs = UserPreference(
                            id=prefs.id,
                            user_id=prefs.user_id,
                            default_sites=prefs.default_sites,
                            default_search_term=prefs.default_search_term,
                            default_company_filter=prefs.default_company_filter,
                            default_location=prefs.default_location,
                            default_distance=prefs.default_distance,
                            default_job_type=prefs.default_job_type,
                            default_remote=prefs.default_remote,
                            default_results_wanted=prefs.default_results_wanted,
                            default_hours_old=prefs.default_hours_old,
                            default_country=prefs.default_country,
                            default_max_experience=prefs.default_max_experience,
                            default_exclude_keywords=prefs.default_exclude_keywords,
                            min_salary=prefs.min_salary,
                            max_salary=prefs.max_salary,
                            salary_currency=prefs.salary_currency,
                            email_notifications=prefs.email_notifications,
                            job_alert_frequency=prefs.job_alert_frequency,
                            jobs_per_page=prefs.jobs_per_page,
                            default_sort=prefs.default_sort,
                            created_at=prefs.created_at,
                            updated_at=prefs.updated_at
                        )
                        postgres_session.add(new_prefs)
                    
                    postgres_session.commit()
                    migrated_count += 1
                    
                except IntegrityError:
                    postgres_session.rollback()
                    print(f"  ⚠️ User {user.username} constraint violation, skipping")
                except Exception as e:
                    postgres_session.rollback()
                    print(f"  ❌ Error migrating user {user.username}: {e}")
        
        finally:
            sqlite_session.close()
            postgres_session.close()
        
        return migrated_count
    
    def migrate_companies(self, dry_run: bool = False) -> int:
        """Migrate target companies"""
        sqlite_session = self.SqliteSession()
        postgres_session = self.PostgresSession()
        migrated_count = 0
        
        try:
            companies = sqlite_session.query(TargetCompany).all()
            print(f"🏢 Migrating {len(companies)} target companies...")
            
            for company in tqdm(companies, desc="Companies"):
                if dry_run:
                    print(f"  [DRY-RUN] Would migrate company: {company.name}")
                    continue
                
                # Check if company already exists
                existing = postgres_session.query(TargetCompany).filter(
                    TargetCompany.id == company.id
                ).first()
                
                if existing:
                    continue
                
                try:
                    new_company = TargetCompany(
                        id=company.id,
                        name=company.name,
                        display_name=company.display_name,
                        is_active=company.is_active,
                        preferred_sites=company.preferred_sites,
                        search_terms=company.search_terms,
                        location_filters=company.location_filters,
                        last_scraped=company.last_scraped,
                        total_jobs_found=company.total_jobs_found,
                        created_at=company.created_at,
                        updated_at=company.updated_at
                    )
                    postgres_session.add(new_company)
                    postgres_session.commit()
                    migrated_count += 1
                    
                except IntegrityError:
                    postgres_session.rollback()
                except Exception as e:
                    postgres_session.rollback()
                    print(f"  ❌ Error migrating company {company.name}: {e}")
        
        finally:
            sqlite_session.close()
            postgres_session.close()
        
        return migrated_count
    
    def migrate_scraping_runs(self, dry_run: bool = False) -> int:
        """Migrate scraping run records"""
        sqlite_session = self.SqliteSession()
        postgres_session = self.PostgresSession()
        migrated_count = 0
        
        try:
            runs = sqlite_session.query(ScrapingRun).all()
            print(f"🔄 Migrating {len(runs)} scraping runs...")
            
            for run in tqdm(runs, desc="Scraping runs"):
                if dry_run:
                    print(f"  [DRY-RUN] Would migrate run: {run.id}")
                    continue
                
                # Check if run already exists
                existing = postgres_session.query(ScrapingRun).filter(
                    ScrapingRun.id == run.id
                ).first()
                
                if existing:
                    continue
                
                try:
                    new_run = ScrapingRun(
                        id=run.id,
                        run_type=run.run_type,
                        status=run.status,
                        companies_scraped=run.companies_scraped,
                        sites_used=run.sites_used,
                        search_parameters=run.search_parameters,
                        total_jobs_found=run.total_jobs_found,
                        new_jobs_added=run.new_jobs_added,
                        duplicate_jobs_skipped=run.duplicate_jobs_skipped,
                        started_at=run.started_at,
                        completed_at=run.completed_at,
                        duration_seconds=run.duration_seconds,
                        error_message=run.error_message,
                        search_analytics=run.search_analytics,
                        current_progress=run.current_progress
                    )
                    postgres_session.add(new_run)
                    postgres_session.commit()
                    migrated_count += 1
                    
                except IntegrityError:
                    postgres_session.rollback()
                except Exception as e:
                    postgres_session.rollback()
                    print(f"  ❌ Error migrating scraping run {run.id}: {e}")
        
        finally:
            sqlite_session.close()
            postgres_session.close()
        
        return migrated_count
    
    def migrate_jobs(self, dry_run: bool = False) -> Tuple[int, int]:
        """Migrate scraped jobs with deduplication"""
        sqlite_session = self.SqliteSession()
        postgres_session = self.PostgresSession()
        migrated_count = 0
        skipped_count = 0
        
        try:
            # Get total count for progress bar
            total_jobs = sqlite_session.query(ScrapedJob).count()
            print(f"💼 Migrating {total_jobs} scraped jobs...")
            
            # Process in batches
            offset = 0
            with tqdm(total=total_jobs, desc="Jobs") as pbar:
                while True:
                    # Get batch of jobs from SQLite
                    batch = sqlite_session.query(ScrapedJob).offset(offset).limit(self.batch_size).all()
                    if not batch:
                        break
                    
                    # Check existing hashes in PostgreSQL to avoid duplicates
                    batch_hashes = [job.job_hash for job in batch]
                    existing_hashes = set()
                    if batch_hashes:
                        existing_query = postgres_session.query(ScrapedJob.job_hash).filter(
                            ScrapedJob.job_hash.in_(batch_hashes)
                        )
                        existing_hashes = {row[0] for row in existing_query.all()}
                    
                    # Process each job in batch
                    for job in batch:
                        pbar.update(1)
                        
                        if dry_run:
                            if job.job_hash not in existing_hashes:
                                migrated_count += 1
                            else:
                                skipped_count += 1
                            continue
                        
                        # Skip if job already exists (by hash)
                        if job.job_hash in existing_hashes:
                            skipped_count += 1
                            continue
                        
                        try:
                            new_job = ScrapedJob(
                                id=job.id,
                                job_hash=job.job_hash,
                                job_url=job.job_url,
                                title=job.title,
                                company=job.company,
                                location=job.location,
                                site=job.site,
                                description=job.description,
                                job_type=job.job_type,
                                is_remote=job.is_remote,
                                min_amount=job.min_amount,
                                max_amount=job.max_amount,
                                salary_interval=job.salary_interval,
                                currency=job.currency,
                                date_posted=job.date_posted,
                                date_scraped=job.date_scraped,
                                is_active=job.is_active,
                                min_experience_years=job.min_experience_years,
                                max_experience_years=job.max_experience_years,
                                target_company_id=job.target_company_id,
                                scraping_run_id=job.scraping_run_id
                            )
                            postgres_session.add(new_job)
                            migrated_count += 1
                            
                        except Exception as e:
                            print(f"  ❌ Error preparing job {job.id}: {e}")
                            skipped_count += 1
                    
                    # Commit batch
                    if not dry_run:
                        try:
                            postgres_session.commit()
                        except Exception as e:
                            postgres_session.rollback()
                            print(f"  ❌ Batch commit error: {e}")
                            # Count all jobs in this batch as skipped
                            migrated_count -= len([j for j in batch if j.job_hash not in existing_hashes])
                            skipped_count += len([j for j in batch if j.job_hash not in existing_hashes])
                    
                    offset += self.batch_size
        
        finally:
            sqlite_session.close()
            postgres_session.close()
        
        return migrated_count, skipped_count
    
    def run_full_migration(self, dry_run: bool = False, skip_confirmation: bool = False) -> Dict[str, Any]:
        """Run complete migration process"""
        print(f"🚀 Starting {'DRY RUN' if dry_run else 'ACTUAL'} migration...")
        start_time = datetime.now()
        
        # Get initial stats
        stats = self.get_migration_stats()
        print(f"\n📊 Migration Overview:")
        print(f"  SQLite Jobs: {stats['sqlite_jobs']:,}")
        print(f"  PostgreSQL Jobs: {stats['postgres_jobs']:,}")
        print(f"  Estimated New Jobs: {stats['new_jobs_estimate']:,}")
        print(f"  SQLite Companies: {stats['sqlite_companies']:,}")
        print(f"  SQLite Users: {stats['sqlite_users']:,}")
        print(f"  SQLite Scraping Runs: {stats['sqlite_scraping_runs']:,}")
        
        if not dry_run and not skip_confirmation:
            confirm = input("\n❓ Proceed with migration? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Migration cancelled by user")
                return {}
        
        results = {}
        
        try:
            # Migrate in dependency order
            print(f"\n1️⃣ Migrating users...")
            results['users'] = self.migrate_users(dry_run)
            
            print(f"\n2️⃣ Migrating companies...")
            results['companies'] = self.migrate_companies(dry_run)
            
            print(f"\n3️⃣ Migrating scraping runs...")
            results['scraping_runs'] = self.migrate_scraping_runs(dry_run)
            
            print(f"\n4️⃣ Migrating jobs...")
            jobs_migrated, jobs_skipped = self.migrate_jobs(dry_run)
            results['jobs_migrated'] = jobs_migrated
            results['jobs_skipped'] = jobs_skipped
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            results['error'] = str(e)
            return results
        
        # Final summary
        duration = datetime.now() - start_time
        results['duration_seconds'] = int(duration.total_seconds())
        
        print(f"\n🎉 Migration {'simulation' if dry_run else ''} completed!")
        print(f"⏱️  Duration: {duration}")
        print(f"👥 Users migrated: {results.get('users', 0)}")
        print(f"🏢 Companies migrated: {results.get('companies', 0)}")
        print(f"🔄 Scraping runs migrated: {results.get('scraping_runs', 0)}")
        print(f"💼 Jobs migrated: {results.get('jobs_migrated', 0):,}")
        print(f"⏭️  Jobs skipped (duplicates): {results.get('jobs_skipped', 0):,}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite job data to PostgreSQL")
    parser.add_argument("--sqlite-path", 
                       default="../jobsearch.db",
                       help="Path to SQLite database (default: ../jobsearch.db)")
    parser.add_argument("--postgres-url",
                       default=None,
                       help="PostgreSQL URL (uses DATABASE_URL env var if not specified)")
    parser.add_argument("--dry-run", 
                       action="store_true",
                       help="Simulate migration without making changes")
    parser.add_argument("--batch-size", 
                       type=int, 
                       default=1000,
                       help="Batch size for job migration (default: 1000)")
    
    args = parser.parse_args()
    
    # Get PostgreSQL URL
    postgres_url = args.postgres_url or os.getenv("DATABASE_URL")
    if not postgres_url:
        print("❌ No PostgreSQL URL provided. Use --postgres-url or set DATABASE_URL environment variable")
        sys.exit(1)
    
    # Check if SQLite file exists
    if not os.path.exists(args.sqlite_path):
        print(f"❌ SQLite database not found: {args.sqlite_path}")
        sys.exit(1)
    
    # Initialize migrator
    migrator = DatabaseMigrator(
        sqlite_path=args.sqlite_path,
        postgres_url=postgres_url,
        batch_size=args.batch_size
    )
    
    # Verify connections
    if not migrator.verify_connections():
        sys.exit(1)
    
    # Run migration
    results = migrator.run_full_migration(dry_run=args.dry_run)
    
    if 'error' in results:
        sys.exit(1)
    
    print(f"\n✅ Migration completed successfully!")


if __name__ == "__main__":
    main()