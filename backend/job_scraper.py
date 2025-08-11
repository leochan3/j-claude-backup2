"""
Job Scraping Service

This module handles proactive job scraping from various job boards,
deduplication, and storage in the local database.
"""

import asyncio
import time
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import pandas as pd
from jobspy import scrape_jobs

from database import get_db, TargetCompany, ScrapedJob, ScrapingRun, create_job_hash
from models import ScrapingRunCreate, BulkScrapingRequest


class JobScrapingService:
    """Service for scraping and managing job data."""
    
    def __init__(self):
        self.supported_sites = ["indeed", "linkedin", "glassdoor", "zip_recruiter"]
    
    def extract_experience_years(self, description: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract minimum and maximum years of experience from job description."""
        if not description or not isinstance(description, str):
            return None, None
        
        # Common patterns for experience requirements
        patterns = [
            r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:and above|and up|or more|or greater|or higher|plus)?\s*(?:of)?\s*(?:relevant\s*)?(?:experience|exp)",
            r"minimum\s*(\d+)\s*(?:years?|yrs?)",
            r"at least\s*(\d+)\s*(?:years?|yrs?)",
            r"(\d+)[-–](\d+)\s*(?:years?|yrs?)",  # Range pattern
            r"(\d+)\s*to\s*(\d+)\s*(?:years?|yrs?)"  # Range pattern with "to"
        ]
        
        min_years = None
        max_years = None
        
        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        # Range pattern (min-max years)
                        years = [int(y) for y in match if y]
                        if len(years) == 2:
                            min_years = min(years) if min_years is None else min(min_years, min(years))
                            max_years = max(years) if max_years is None else max(max_years, max(years))
                        elif len(years) == 1:
                            year = years[0]
                            min_years = year if min_years is None else min(min_years, year)
                    else:
                        # Single number pattern
                        year = int(match)
                        min_years = year if min_years is None else min(min_years, year)
                except (ValueError, TypeError):
                    continue
        
        return min_years, max_years
    
    async def scrape_company_jobs(
        self, 
        company_name: str, 
        search_terms: List[str] = None,
        sites: List[str] = None,
        locations: List[str] = None,
        results_wanted: int = 1000,
        hours_old: int = 72,  # 3 days
        comprehensive_terms: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Scrape jobs for a specific company."""
        
        if not search_terms:
            # If no search terms provided, use comprehensive default terms for better coverage
            search_terms = ["software engineer", "developer", "data scientist", "product manager", "analyst", "designer"]
        elif len(search_terms) == 1 and search_terms[0].lower().strip() in ["all", "*", "all jobs"]:
            # Use comprehensive terms from frontend, or default comprehensive terms
            if comprehensive_terms:
                search_terms = comprehensive_terms
            else:
                # Fallback to default comprehensive terms
                search_terms = [
                    "tech", "analyst", "manager", "product", "engineer", "market", 
                    "finance", "business", "associate", "senior", "director", 
                    "president", "lead", "data", "science", "software", "cloud", 
                    "developer", "staff", "program", "quality", "security", "specialist"
                ]
        if not sites:
            sites = ["indeed"]
        if not locations:
            locations = ["USA", "Remote", "United States"]
        
        all_jobs = []
        search_analytics = {}  # Track results per search term
        
        print(f"🔍 Scraping jobs for {company_name}")
        
        for location in locations:
            for search_term in search_terms:
                # Create search term with company name
                if search_term.strip():
                    full_search_term = f"{search_term} {company_name}"
                else:
                    # If no specific search term, just search for the company name
                    full_search_term = company_name
                
                try:
                    print(f"  📍 Searching: '{full_search_term}' in {location}")
                    
                    # Call JobSpy
                    jobs_df = await asyncio.to_thread(
                        scrape_jobs,
                        site_name=sites,
                        search_term=full_search_term,
                        location=location,
                        results_wanted=results_wanted,
                        hours_old=hours_old,
                        country_indeed="USA",
                        verbose=1
                    )
                    
                    if jobs_df is not None and not jobs_df.empty:
                        # Filter jobs to include the target company (flexible matching)
                        company_filter = company_name.lower().strip()
                        # Use broader matching for better coverage - include partial matches
                        company_parts = company_filter.split()
                        mask = jobs_df['company'].str.lower().str.contains('|'.join(company_parts), na=False, regex=True)
                        jobs_df = jobs_df[mask]
                        
                        if not jobs_df.empty:
                            jobs_list = jobs_df.to_dict('records')
                            
                            # Track analytics for this search term
                            search_key = f"{search_term}@{location}"
                            search_analytics[search_key] = len(jobs_list)
                            
                            # Clean up NaN values
                            for job in jobs_list:
                                for key, value in job.items():
                                    if pd.isna(value):
                                        job[key] = None
                                # Add metadata
                                job['scraped_search_term'] = search_term
                                job['scraped_location'] = location
                            
                            all_jobs.extend(jobs_list)
                            print(f"  ✅ Found {len(jobs_list)} jobs for {company_name} using '{search_term}' in {location}")
                        else:
                            search_analytics[f"{search_term}@{location}"] = 0
                            print(f"  ❌ No jobs found for {company_name} using '{search_term}' in {location}")
                    else:
                        print(f"  ❌ No results from JobSpy for {company_name} in {location}")
                
                except Exception as e:
                    print(f"  ❌ Error scraping {company_name} in {location}: {str(e)}")
                    continue
                
                # Small delay between requests to be respectful
                await asyncio.sleep(0.5)
        
        # Remove duplicates based on job_url
        unique_jobs = {}
        for job in all_jobs:
            job_url = job.get('job_url', '')
            if job_url and job_url not in unique_jobs:
                unique_jobs[job_url] = job
            elif not job_url:
                # For jobs without URL, use title+company+location as key
                key = f"{job.get('title', '')}-{job.get('company', '')}-{job.get('location', '')}"
                if key not in unique_jobs:
                    unique_jobs[key] = job
        
        final_jobs = list(unique_jobs.values())
        
        # Print analytics summary
        print(f"🎯 Total unique jobs found for {company_name}: {len(final_jobs)}")
        print(f"\n📊 SEARCH TERM PERFORMANCE ANALYTICS:")
        sorted_analytics = sorted(search_analytics.items(), key=lambda x: x[1], reverse=True)
        for search_key, count in sorted_analytics:
            if count > 0:
                print(f"  🏆 {search_key}: {count} jobs")
        
        return final_jobs, search_analytics
    
    def store_jobs_in_database(
        self, 
        jobs: List[Dict[str, Any]], 
        db: Session,
        target_company_id: str = None,
        scraping_run_id: str = None
    ) -> Tuple[int, int]:
        """Store jobs in database with deduplication. Returns (new_jobs, duplicates).

        Robust to duplicates by handling IntegrityError per-row and committing per insert
        to avoid rolling back the entire batch on a single duplicate.
        """

        from sqlalchemy.exc import IntegrityError

        new_jobs_count = 0
        duplicate_jobs_count = 0

        for job_data in jobs:
            # Create job hash for deduplication
            job_hash = create_job_hash(
                title=job_data.get('title', ''),
                company=job_data.get('company', ''),
                location=job_data.get('location', ''),
                job_url=job_data.get('job_url', '')
            )

            # Quick existence check
            try:
                existing_job = db.query(ScrapedJob).filter(
                    ScrapedJob.job_hash == job_hash
                ).first()
                if existing_job:
                    duplicate_jobs_count += 1
                    continue
            except Exception:
                # If the check fails, proceed to try insert and handle IntegrityError
                pass

            # Extract experience years
            min_exp, max_exp = self.extract_experience_years(
                job_data.get('description', '')
            )

            # Parse date_posted
            date_posted = None
            if job_data.get('date_posted'):
                try:
                    if isinstance(job_data['date_posted'], str):
                        date_posted = datetime.fromisoformat(job_data['date_posted'])
                    elif isinstance(job_data['date_posted'], datetime):
                        date_posted = job_data['date_posted']
                    else:
                        # JobSpy often returns date objects
                        from datetime import date
                        if isinstance(job_data['date_posted'], date):
                            date_posted = datetime.combine(job_data['date_posted'], datetime.min.time())
                except Exception as e:
                    print(f"Failed to parse date_posted: {job_data.get('date_posted')} - {e}")
                    date_posted = None

            # Create new job record - use direct apply URL if available
            job_url = job_data.get('job_url_direct') or job_data.get('job_url')
            scraped_job = ScrapedJob(
                job_hash=job_hash,
                job_url=job_url,
                title=job_data.get('title', ''),
                company=job_data.get('company', ''),
                location=job_data.get('location'),
                site=job_data.get('site', 'indeed'),
                description=job_data.get('description'),
                job_type=job_data.get('job_type'),
                is_remote=job_data.get('is_remote'),
                min_amount=job_data.get('min_amount'),
                max_amount=job_data.get('max_amount'),
                salary_interval=job_data.get('interval', 'yearly'),
                currency=job_data.get('currency', 'USD'),
                date_posted=date_posted,
                min_experience_years=min_exp,
                max_experience_years=max_exp,
                target_company_id=target_company_id,
                scraping_run_id=scraping_run_id
            )

            try:
                db.add(scraped_job)
                db.commit()
                new_jobs_count += 1
            except IntegrityError as ie:
                # Unique violation (duplicate hash) or similar
                db.rollback()
                duplicate_jobs_count += 1
            except Exception as e:
                # Unexpected error on this row, rollback and continue
                db.rollback()
                print(f"❌ Error storing job: {str(e)}")
                continue

        print(f"💾 Stored {new_jobs_count} new jobs, skipped {duplicate_jobs_count} duplicates")
        return new_jobs_count, duplicate_jobs_count
    
    async def bulk_scrape_companies(
        self, 
        request: BulkScrapingRequest,
        db: Session
    ) -> ScrapingRun:
        """Scrape multiple companies in bulk."""
        
        # Create scraping run record
        start_time = datetime.now(timezone.utc)
        scraping_run = ScrapingRun(
            run_type="bulk_manual",
            status="running",
            companies_scraped=request.company_names,
            sites_used=request.sites,
            search_parameters=request.model_dump(),
            started_at=start_time
        )
        db.add(scraping_run)
        db.commit()
        
        total_jobs_found = 0
        total_new_jobs = 0
        total_duplicates = 0
        all_company_analytics = {}
        
        try:
            for company_name in request.company_names:
                print(f"\n🏢 Processing company: {company_name}")
                
                # Get or create target company
                target_company = db.query(TargetCompany).filter(
                    TargetCompany.name.ilike(f"%{company_name}%")
                ).first()
                
                if not target_company:
                    target_company = TargetCompany(
                        name=company_name,
                        display_name=company_name,
                        preferred_sites=request.sites,
                        search_terms=request.search_terms,
                        location_filters=request.locations
                    )
                    db.add(target_company)
                    db.commit()
                
                # Scrape jobs for this company
                jobs, company_analytics = await self.scrape_company_jobs(
                    company_name=company_name,
                    search_terms=request.search_terms,
                    sites=request.sites,
                    locations=request.locations,
                    results_wanted=request.results_per_company,
                    hours_old=request.hours_old,
                    comprehensive_terms=getattr(request, 'comprehensive_terms', None)
                )
                
                # Store analytics for this company
                all_company_analytics[company_name] = company_analytics
                
                if jobs:
                    # Store jobs in database
                    new_jobs, duplicates = self.store_jobs_in_database(
                        jobs=jobs,
                        db=db,
                        target_company_id=target_company.id,
                        scraping_run_id=scraping_run.id
                    )
                    
                    total_jobs_found += len(jobs)
                    total_new_jobs += new_jobs
                    total_duplicates += duplicates
                    
                    # Update target company stats
                    target_company.last_scraped = datetime.now(timezone.utc)
                    target_company.total_jobs_found = db.query(ScrapedJob).filter(
                        ScrapedJob.target_company_id == target_company.id,
                        ScrapedJob.is_active == True
                    ).count()
                    
                    db.commit()
                
                # Delay between companies
                await asyncio.sleep(2)
            
            # Update scraping run with results
            scraping_run.status = "completed"
            scraping_run.completed_at = datetime.now(timezone.utc)
            scraping_run.duration_seconds = int((scraping_run.completed_at - start_time).total_seconds())
            scraping_run.total_jobs_found = total_jobs_found
            scraping_run.new_jobs_added = total_new_jobs
            scraping_run.duplicate_jobs_skipped = total_duplicates
            scraping_run.search_analytics = all_company_analytics
            
            db.commit()
            
            print(f"\n🎉 Scraping completed!")
            print(f"📊 Total jobs found: {total_jobs_found}")
            print(f"➕ New jobs added: {total_new_jobs}")
            print(f"🔄 Duplicates skipped: {total_duplicates}")
            
        except Exception as e:
            scraping_run.status = "failed"
            scraping_run.error_message = str(e)
            scraping_run.completed_at = datetime.now(timezone.utc)
            scraping_run.duration_seconds = int((datetime.now(timezone.utc) - start_time).total_seconds())
            scraping_run.search_analytics = all_company_analytics
            db.commit()
            print(f"❌ Scraping failed: {str(e)}")
            raise
        
        return scraping_run

    async def bulk_scrape_companies_with_progress(
        self,
        request: BulkScrapingRequest,
        db: Session,
        scraping_run: ScrapingRun,
        company_timeout_seconds: int = 900
    ) -> None:
        """Scrape companies with real-time progress updates and per-company timeout.

        Updates scraping_run.current_progress, totals, and analytics as it proceeds.
        """
        start_time = scraping_run.started_at
        if start_time and start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        total_jobs_found = 0
        total_new_jobs = 0
        total_duplicates = 0
        all_company_analytics: Dict[str, Dict[str, int]] = {}

        try:
            total_companies = len(request.company_names)
            for company_index, company_name in enumerate(request.company_names, start=1):
                company_start = datetime.now(timezone.utc)

                # Initialize/Update progress for this company
                scraping_run.current_progress = {
                    "phase": "processing_company",
                    "total_companies": total_companies,
                    "completed_companies": company_index - 1,
                    "current_company": company_name,
                    "current_company_index": company_index,
                    "current_search_term": None,
                    "current_location": None,
                    "term_index": 0,
                    "total_terms": 0,
                    "jobs_found_current_company": 0,
                    "company_elapsed_seconds": 0,
                    "company_timeout": False
                }
                db.commit()

                print(f"\n🏢 Processing company: {company_name}")

                # Get or create target company
                target_company = db.query(TargetCompany).filter(
                    TargetCompany.name.ilike(f"%{company_name}%")
                ).first()

                if not target_company:
                    target_company = TargetCompany(
                        name=company_name,
                        display_name=company_name,
                        preferred_sites=request.sites,
                        search_terms=request.search_terms,
                        location_filters=request.locations
                    )
                    db.add(target_company)
                    db.commit()

                # Prepare terms
                if not request.search_terms:
                    terms = [
                        "software engineer", "developer", "data scientist",
                        "product manager", "analyst", "designer"
                    ]
                elif len(request.search_terms) == 1 and request.search_terms[0].lower().strip() in ["all", "*", "all jobs"]:
                    terms = getattr(request, 'comprehensive_terms', None) or [
                        "tech", "analyst", "manager", "product", "engineer", "market",
                        "finance", "business", "associate", "senior", "director",
                        "president", "lead", "data", "science", "software", "cloud",
                        "developer", "staff", "program", "quality", "security", "specialist"
                    ]
                else:
                    terms = request.search_terms.copy() if request.search_terms else []

                # Always add company name as a search term for more targeted results
                company_term = company_name.lower().strip()
                if company_term not in [term.lower().strip() for term in terms]:
                    terms.append(company_name)
                    print(f"🎯 Added company name '{company_name}' as search term")

                locations = request.locations or ["USA", "Remote", "United States"]
                sites = request.sites or ["indeed"]
                results_wanted = request.results_per_company or 1000
                hours_old = request.hours_old or 72

                # In-memory dedup structure for this company to avoid duplicate inserts
                company_unique_map: Dict[str, Dict[str, Any]] = {}
                company_jobs: List[Dict[str, Any]] = []  # will be filled from the map
                company_analytics: Dict[str, int] = {}

                # Set total terms for display (terms x locations)
                total_term_steps = len(terms) * len(locations)
                scraping_run.current_progress.update({
                    "total_terms": total_term_steps
                })
                db.commit()

                step_counter = 0
                # Iterate locations and terms
                for location in locations:
                    for term in terms:
                        step_counter += 1
                        # Check timeout
                        company_elapsed = int((datetime.now(timezone.utc) - company_start).total_seconds())
                        if company_elapsed > company_timeout_seconds:
                            scraping_run.current_progress.update({
                                "phase": "company_timeout",
                                "company_elapsed_seconds": company_elapsed,
                                "company_timeout": True
                            })
                            db.commit()
                            print(f"⏱️ Timeout reached for {company_name}, skipping remaining steps")
                            break

                        # Update progress for this step
                        scraping_run.current_progress.update({
                            "phase": "searching",
                            "current_search_term": term,
                            "current_location": location,
                            "term_index": step_counter,
                            "company_elapsed_seconds": company_elapsed
                        })
                        db.commit()

                        # Build full search term with company
                        full_search = f"{term} {company_name}" if term.strip() else company_name
                        try:
                            print(f"  📍 Searching: '{full_search}' in {location}")
                            jobs_df = await asyncio.to_thread(
                                scrape_jobs,
                                site_name=sites,
                                search_term=full_search,
                                location=location,
                                results_wanted=results_wanted,
                                hours_old=hours_old,
                                country_indeed="USA",
                                verbose=1
                            )

                            found_count = 0
                            newly_added_this_step = 0
                            if jobs_df is not None and not jobs_df.empty:
                                # Filter by company name permissively
                                company_parts = company_name.lower().strip().split()
                                mask = jobs_df['company'].str.lower().str.contains('|'.join(company_parts), na=False, regex=True)
                                jobs_df = jobs_df[mask]

                                if not jobs_df.empty:
                                    jobs_list = jobs_df.to_dict('records')
                                    # Clean NaNs and add metadata
                                    for job in jobs_list:
                                        for key, value in job.items():
                                            if pd.isna(value):
                                                job[key] = None
                                        job['scraped_search_term'] = term
                                        job['scraped_location'] = location
                                    # In-memory dedup by URL or title-company-location
                                    for job in jobs_list:
                                        job_url = job.get('job_url') or job.get('job_url_direct')
                                        if job_url and isinstance(job_url, str) and job_url.strip():
                                            key = f"url::{job_url.strip()}"
                                        else:
                                            key = f"tcl::{(job.get('title') or '').strip().lower()}|{(job.get('company') or '').strip().lower()}|{(job.get('location') or '').strip().lower()}"
                                        if key not in company_unique_map:
                                            company_unique_map[key] = job
                                            newly_added_this_step += 1
                                    found_count = len(jobs_list)

                            # Update analytics and progress
                            search_key = f"{term}@{location}"
                            company_analytics[search_key] = found_count
                            scraping_run.current_progress.update({
                                "last_step_found": found_count,
                                # count only newly added (deduped) jobs for company progress
                                "jobs_found_current_company": scraping_run.current_progress.get("jobs_found_current_company", 0) + newly_added_this_step
                            })
                            db.commit()
                            if found_count:
                                print(f"  ✅ Found {found_count} jobs")
                            else:
                                print(f"  ❌ No jobs found")

                        except Exception as e:
                            print(f"  ❌ Error: {str(e)}")
                            continue

                    # If timeout on inner loop, break outer too
                    if scraping_run.current_progress.get("company_timeout"):
                        break

                # Store results for this company (values of dedup map)
                company_jobs = list(company_unique_map.values())
                if company_jobs:
                    try:
                        new_jobs, duplicates = self.store_jobs_in_database(
                            jobs=company_jobs,
                            db=db,
                            target_company_id=target_company.id,
                            scraping_run_id=scraping_run.id
                        )
                    except Exception as e:
                        # If a unique violation slips through (race/parallel), continue gracefully
                        print(f"  ⚠️ Insert error (continuing): {str(e)}")
                        db.rollback()
                        # Recompute counts conservatively
                        new_jobs = 0
                        duplicates = len(company_jobs)

                    total_jobs_found += len(company_jobs)
                    total_new_jobs += new_jobs
                    total_duplicates += duplicates

                    # Update company stats
                    target_company.last_scraped = datetime.now(timezone.utc)
                    target_company.total_jobs_found = db.query(ScrapedJob).filter(
                        ScrapedJob.target_company_id == target_company.id,
                        ScrapedJob.is_active == True
                    ).count()

                # Save analytics per company
                all_company_analytics[company_name] = company_analytics

                # Update running totals and mark company completed
                scraping_run.total_jobs_found = total_jobs_found
                scraping_run.new_jobs_added = total_new_jobs
                scraping_run.duplicate_jobs_skipped = total_duplicates
                scraping_run.current_progress.update({
                    "phase": "company_completed",
                    "completed_companies": company_index,
                })
                db.commit()

                # Small delay for visibility
                await asyncio.sleep(1)

            # Finalize run
            scraping_run.status = "completed"
            scraping_run.completed_at = datetime.now(timezone.utc)
            scraping_run.duration_seconds = int((scraping_run.completed_at - start_time).total_seconds())
            scraping_run.search_analytics = all_company_analytics
            scraping_run.current_progress = {
                "phase": "completed",
                "total_companies": total_companies,
                "completed_companies": total_companies,
            }
            db.commit()
            print("\n🎉 Background scraping completed!")

        except Exception as e:
            scraping_run.status = "failed"
            scraping_run.error_message = str(e)
            scraping_run.completed_at = datetime.now(timezone.utc)
            scraping_run.current_progress = {
                "phase": "failed",
                "error": str(e)
            }
            db.commit()
    
    def search_local_jobs(
        self, 
        db: Session,
        search_term: str = None,
        company_names: List[str] = None,
        locations: List[str] = None,
        job_types: List[str] = None,
        is_remote: bool = None,
        min_salary: float = None,
        max_salary: float = None,
        max_experience_years: int = None,
        sites: List[str] = None,
        days_old: int = 30,
        limit: int = 100,
        offset: int = 0,
        exclude_keywords: str = None
    ) -> Tuple[List[ScrapedJob], int]:
        """Search jobs in local database."""
        
        print(f"🔎 SEARCH_LOCAL_JOBS called with: search_term={search_term}, sites={sites}, days_old={days_old}")
        
        # Build query
        query = db.query(ScrapedJob).filter(ScrapedJob.is_active == True)
        print(f"🗄️ Base query created, checking for active jobs")
        
        # Date filter
        if days_old:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            query = query.filter(
                or_(
                    ScrapedJob.date_posted >= cutoff_date,
                    ScrapedJob.date_posted.is_(None)  # Include jobs without date_posted
                )
            )
        
        # Search term filter
        if search_term:
            search_filter = or_(
                ScrapedJob.title.ilike(f"%{search_term}%"),
                ScrapedJob.description.ilike(f"%{search_term}%"),
                ScrapedJob.company.ilike(f"%{search_term}%")
            )
            query = query.filter(search_filter)
        
        # Company filter
        if company_names:
            company_filter = or_(*[
                ScrapedJob.company.ilike(f"%{company}%") 
                for company in company_names
            ])
            query = query.filter(company_filter)
        
        # Location filter
        if locations:
            location_filter = or_(*[
                ScrapedJob.location.ilike(f"%{location}%") 
                for location in locations
            ])
            query = query.filter(location_filter)
        
        # Job type filter
        if job_types:
            query = query.filter(ScrapedJob.job_type.in_(job_types))
        
        # Remote filter
        if is_remote is not None:
            query = query.filter(ScrapedJob.is_remote == is_remote)
        
        # Salary filters
        if min_salary:
            query = query.filter(
                or_(
                    ScrapedJob.min_amount >= min_salary,
                    ScrapedJob.max_amount >= min_salary
                )
            )
        
        if max_salary:
            query = query.filter(
                or_(
                    ScrapedJob.min_amount <= max_salary,
                    ScrapedJob.max_amount <= max_salary
                )
            )
        
        # Experience filter
        if max_experience_years:
            query = query.filter(
                or_(
                    ScrapedJob.min_experience_years <= max_experience_years,
                    ScrapedJob.min_experience_years.is_(None)
                )
            )
        
        # Site filter
        if sites:
            query = query.filter(ScrapedJob.site.in_(sites))
        
        # Exclude keywords filter
        if exclude_keywords:
            exclude_terms = [term.strip().lower() for term in exclude_keywords.split(',') if term.strip()]
            if exclude_terms:
                # Exclude jobs that contain any of the exclude terms in the title
                exclude_filters = []
                for term in exclude_terms:
                    exclude_filters.append(~ScrapedJob.title.ilike(f'%{term}%'))
                query = query.filter(and_(*exclude_filters))
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination and ordering
        jobs = query.order_by(
            ScrapedJob.date_posted.desc().nullslast(),
            ScrapedJob.date_scraped.desc()
        ).offset(offset).limit(limit).all()
        
        return jobs, total_count


# Global instance
job_scraper = JobScrapingService()