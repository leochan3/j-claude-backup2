from fastapi import FastAPI, HTTPException, Request, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from jobspy import scrape_jobs
import uvicorn
import requests
import io
from datetime import datetime, timedelta
import re
import os
from dotenv import load_dotenv
import json
import asyncio
from openai import OpenAI
import uuid
from sqlalchemy.orm import Session
from database import create_tables, get_db, User
from models import (
    UserCreate, UserLogin, UserResponse, Token,
    UserPreferencesCreate, UserPreferencesUpdate, UserPreferencesResponse,
    SaveJobRequest as NewSaveJobRequest, SavedJobUpdate, SavedJobResponse as NewSavedJobResponse,
    SearchHistoryResponse, SavedSearchCreate, SavedSearchUpdate, SavedSearchResponse,
    AuthenticatedJobSearchRequest
)
from auth import authenticate_user, create_access_token, get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
from user_service import UserService
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client with better error handling
openai_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# Environment validation
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized for AI filtering")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        openai_client = None
else:
    print("OpenAI API key not found or not configured. AI filtering will not be available.")
    print("To enable AI features, add your OpenAI API key to .env file")

# Initialize database
create_tables()

app = FastAPI(
    title="JobSpy API with User Accounts",
    description="Job scraping API with user authentication, preferences, and personalized job management",
    version="3.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Saved Jobs Storage Management
SAVED_JOBS_FILE = "saved_jobs.json"

class JobSearchRequest(BaseModel):
    site_name: Optional[List[str]] = ["indeed"]  # Default to Indeed only
    search_term: str  # Job title/role only
    company_filter: Optional[str] = None  # Company to filter for (None = no filter)
    location: Optional[str] = "USA"  # Comma-separated locations supported, e.g. "New York, Boston, Los Angeles"
    distance: Optional[int] = 50
    job_type: Optional[str] = None  # fulltime, parttime, internship, contract
    is_remote: Optional[bool] = None
    results_wanted: Optional[int] = 1000  # Match your Jupyter example
    hours_old: Optional[int] = 10000  # Match your Jupyter example
    country_indeed: Optional[str] = "USA"
    easy_apply: Optional[bool] = None
    description_format: Optional[str] = "markdown"
    offset: Optional[int] = 0
    verbose: Optional[int] = 2  # More verbose to help debug
    max_years_experience: Optional[int] = None  # New: filter jobs by max years of experience

class JobSearchResponse(BaseModel):
    success: bool
    message: str
    job_count: int
    jobs: List[dict]
    search_params: dict
    timestamp: str

# AI Filtering Models
class AIFilterRequest(BaseModel):
    jobs: List[Dict[str, Any]]  # The jobs to filter
    analysis_prompt: str  # What to analyze (e.g., "summarize years of experience required")
    filter_criteria: Optional[str] = None  # How to filter (e.g., "filter jobs requiring 5+ years")

class AIAnalysisResult(BaseModel):
    job_id: int
    job_title: str
    job_company: str
    analysis_result: str  # AI's analysis of this job
    meets_criteria: Optional[bool] = None  # Whether it meets filter criteria

class AIFilterResponse(BaseModel):
    success: bool
    message: str
    original_count: int
    analyzed_jobs: List[AIAnalysisResult]
    filtered_count: Optional[int] = None
    filtered_jobs: Optional[List[Dict[str, Any]]] = None
    timestamp: str

# Saved Jobs Models
class SaveJobRequest(BaseModel):
    job_data: Dict[str, Any]  # The complete job object
    notes: Optional[str] = ""  # User notes about the job

class SavedJob(BaseModel):
    id: str
    job_data: Dict[str, Any]
    notes: str
    saved_at: str
    applied: bool = False  # New field to track application status
    applied_at: Optional[str] = None  # When the job was applied to
    save_for_later: bool = False  # New field for save for later
    not_interested: bool = False  # New field for not interested
    tags: List[str] = []

class SavedJobResponse(BaseModel):
    success: bool
    message: str
    saved_job: Optional[SavedJob] = None

class SavedJobsListResponse(BaseModel):
    success: bool
    message: str
    saved_jobs: List[SavedJob]
    total_count: int
    timestamp: str

# Saved Jobs Utility Functions (defined after models)
def load_saved_jobs() -> List[SavedJob]:
    """Load saved jobs from JSON file, skipping invalid records but logging errors."""
    try:
        if os.path.exists(SAVED_JOBS_FILE):
            with open(SAVED_JOBS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                saved_jobs = []
                errors = []
                for idx, job_data in enumerate(data):
                    # Handle backward compatibility for existing jobs without applied status
                    if 'applied' not in job_data:
                        job_data['applied'] = False
                    if 'applied_at' not in job_data:
                        job_data['applied_at'] = None
                    if 'save_for_later' not in job_data:
                        job_data['save_for_later'] = False
                    if 'not_interested' not in job_data:
                        job_data['not_interested'] = False
                    try:
                        saved_jobs.append(SavedJob(**job_data))
                    except Exception as e:
                        print(f"Error loading saved job at index {idx}: {e}\nData: {job_data}")
                        errors.append((idx, str(e)))
                if not saved_jobs and errors:
                    raise Exception(f"All saved jobs failed to load. Errors: {errors}")
                return saved_jobs
        return []
    except Exception as e:
        print(f"Error loading saved jobs: {e}")
        raise

def save_jobs_to_file(saved_jobs: List[SavedJob]):
    """Save jobs list to JSON file"""
    try:
        data = [job.dict() for job in saved_jobs]
        with open(SAVED_JOBS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving jobs to file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save jobs: {str(e)}")

def job_already_saved(job_data: Dict[str, Any], saved_jobs: List[SavedJob]) -> bool:
    """Check if a job is already saved based on job URL or title+company combination"""
    job_url = job_data.get('job_url', '')
    job_title = job_data.get('title', '').lower().strip()
    job_company = job_data.get('company', '').lower().strip()
    
    for saved_job in saved_jobs:
        saved_url = saved_job.job_data.get('job_url', '')
        saved_title = saved_job.job_data.get('title', '').lower().strip()
        saved_company = saved_job.job_data.get('company', '').lower().strip()
        
        # Check by URL first (most reliable)
        if job_url and saved_url and job_url == saved_url:
            return True
            
        # Check by title + company combination
        if job_title and job_company and saved_title and saved_company:
            if job_title == saved_title and job_company == saved_company:
                return True
    
    return False

def extract_max_years_experience(description: str) -> Optional[int]:
    """Extract the maximum years of experience required from a job description using improved regex patterns."""
    if not description or not isinstance(description, str):
        return None
    patterns = [
        r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:and above|and up|or more|or greater|or higher|plus)?\s*(?:of)?\s*(?:relevant\s*)?(?:experience|exp|in|as)?",
        r"minimum\s*(\d+)\s*(?:years?|yrs?)",
        r"at least\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)[-–]year"
    ]
    matches = []
    for pattern in patterns:
        for match in re.findall(pattern, description, re.IGNORECASE):
            try:
                matches.append(int(match))
            except Exception:
                continue
    if matches:
        return max(matches)
    return None

@app.get("/")
async def root():
    return {
        "message": "JobSpy API with User Accounts is running!", 
        "endpoints": [
            "/docs - API documentation",
            "/auth/register - User registration",
            "/auth/login - User login",
            "/search-jobs - Search for jobs (authenticated)",
            "/ai-filter-jobs - AI-powered job analysis and filtering",
            "/user/preferences - Manage user preferences",
            "/user/saved-jobs - Manage saved jobs",
            "/user/search-history - View search history",
            "/user/saved-searches - Manage saved search templates",
            "/supported-sites - Get supported job sites",
            "/supported-countries - Get supported countries",
            "/health - Health check"
        ],
        "features": {
            "user_accounts": True,
            "personalized_preferences": True,
            "job_session_storage": True,
            "ai_filtering": openai_client is not None,
            "ai_model": OPENAI_MODEL if openai_client else "Not configured"
        }
    }

@app.get("/supported-sites")
async def get_supported_sites():
    """Get list of supported job sites"""
    return {
        "supported_sites": [
            "linkedin",
            "indeed", 
            "glassdoor",
            "zip_recruiter", 
            "google",
            "bayt",
            "naukri"
        ],
        "notes": {
            "linkedin": "Global search, may require rate limiting",
            "indeed": "Best scraper with no rate limiting, supports many countries",
            "glassdoor": "Supports many countries, requires country_indeed parameter",
            "zip_recruiter": "US/Canada only",
            "google": "Requires very specific search syntax in google_search_term",
            "bayt": "International search, uses search_term only",
            "naukri": "India-focused job board"
        }
    }

# Authentication Endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_create: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account"""
    return UserService.create_user(db, user_create)

@app.post("/auth/login", response_model=Token)
async def login_user(user_login: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token"""
    user = authenticate_user(db, user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.from_orm(user)
    )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse.from_orm(current_user)

# User Preferences Endpoints
@app.get("/user/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user preferences"""
    preferences = UserService.get_user_preferences(db, current_user.id)
    if not preferences:
        raise HTTPException(status_code=404, detail="User preferences not found")
    return UserPreferencesResponse.from_orm(preferences)

@app.put("/user/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    preferences_update: UserPreferencesUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user preferences"""
    updated_preferences = UserService.update_user_preferences(
        db, current_user.id, preferences_update
    )
    return UserPreferencesResponse.from_orm(updated_preferences)

@app.get("/supported-countries")
async def get_supported_countries():
    """Get list of supported countries for Indeed/Glassdoor"""
    countries = [
        "Argentina", "Australia", "Austria", "Bahrain", "Belgium", "Brazil", 
        "Canada", "Chile", "China", "Colombia", "Costa Rica", "Czech Republic",
        "Denmark", "Ecuador", "Egypt", "Finland", "France", "Germany", "Greece", 
        "Hong Kong", "Hungary", "India", "Indonesia", "Ireland", "Israel", "Italy",
        "Japan", "Kuwait", "Luxembourg", "Malaysia", "Mexico", "Morocco", 
        "Netherlands", "New Zealand", "Nigeria", "Norway", "Oman", "Pakistan",
        "Panama", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
        "Saudi Arabia", "Singapore", "South Africa", "South Korea", "Spain", 
        "Sweden", "Switzerland", "Taiwan", "Thailand", "Turkey", "Ukraine",
        "United Arab Emirates", "UK", "USA", "Uruguay", "Venezuela", "Vietnam"
    ]
    return {
        "supported_countries": countries,
        "note": "These countries are supported for Indeed and Glassdoor. LinkedIn searches globally, ZipRecruiter supports US/Canada only."
    }


def filter_jobs_by_company(jobs_df, company_filter):
    """A simplified, stricter filter for companies."""
    if not company_filter or jobs_df is None or jobs_df.empty:
        return jobs_df

    company_filter_clean = company_filter.lower().strip()
    
    print("--- Simplified Company Filtering ---")
    print(f"Filtering for companies that start with: '{company_filter_clean}'")

    # This prevents errors if the 'company' column contains non-string data
    jobs_df['company'] = jobs_df['company'].astype(str)

    mask = jobs_df['company'].str.lower().str.strip().str.startswith(company_filter_clean, na=False)
    
    filtered_df = jobs_df[mask].copy()
    
    print(f"Before: {len(jobs_df)} jobs. After: {len(filtered_df)} jobs.")
    print("---------------------------------")
    return filtered_df

async def search_single_company(search_term: str, company: str, search_params: dict):
    """Search for jobs for a single company"""
    # Create search term with company
    actual_search_term = f"{search_term} {company}".strip()
    
    # Update search params with company-specific term
    company_search_params = search_params.copy()
    company_search_params["search_term"] = actual_search_term
    
    print(f"Searching for '{search_term}' at '{company}' with term: '{actual_search_term}'")
    
    try:
        # Call JobSpy for this company
        jobs_df = scrape_jobs(**company_search_params)
        
        if jobs_df is not None and not jobs_df.empty:
            print(f"Found {len(jobs_df)} jobs for company '{company}' before filtering")
            
            # Apply strict company filtering
            jobs_df = filter_jobs_by_company(jobs_df, company)
            
            print(f"Found {len(jobs_df)} jobs for company '{company}' after filtering")
            return jobs_df
        else:
            print(f"No jobs found for company '{company}'")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error searching for company '{company}': {str(e)}")
        return pd.DataFrame()


@app.post("/search-jobs", response_model=JobSearchResponse)
async def search_jobs(
    request: AuthenticatedJobSearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Search for jobs using JobSpy with user preferences and search history tracking."""
    try:
        start_time = time.time()
        
        # Get user preferences to fill in missing values
        user_preferences = UserService.get_user_preferences(db, current_user.id)
        
        # Apply user preferences as defaults for missing values
        effective_request = JobSearchRequest(
            site_name=request.site_name or (user_preferences.default_sites if user_preferences else ["indeed"]),
            search_term=request.search_term,
            company_filter=request.company_filter,
            location=request.location or (user_preferences.default_location if user_preferences else "USA"),
            distance=request.distance or (user_preferences.default_distance if user_preferences else 50),
            job_type=request.job_type or (user_preferences.default_job_type if user_preferences else None),
            is_remote=request.is_remote if request.is_remote is not None else (user_preferences.default_remote if user_preferences else None),
            results_wanted=request.results_wanted or (user_preferences.default_results_wanted if user_preferences else 100),
            hours_old=request.hours_old or (user_preferences.default_hours_old if user_preferences else 168),
            country_indeed=request.country_indeed or (user_preferences.default_country if user_preferences else "USA"),
            max_years_experience=request.max_years_experience or (user_preferences.default_max_experience if user_preferences else None)
        )
        
        # Parse multiple job titles (comma-separated)
        job_titles = [t.strip() for t in effective_request.search_term.split(',') if t.strip()]
        # Parse multiple companies (comma-separated)
        companies = []
        if effective_request.company_filter and effective_request.company_filter.strip():
            companies = [company.strip() for company in effective_request.company_filter.split(',') if company.strip()]
        # Parse multiple locations (comma-separated)
        locations = [loc.strip() for loc in effective_request.location.split(',') if loc.strip()] if effective_request.location else ["USA"]
        # Prepare base parameters for JobSpy (except location)
        base_search_params = {
            "site_name": effective_request.site_name,
            # location will be set per-iteration
            "distance": effective_request.distance,
            "job_type": effective_request.job_type,
            "is_remote": effective_request.is_remote,
            "results_wanted": effective_request.results_wanted,
            "hours_old": effective_request.hours_old,
            "country_indeed": effective_request.country_indeed,
            "easy_apply": getattr(effective_request, 'easy_apply', None),
            "description_format": getattr(effective_request, 'description_format', 'markdown'),
            "offset": getattr(effective_request, 'offset', 0),
            "verbose": getattr(effective_request, 'verbose', 2)
        }
        base_search_params = {k: v for k, v in base_search_params.items() if v is not None}
        all_jobs_df = pd.DataFrame()
        # Nested loops for all combinations
        if companies:
            print(f"Multi-company search for {len(companies)} companies: {companies}")
            for job_title in job_titles:
                for company in companies:
                    for location in locations:
                        company_search_params = base_search_params.copy()
                        company_search_params["location"] = location
                        company_jobs_df = await search_single_company(job_title, company, company_search_params)
                        if not company_jobs_df.empty:
                            company_jobs_df = company_jobs_df.copy()
                            company_jobs_df['search_company'] = company
                            company_jobs_df['search_title'] = job_title
                            company_jobs_df['search_location'] = location
                            if all_jobs_df.empty:
                                all_jobs_df = company_jobs_df
                            else:
                                all_jobs_df = pd.concat([all_jobs_df, company_jobs_df], ignore_index=True)
        else:
            print("No company filter - searching all companies")
            for job_title in job_titles:
                for location in locations:
                    search_params = base_search_params.copy()
                    search_params["search_term"] = job_title
                    search_params["location"] = location
                    print(f"JobSpy Parameters: {search_params}")
                    jobs_df = scrape_jobs(**search_params)
                    if jobs_df is not None and not jobs_df.empty:
                        jobs_df = jobs_df.copy()
                        jobs_df['search_title'] = job_title
                        jobs_df['search_location'] = location
                        if all_jobs_df.empty:
                            all_jobs_df = jobs_df
                        else:
                            all_jobs_df = pd.concat([all_jobs_df, jobs_df], ignore_index=True)
        # Remove duplicates based on job_url or title+company+location combination
        if not all_jobs_df.empty:
            print(f"Total jobs found across all searches: {len(all_jobs_df)}")
            original_count = len(all_jobs_df)
            all_jobs_df = all_jobs_df.drop_duplicates(subset=['job_url'], keep='first')
            if len(all_jobs_df) < original_count:
                print(f"Removed {original_count - len(all_jobs_df)} duplicate jobs based on URL")
        else:
            print("No jobs found")
        # Convert DataFrame to list of dictionaries
        if not all_jobs_df.empty:
            jobs_list = all_jobs_df.to_dict('records')
            for job in jobs_list:
                for key, value in job.items():
                    if pd.isna(value):
                        job[key] = None
            # Filter by max_years_experience if set
            if effective_request.max_years_experience is not None:
                filtered_jobs = []
                for job in jobs_list:
                    max_yoe = extract_max_years_experience(job.get('description', ''))
                    if max_yoe is None or max_yoe <= effective_request.max_years_experience:
                        filtered_jobs.append(job)
                jobs_list = filtered_jobs
            filter_info = ""
            if companies:
                if len(companies) == 1:
                    filter_info = f" (company: {companies[0]})"
                else:
                    filter_info = f" (companies: {', '.join(companies)})"
            if len(job_titles) > 1:
                filter_info += f" (titles: {', '.join(job_titles)})"
            if len(locations) > 1:
                filter_info += f" (locations: {', '.join(locations)})"
            if effective_request.max_years_experience is not None:
                filter_info += f" (max YOE: {effective_request.max_years_experience})"
            
            # Save search to history if requested
            search_duration = int(time.time() - start_time)
            if getattr(request, 'save_search', True):  # Default to saving search history
                UserService.add_search_history(
                    db, current_user.id,
                    effective_request.dict(),
                    len(jobs_list),
                    search_duration
                )
            
            return JobSearchResponse(
                success=True,
                message=f"Successfully found {len(jobs_list)} jobs{filter_info}",
                job_count=len(jobs_list),
                jobs=jobs_list,
                search_params={**base_search_params, "company_filter": effective_request.company_filter, "search_term": effective_request.search_term, "location": effective_request.location, "max_years_experience": effective_request.max_years_experience},
                timestamp=datetime.now().isoformat()
            )
        else:
            filter_info = ""
            if companies:
                if len(companies) == 1:
                    filter_info = f" for company '{companies[0]}'"
                else:
                    filter_info = f" for companies: {', '.join(companies)}"
            if len(job_titles) > 1:
                filter_info += f" for titles: {', '.join(job_titles)}"
            if len(locations) > 1:
                filter_info += f" for locations: {', '.join(locations)}"
            # Still save search to history even if no results
            search_duration = int(time.time() - start_time)
            if getattr(request, 'save_search', True):
                UserService.add_search_history(
                    db, current_user.id,
                    effective_request.dict(),
                    0,
                    search_duration
                )
            
            return JobSearchResponse(
                success=True,
                message=f"No jobs found matching your criteria{filter_info}",
                job_count=0,
                jobs=[],
                search_params={**base_search_params, "company_filter": effective_request.company_filter, "search_term": effective_request.search_term, "location": effective_request.location},
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error scraping jobs: {str(e)}"
        )

@app.post("/search-jobs-public", response_model=JobSearchResponse)
async def search_jobs_public(request: JobSearchRequest):
    """Search for jobs using JobSpy without authentication (public endpoint)."""
    try:
        start_time = time.time()
        
        # Use the request as-is since it already has default values
        effective_request = request
        
        # Parse multiple job titles (comma-separated)
        job_titles = [t.strip() for t in effective_request.search_term.split(',') if t.strip()]
        # Parse multiple companies (comma-separated)
        companies = []
        if effective_request.company_filter and effective_request.company_filter.strip():
            companies = [company.strip() for company in effective_request.company_filter.split(',') if company.strip()]
        # Parse multiple locations (comma-separated)
        locations = [loc.strip() for loc in effective_request.location.split(',') if loc.strip()] if effective_request.location else ["USA"]
        # Prepare base parameters for JobSpy (except location)
        base_search_params = {
            "site_name": effective_request.site_name,
            # location will be set per-iteration
            "distance": effective_request.distance,
            "job_type": effective_request.job_type,
            "is_remote": effective_request.is_remote,
            "results_wanted": effective_request.results_wanted,
            "hours_old": effective_request.hours_old,
            "country_indeed": effective_request.country_indeed,
            "easy_apply": getattr(effective_request, 'easy_apply', None),
            "description_format": getattr(effective_request, 'description_format', 'markdown'),
            "offset": getattr(effective_request, 'offset', 0),
            "verbose": getattr(effective_request, 'verbose', 2)
        }
        base_search_params = {k: v for k, v in base_search_params.items() if v is not None}
        all_jobs_df = pd.DataFrame()
        # Nested loops for all combinations
        if companies:
            print(f"Multi-company search for {len(companies)} companies: {companies}")
            for job_title in job_titles:
                for company in companies:
                    for location in locations:
                        company_search_params = base_search_params.copy()
                        company_search_params["location"] = location
                        company_jobs_df = await search_single_company(job_title, company, company_search_params)
                        if not company_jobs_df.empty:
                            company_jobs_df = company_jobs_df.copy()
                            company_jobs_df['search_company'] = company
                            company_jobs_df['search_title'] = job_title
                            company_jobs_df['search_location'] = location
                            if all_jobs_df.empty:
                                all_jobs_df = company_jobs_df
                            else:
                                all_jobs_df = pd.concat([all_jobs_df, company_jobs_df], ignore_index=True)
        else:
            print("No company filter - searching all companies")
            for job_title in job_titles:
                for location in locations:
                    search_params = base_search_params.copy()
                    search_params["search_term"] = job_title
                    search_params["location"] = location
                    print(f"JobSpy Parameters: {search_params}")
                    jobs_df = scrape_jobs(**search_params)
                    if not jobs_df.empty:
                        jobs_df = jobs_df.copy()
                        jobs_df['search_title'] = job_title
                        jobs_df['search_location'] = location
                        if all_jobs_df.empty:
                            all_jobs_df = jobs_df
                        else:
                            all_jobs_df = pd.concat([all_jobs_df, jobs_df], ignore_index=True)
        
        # Process results (same as authenticated version)
        if not all_jobs_df.empty:
            # Convert to dict format
            jobs_list = []
            for _, row in all_jobs_df.iterrows():
                job_dict = {}
                for col in all_jobs_df.columns:
                    if pd.notna(row[col]):
                        if col in ['min_amount', 'max_amount'] and isinstance(row[col], (int, float)):
                            job_dict[col] = float(row[col])
                        elif col == 'emails' and isinstance(row[col], list):
                            job_dict[col] = row[col]
                        else:
                            job_dict[col] = str(row[col])
                jobs_list.append(job_dict)
            
            # Remove duplicates based on job_url
            unique_jobs = {}
            for job in jobs_list:
                job_url = job.get('job_url', '')
                if job_url and job_url not in unique_jobs:
                    unique_jobs[job_url] = job
                elif not job_url:
                    # For jobs without URL, add them all (they'll be handled by frontend)
                    title_company_key = f"{job.get('title', '')}-{job.get('company', '')}-{job.get('location', '')}"
                    unique_jobs[title_company_key] = job
            
            unique_jobs_list = list(unique_jobs.values())
            
            end_time = time.time()
            search_duration = end_time - start_time
            
            return JobSearchResponse(
                success=True,
                message=f"Found {len(unique_jobs_list)} unique jobs (removed {len(jobs_list) - len(unique_jobs_list)} duplicates) in {search_duration:.2f} seconds",
                job_count=len(unique_jobs_list),
                jobs=unique_jobs_list,
                search_params={**base_search_params, "company_filter": effective_request.company_filter, "search_term": effective_request.search_term, "location": effective_request.location},
                timestamp=datetime.now().isoformat()
            )
        else:
            return JobSearchResponse(
                success=True,
                message="No jobs found matching your criteria",
                job_count=0,
                jobs=[],
                search_params={**base_search_params, "company_filter": effective_request.company_filter, "search_term": effective_request.search_term, "location": effective_request.location},
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error scraping jobs: {str(e)}"
        )

@app.post("/generate-resume-pdf")
async def generate_resume_pdf(request: dict):
    """Generate PDF from LaTeX resume code with multiple service fallbacks"""
    try:
        from fastapi.responses import StreamingResponse
        import base64
        
        latex_code = request.get('latex_code', '')
        if not latex_code:
            raise HTTPException(status_code=400, detail="LaTeX code is required")
        
        # List of LaTeX compilation services to try (most reliable first)
        services = [
            {
                "name": "LaTeX.Online (texlive2021)",
                "url": "https://texlive2021.latexonline.cc/compile",
                "method": "multipart"
            },
            {
                "name": "LaTeX.Online (main)",
                "url": "https://latexonline.cc/compile",
                "method": "multipart_alt"
            },
            {
                "name": "LaTeX.Online (aslushnikov)",
                "url": "https://latex.aslushnikov.com/compile",
                "method": "multipart"
            }
        ]
        
        last_error = None
        
        for service in services:
            try:
                print(f"Trying {service['name']}...")
                
                if service['method'] == 'multipart':
                    # Standard multipart file upload
                    files = {
                        'file': ('resume.tex', latex_code, 'text/plain')
                    }
                    headers = {'User-Agent': 'JobSpy-Resume-Builder/1.0'}
                    response = requests.post(service['url'], files=files, headers=headers, timeout=60)
                    
                elif service['method'] == 'multipart_alt':
                    # Alternative multipart format
                    files = {
                        'filecontents[]': ('resume.tex', latex_code, 'text/plain')
                    }
                    data = {
                        'filename[]': 'resume.tex',
                        'compiler': 'pdflatex'
                    }
                    headers = {'User-Agent': 'JobSpy-Resume-Builder/1.0'}
                    response = requests.post(service['url'], files=files, data=data, headers=headers, timeout=60)
                
                else:
                    print(f"⚠️ Unknown method for {service['name']}")
                    continue
                
                print(f"{service['name']} response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Check if response is actually a PDF
                    content = response.content
                    if content.startswith(b'%PDF'):
                        print(f"✅ PDF generation successful with {service['name']}")
                        return StreamingResponse(
                            io.BytesIO(content),
                            media_type="application/pdf",
                            headers={"Content-Disposition": "attachment; filename=resume.pdf"}
                        )
                    else:
                        print(f"❌ {service['name']} returned non-PDF content")
                        last_error = f"{service['name']}: Response not a valid PDF"
                else:
                    error_text = response.text[:500] if response.text else "No error message"
                    print(f"❌ {service['name']} failed: {response.status_code} - {error_text}")
                    last_error = f"{service['name']}: HTTP {response.status_code} - {error_text}"
                    
            except requests.exceptions.Timeout:
                print(f"⏰ {service['name']} timed out")
                last_error = f"{service['name']}: Request timed out"
                continue
            except requests.exceptions.RequestException as e:
                print(f"🌐 {service['name']} connection error: {str(e)}")
                last_error = f"{service['name']}: Connection error - {str(e)}"
                continue
            except Exception as e:
                print(f"❌ {service['name']} unexpected error: {str(e)}")
                last_error = f"{service['name']}: Unexpected error - {str(e)}"
                continue
        
        # If all services failed, provide detailed error message
        raise HTTPException(
            status_code=503,
            detail=f"All LaTeX compilation services are currently unavailable. Last error: {last_error}. Please try downloading the LaTeX file and compiling it locally using Overleaf (overleaf.com), TeXLive, or MiKTeX."
        )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"💥 Unexpected error in PDF generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error generating PDF: {str(e)}. Please try downloading the LaTeX file instead."
        )

# AI Filtering Functions
async def analyze_job_with_ai(job: Dict[str, Any], analysis_prompt: str, job_id: int, client) -> AIAnalysisResult:
    """Analyze a single job using OpenAI"""
    
    # Prepare job information for analysis
    job_info = {
        "title": job.get("title", "N/A"),
        "company": job.get("company", "N/A"),
        "location": job.get("location", "N/A"),
        "description": job.get("description", "N/A")[:5000],  # Increased limit for better analysis
        "job_type": job.get("job_type", "N/A"),
        "salary_min": job.get("min_amount", "N/A"),
        "salary_max": job.get("max_amount", "N/A"),
        "date_posted": job.get("date_posted", "N/A")
    }
    
    prompt = f"""
    Analyze this job posting based on the following request: "{analysis_prompt}"
    
    Job Information:
    - Title: {job_info['title']}
    - Company: {job_info['company']}
    - Location: {job_info['location']}
    - Type: {job_info['job_type']}
    - Salary: ${job_info['salary_min']} - ${job_info['salary_max']}
    - Posted: {job_info['date_posted']}
    - Description: {job_info['description']}
    
    IMPORTANT: Read the ENTIRE job description carefully. Look for experience requirements in sections like:
    - "Requirements", "Qualifications", "What we're looking for"
    - "Minimum X years", "X+ years", "At least X years", "X years of experience"
    - Any mention of "experience", "background", "expertise"
    
    If asking about years of experience:
    - Extract the MINIMUM number mentioned (e.g., "5" from "5+ years")
    - If multiple numbers are mentioned, use the minimum required
    - If no specific number is found, state "No specific experience requirement found"
    - Do NOT return 0 unless explicitly stated as "0 years" or "no experience required"
    
    Response format: Provide only the direct answer to the analysis request.
    """
    
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1
        )
        
        analysis_result = response.choices[0].message.content.strip()
        
        return AIAnalysisResult(
            job_id=job_id,
            job_title=job_info['title'],
            job_company=job_info['company'],
            analysis_result=analysis_result
        )
    except Exception as e:
        return AIAnalysisResult(
            job_id=job_id,
            job_title=job_info['title'],
            job_company=job_info['company'],
            analysis_result=f"Analysis failed: {str(e)}"
        )

async def filter_jobs_with_ai(analyzed_jobs: List[AIAnalysisResult], filter_criteria: str, client) -> List[AIAnalysisResult]:
    """Apply AI filtering to analyzed jobs"""
    if not filter_criteria:
        # If no filtering criteria, return all jobs
        for job in analyzed_jobs:
            job.meets_criteria = True
        return analyzed_jobs
    
    # Prepare all analysis results for batch filtering
    analyses_text = "\n".join([
        f"Job {job.job_id}: {job.job_title} at {job.job_company} - Analysis: {job.analysis_result}"
        for job in analyzed_jobs
    ])
    
    prompt = f"""
    Based on the following job analyses, determine which jobs meet this criteria: "{filter_criteria}"
    
    Job Analyses:
    {analyses_text}
    
    For each job, respond with ONLY the job ID followed by either "YES" or "NO".
    Format: "Job 1: YES" or "Job 1: NO"
    
    Example response:
    Job 1: YES
    Job 2: NO
    Job 3: YES
    """
    
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        filter_result = response.choices[0].message.content.strip()
        
        # Parse the filtering results
        filter_decisions = {}
        for line in filter_result.split('\n'):
            if ':' in line and ('YES' in line or 'NO' in line):
                try:
                    job_part, decision = line.split(':', 1)
                    job_id = int(job_part.strip().replace('Job', '').strip())
                    meets_criteria = 'YES' in decision.upper()
                    filter_decisions[job_id] = meets_criteria
                except:
                    continue
        
        # Apply filtering decisions
        for job in analyzed_jobs:
            job.meets_criteria = filter_decisions.get(job.job_id, False)
        
        return analyzed_jobs
        
    except Exception as e:
        # If filtering fails, mark all as not meeting criteria
        for job in analyzed_jobs:
            job.meets_criteria = False
        return analyzed_jobs

@app.post("/ai-filter-jobs", response_model=AIFilterResponse)
async def ai_filter_jobs(request: AIFilterRequest, http_request: Request):
    """Apply AI-powered analysis and filtering to job search results"""
    try:
        # Get API key from request header or fallback to environment
        api_key = http_request.headers.get("X-OpenAI-API-Key") or OPENAI_API_KEY
        
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key is required. Please provide it in the X-OpenAI-API-Key header or configure OPENAI_API_KEY in your environment."
            )
        
        # Initialize OpenAI client with the provided API key
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        start_time = datetime.now()
        original_count = len(request.jobs)
        
        if original_count == 0:
            return AIFilterResponse(
                success=True,
                message="No jobs provided for analysis",
                original_count=0,
                analyzed_jobs=[],
                timestamp=start_time.isoformat()
            )
        
        print(f"Starting AI analysis of {original_count} jobs...")
        print(f"Analysis prompt: {request.analysis_prompt}")
        if request.filter_criteria:
            print(f"Filter criteria: {request.filter_criteria}")
        
        # Step 1: Analyze each job with AI
        analysis_tasks = [
            analyze_job_with_ai(job, request.analysis_prompt, i, client)
            for i, job in enumerate(request.jobs)
        ]
        
        # Process in batches to avoid rate limits (adjust batch size as needed)
        batch_size = 5
        analyzed_jobs = []
        
        for i in range(0, len(analysis_tasks), batch_size):
            batch = analysis_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            analyzed_jobs.extend(batch_results)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(analysis_tasks):
                await asyncio.sleep(1)
        
        print(f"Completed analysis of {len(analyzed_jobs)} jobs")
        
        # Step 2: Apply filtering if criteria provided
        filtered_jobs = None
        filtered_count = None
        
        if request.filter_criteria:
            print("Applying AI filtering...")
            analyzed_jobs = await filter_jobs_with_ai(analyzed_jobs, request.filter_criteria, client)
            
            # Extract jobs that meet criteria
            jobs_meeting_criteria = [
                job for job in analyzed_jobs if job.meets_criteria
            ]
            
            if jobs_meeting_criteria:
                filtered_jobs = [
                    request.jobs[job.job_id] for job in jobs_meeting_criteria
                ]
                filtered_count = len(filtered_jobs)
            else:
                filtered_jobs = []
                filtered_count = 0
            
            print(f"Filtered to {filtered_count} jobs meeting criteria")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        message = f"Successfully analyzed {original_count} jobs in {duration:.1f} seconds"
        if filtered_count is not None:
            message += f" and filtered to {filtered_count} jobs"
        
        return AIFilterResponse(
            success=True,
            message=message,
            original_count=original_count,
            analyzed_jobs=analyzed_jobs,
            filtered_count=filtered_count,
            filtered_jobs=filtered_jobs,
            timestamp=end_time.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in AI filtering: {str(e)}"
        )

# User Saved Jobs Endpoints

@app.post("/user/save-job")
async def save_job_for_user(
    request: NewSaveJobRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save a job for the authenticated user"""
    saved_job = UserService.save_job(db, current_user.id, request)
    return {
        "success": True,
        "message": "Job saved successfully",
        "saved_job": NewSavedJobResponse.from_orm(saved_job)
    }

@app.get("/user/saved-jobs")
async def get_user_saved_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get saved jobs for the authenticated user"""
    saved_jobs = UserService.get_saved_jobs(db, current_user.id, skip, limit)
    return {
        "success": True,
        "message": f"Retrieved {len(saved_jobs)} saved jobs",
        "saved_jobs": [NewSavedJobResponse.from_orm(job) for job in saved_jobs],
        "total_count": len(saved_jobs),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/user/saved-jobs/categorized")
async def get_user_saved_jobs_categorized(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get saved jobs categorized by status for the authenticated user"""
    categorized_jobs = UserService.get_categorized_jobs(db, current_user.id)
    
    # Convert to response format
    categorized_response = {}
    for category, jobs in categorized_jobs.items():
        categorized_response[category] = [NewSavedJobResponse.from_orm(job) for job in jobs]
    
    return {
        "success": True,
        "message": f"Retrieved categorized saved jobs",
        "saved_jobs": categorized_response,
        "counts": {category: len(jobs) for category, jobs in categorized_jobs.items()},
        "timestamp": datetime.now().isoformat()
    }

@app.put("/user/saved-job/{job_id}")
async def update_user_saved_job(
    job_id: str,
    job_update: SavedJobUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a saved job for the authenticated user"""
    updated_job = UserService.update_saved_job(db, current_user.id, job_id, job_update)
    if not updated_job:
        raise HTTPException(status_code=404, detail="Saved job not found")
    return {
        "success": True,
        "message": "Job updated successfully",
        "saved_job": NewSavedJobResponse.from_orm(updated_job)
    }

@app.delete("/user/saved-job/{job_id}")
async def delete_user_saved_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a saved job for the authenticated user"""
    deleted = UserService.delete_saved_job(db, current_user.id, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved job not found")
    return {
        "success": True,
        "message": "Job deleted successfully"
    }

# User Search History Endpoints

@app.get("/user/search-history")
async def get_user_search_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get search history for the authenticated user"""
    search_history = UserService.get_search_history(db, current_user.id, skip, limit)
    return {
        "success": True,
        "message": f"Retrieved {len(search_history)} search history entries",
        "search_history": [SearchHistoryResponse.from_orm(entry) for entry in search_history],
        "timestamp": datetime.now().isoformat()
    }

# User Saved Searches Endpoints

@app.post("/user/saved-searches", response_model=SavedSearchResponse)
async def create_saved_search(
    search_create: SavedSearchCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a saved search template for the authenticated user"""
    saved_search = UserService.create_saved_search(db, current_user.id, search_create)
    return SavedSearchResponse.from_orm(saved_search)

@app.get("/user/saved-searches")
async def get_saved_searches(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get saved search templates for the authenticated user"""
    saved_searches = UserService.get_saved_searches(db, current_user.id)
    return {
        "success": True,
        "message": f"Retrieved {len(saved_searches)} saved searches",
        "saved_searches": [SavedSearchResponse.from_orm(search) for search in saved_searches],
        "timestamp": datetime.now().isoformat()
    }

@app.put("/user/saved-searches/{search_id}")
async def update_saved_search(
    search_id: str,
    search_update: SavedSearchUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a saved search template for the authenticated user"""
    updated_search = UserService.update_saved_search(db, current_user.id, search_id, search_update)
    if not updated_search:
        raise HTTPException(status_code=404, detail="Saved search not found")
    return {
        "success": True,
        "message": "Saved search updated successfully",
        "saved_search": SavedSearchResponse.from_orm(updated_search)
    }

@app.delete("/user/saved-searches/{search_id}")
async def delete_saved_search(
    search_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a saved search template for the authenticated user"""
    deleted = UserService.delete_saved_search(db, current_user.id, search_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved search not found")
    return {
        "success": True,
        "message": "Saved search deleted successfully"
    }

# Legacy Saved Jobs Endpoints (for backwards compatibility)

@app.post("/save-job", response_model=SavedJobResponse)
async def save_job(request: SaveJobRequest):
    """Save a job to the user's collection"""
    try:
        # Load current saved jobs
        saved_jobs = load_saved_jobs()
        
        # Check if job is already saved
        if job_already_saved(request.job_data, saved_jobs):
            return SavedJobResponse(
                success=False,
                message="Job is already saved to your collection"
            )
        
        # Create new saved job
        new_saved_job = SavedJob(
            id=str(uuid.uuid4()),
            job_data=request.job_data,
            notes=request.notes,
            saved_at=datetime.now().isoformat(),
            tags=[]
        )
        
        # Add to list and save
        saved_jobs.append(new_saved_job)
        save_jobs_to_file(saved_jobs)
        
        return SavedJobResponse(
            success=True,
            message="Job saved successfully",
            saved_job=new_saved_job
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving job: {str(e)}"
        )

@app.get("/saved-jobs", response_model=SavedJobsListResponse)
async def get_saved_jobs():
    """Get all saved jobs"""
    try:
        saved_jobs = load_saved_jobs()
        # Sort by saved_at date (newest first)
        saved_jobs.sort(key=lambda x: x.saved_at, reverse=True)
        return SavedJobsListResponse(
            success=True,
            message=f"Retrieved {len(saved_jobs)} saved jobs",
            saved_jobs=saved_jobs,
            total_count=len(saved_jobs),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"Error in /saved-jobs endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving saved jobs: {str(e)}"
        )

@app.delete("/saved-job/{job_id}")
async def delete_saved_job(job_id: str):
    """Delete a saved job by ID"""
    try:
        saved_jobs = load_saved_jobs()
        
        # Find and remove the job
        original_count = len(saved_jobs)
        saved_jobs = [job for job in saved_jobs if job.id != job_id]
        
        if len(saved_jobs) == original_count:
            raise HTTPException(
                status_code=404,
                detail="Saved job not found"
            )
        
        # Save updated list
        save_jobs_to_file(saved_jobs)
        
        return {
            "success": True,
            "message": "Job removed from saved collection",
            "remaining_count": len(saved_jobs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting saved job: {str(e)}"
        )

@app.put("/saved-job/{job_id}/notes")
async def update_job_notes(job_id: str, notes: str):
    """Update notes for a saved job"""
    try:
        saved_jobs = load_saved_jobs()
        
        # Find and update the job
        job_found = False
        for job in saved_jobs:
            if job.id == job_id:
                job.notes = notes
                job_found = True
                break
        
        if not job_found:
            raise HTTPException(
                status_code=404,
                detail="Saved job not found"
            )
        
        # Save updated list
        save_jobs_to_file(saved_jobs)
        
        return {
            "success": True,
            "message": "Job notes updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating job notes: {str(e)}"
        )

@app.put("/saved-job/{job_id}/applied")
async def mark_job_applied(job_id: str, applied: bool = Query(...)):
    """Mark a saved job as applied or not applied"""
    try:
        saved_jobs = load_saved_jobs()
        
        # Find and update the job
        job_found = False
        for job in saved_jobs:
            if job.id == job_id:
                job.applied = applied
                job.applied_at = datetime.now().isoformat() if applied else None
                job_found = True
                break
        
        if not job_found:
            raise HTTPException(
                status_code=404,
                detail="Saved job not found"
            )
        
        # Save updated list
        save_jobs_to_file(saved_jobs)
        
        status_text = "applied to" if applied else "marked as not applied"
        
        return {
            "success": True,
            "message": f"Job {status_text} successfully",
            "applied": applied,
            "applied_at": job.applied_at if applied else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating job application status: {str(e)}"
        )

@app.put("/saved-job/{job_id}/save-for-later")
async def mark_job_save_for_later(job_id: str, save_for_later: bool = Query(...)):
    """Mark a saved job as save for later or not"""
    try:
        saved_jobs = load_saved_jobs()
        job_found = False
        for job in saved_jobs:
            if job.id == job_id:
                job.save_for_later = save_for_later
                job_found = True
                break
        if not job_found:
            raise HTTPException(status_code=404, detail="Saved job not found")
        save_jobs_to_file(saved_jobs)
        status_text = "saved for later" if save_for_later else "removed from save for later"
        return {
            "success": True,
            "message": f"Job {status_text} successfully",
            "save_for_later": save_for_later
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating save for later status: {str(e)}")

@app.put("/saved-job/{job_id}/not-interested")
async def mark_job_not_interested(job_id: str, not_interested: bool = Query(...)):
    """Mark a saved job as not interested or not"""
    try:
        saved_jobs = load_saved_jobs()
        job_found = False
        for job in saved_jobs:
            if job.id == job_id:
                job.not_interested = not_interested
                job_found = True
                break
        if not job_found:
            raise HTTPException(status_code=404, detail="Saved job not found")
        save_jobs_to_file(saved_jobs)
        status_text = "marked as not interested" if not_interested else "removed from not interested"
        return {
            "success": True,
            "message": f"Job {status_text} successfully",
            "not_interested": not_interested
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating not interested status: {str(e)}")

@app.get("/saved-jobs/categorized")
async def get_saved_jobs_categorized():
    """Get saved jobs organized by application status"""
    try:
        saved_jobs = load_saved_jobs()
        saved_jobs.sort(key=lambda x: x.saved_at, reverse=True)
        saved_not_applied = [job for job in saved_jobs if not job.applied and not job.save_for_later and not job.not_interested]
        save_for_later_jobs = [job for job in saved_jobs if job.save_for_later and not job.not_interested]
        applied_jobs = [job for job in saved_jobs if job.applied and not job.not_interested]
        not_interested_jobs = [job for job in saved_jobs if job.not_interested]
        applied_jobs.sort(key=lambda x: x.applied_at or x.saved_at, reverse=True)
        return {
            "success": True,
            "message": f"Retrieved {len(saved_jobs)} saved jobs",
            "saved_jobs": {
                "saved_not_applied": saved_not_applied,
                "save_for_later": save_for_later_jobs,
                "applied": applied_jobs,
                "not_interested": not_interested_jobs
            },
            "counts": {
                "total": len(saved_jobs),
                "saved_not_applied": len(saved_not_applied),
                "save_for_later": len(save_for_later_jobs),
                "applied": len(applied_jobs),
                "not_interested": len(not_interested_jobs)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving categorized saved jobs: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print(f"Starting JobSpy API server on {BACKEND_HOST}:{BACKEND_PORT}")
    print(f"API Documentation: http://localhost:{BACKEND_PORT}/docs")
    print(f"Health Check: http://localhost:{BACKEND_PORT}/health")
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, log_level="info" if DEBUG else "warning") 