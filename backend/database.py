from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import uuid
import hashlib

DATABASE_URL = "sqlite:///./jobsearch.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    saved_jobs = relationship("UserSavedJob", back_populates="user")
    search_history = relationship("SearchHistory", back_populates="user")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Default search preferences
    default_sites = Column(JSON, default=["indeed"])  # List of preferred job sites
    default_search_term = Column(String)  # Default job title/search term
    default_company_filter = Column(String)  # Default company filter
    default_location = Column(String, default="USA")
    default_distance = Column(Integer, default=50)
    default_job_type = Column(String)  # full-time, part-time, contract, etc.
    default_remote = Column(Boolean)
    default_results_wanted = Column(Integer, default=100)
    default_hours_old = Column(Integer, default=168)  # 1 week
    default_country = Column(String, default="USA")
    default_max_experience = Column(Integer)
    default_exclude_keywords = Column(String)  # Comma-separated keywords to exclude from job titles
    
    # Salary preferences
    min_salary = Column(Integer)
    max_salary = Column(Integer)
    salary_currency = Column(String, default="USD")
    
    # Notification preferences
    email_notifications = Column(Boolean, default=True)
    job_alert_frequency = Column(String, default="daily")  # daily, weekly, none
    
    # UI preferences
    jobs_per_page = Column(Integer, default=20)
    default_sort = Column(String, default="date_posted")  # date_posted, relevance, salary
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="preferences")

class UserSavedJob(Base):
    __tablename__ = "user_saved_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Job data (stored as JSON)
    job_data = Column(JSON, nullable=False)
    
    # User-specific job metadata
    notes = Column(Text)
    tags = Column(JSON, default=[])  # List of user-defined tags
    
    # Job status tracking
    applied = Column(Boolean, default=False)
    applied_at = Column(DateTime(timezone=True))
    save_for_later = Column(Boolean, default=False)
    not_interested = Column(Boolean, default=False)
    interview_scheduled = Column(Boolean, default=False)
    interview_date = Column(DateTime(timezone=True))
    
    # Application tracking
    application_status = Column(String)  # applied, interview, rejected, offer, accepted
    application_notes = Column(Text)
    follow_up_date = Column(DateTime(timezone=True))
    
    # Timestamps
    saved_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="saved_jobs")

class SearchHistory(Base):
    __tablename__ = "search_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Search parameters
    search_term = Column(String, nullable=False)
    sites = Column(JSON)
    location = Column(String)
    distance = Column(Integer)
    job_type = Column(String)
    is_remote = Column(Boolean)
    results_wanted = Column(Integer)
    company_filter = Column(String)
    
    # Search results metadata
    results_count = Column(Integer)
    search_duration = Column(Integer)  # in seconds
    
    # Timestamps
    searched_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="search_history")

class SavedSearch(Base):
    __tablename__ = "saved_searches"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Search template
    name = Column(String, nullable=False)  # User-defined name for the search
    description = Column(Text)
    
    # Search parameters
    search_term = Column(String, nullable=False)
    sites = Column(JSON)
    location = Column(String)
    distance = Column(Integer)
    job_type = Column(String)
    is_remote = Column(Boolean)
    results_wanted = Column(Integer)
    company_filter = Column(String)
    
    # Alert settings
    is_alert_active = Column(Boolean, default=False)
    alert_frequency = Column(String, default="daily")  # daily, weekly
    last_alert_sent = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

class TargetCompany(Base):
    __tablename__ = "target_companies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)  # Company name for searching
    display_name = Column(String)  # How to display the company name
    is_active = Column(Boolean, default=True)
    
    # Scraping preferences for this company
    preferred_sites = Column(JSON, default=["indeed"])  # Which sites to scrape
    search_terms = Column(JSON, default=[])  # Additional search terms for this company
    location_filters = Column(JSON, default=["USA"])  # Locations to search
    
    # Metadata
    last_scraped = Column(DateTime(timezone=True))
    total_jobs_found = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    scraped_jobs = relationship("ScrapedJob", back_populates="target_company")

class ScrapedJob(Base):
    __tablename__ = "scraped_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Deduplication fields
    job_url = Column(String, index=True)  # Primary deduplication key
    job_hash = Column(String, unique=True, nullable=False, index=True)  # Hash for deduplication
    
    # Core job information
    title = Column(String, nullable=False, index=True)
    company = Column(String, nullable=False, index=True)
    location = Column(String, index=True)
    site = Column(String, nullable=False)  # indeed, linkedin, etc.
    
    # Job details
    description = Column(Text)
    job_type = Column(String)  # fulltime, parttime, etc.
    is_remote = Column(Boolean)
    
    # Salary information
    min_amount = Column(Float)
    max_amount = Column(Float)
    salary_interval = Column(String)  # yearly, monthly, hourly
    currency = Column(String)
    
    # Metadata
    date_posted = Column(DateTime(timezone=True))
    date_scraped = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)  # For soft deletion
    
    # Experience requirements (extracted from description)
    min_experience_years = Column(Integer)
    max_experience_years = Column(Integer)
    
    # Relationships
    target_company_id = Column(String, ForeignKey("target_companies.id"))
    target_company = relationship("TargetCompany", back_populates="scraped_jobs")
    scraping_run_id = Column(String, ForeignKey("scraping_runs.id"))
    scraping_run = relationship("ScrapingRun", back_populates="jobs")
    
    # Additional indexes for fast searching
    __table_args__ = (
        Index('idx_job_search', 'title', 'company', 'location'),
        Index('idx_job_date', 'date_posted', 'is_active'),
        Index('idx_job_salary', 'min_amount', 'max_amount'),
        Index('idx_job_experience', 'min_experience_years', 'max_experience_years'),
    )

class ScrapingRun(Base):
    __tablename__ = "scraping_runs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Run metadata
    run_type = Column(String, nullable=False)  # 'scheduled', 'manual', 'company_specific'
    status = Column(String, default='running')  # 'running', 'completed', 'failed'
    
    # Run parameters
    companies_scraped = Column(JSON)  # List of company IDs scraped
    sites_used = Column(JSON)  # Sites used for scraping
    search_parameters = Column(JSON)  # Full search parameters
    
    # Results
    total_jobs_found = Column(Integer, default=0)
    new_jobs_added = Column(Integer, default=0)
    duplicate_jobs_skipped = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    
    # Error handling
    error_message = Column(Text)
    
    # Relationships
    jobs = relationship("ScrapedJob", back_populates="scraping_run")

def create_job_hash(title: str, company: str, location: str, job_url: str = None) -> str:
    """Create a hash for job deduplication."""
    # Use job_url if available, otherwise use title+company+location
    if job_url and job_url.strip():
        hash_string = job_url.strip().lower()
    else:
        hash_string = f"{title.strip().lower()}|{company.strip().lower()}|{location.strip().lower()}"
    
    return hashlib.md5(hash_string.encode('utf-8')).hexdigest()

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()