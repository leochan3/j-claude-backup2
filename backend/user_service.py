from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from backend.database import User, UserPreference, UserSavedJob, SearchHistory, SavedSearch
from backend.models import (
    UserCreate, UserPreferencesCreate, UserPreferencesUpdate,
    SaveJobRequest, SavedJobUpdate, SavedSearchCreate, SavedSearchUpdate
)
from backend.auth import get_password_hash
from datetime import datetime
from typing import List, Optional
import uuid

class UserService:
    
    @staticmethod
    def create_user(db: Session, user_create: UserCreate) -> User:
        """Create a new user with default preferences."""
        # Check if username or email already exists
        existing_user = db.query(User).filter(
            (User.username == user_create.username) | (User.email == user_create.email)
        ).first()
        
        if existing_user:
            if existing_user.username == user_create.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Create new user
        hashed_password = get_password_hash(user_create.password)
        db_user = User(
            username=user_create.username,
            email=user_create.email,
            hashed_password=hashed_password,
            full_name=user_create.full_name
        )
        
        try:
            db.add(db_user)
            db.flush()  # Flush to get the user ID
            
            # Create default preferences for the user
            default_preferences = UserPreference(user_id=db_user.id)
            db.add(default_preferences)
            
            db.commit()
            db.refresh(db_user)
            return db_user
            
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user"
            )
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_preferences(db: Session, user_id: str) -> Optional[UserPreference]:
        """Get user preferences."""
        return db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    
    @staticmethod
    def update_user_preferences(
        db: Session, 
        user_id: str, 
        preferences_update: UserPreferencesUpdate
    ) -> UserPreference:
        """Update user preferences."""
        db_preferences = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        
        if not db_preferences:
            # Create default preferences if they don't exist
            db_preferences = UserPreference(user_id=user_id)
            db.add(db_preferences)
        
        # Update only provided fields
        update_data = preferences_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_preferences, field, value)
        
        db_preferences.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_preferences)
        return db_preferences
    
    @staticmethod
    def save_job(db: Session, user_id: str, job_request: SaveJobRequest) -> UserSavedJob:
        """Save a job for a user."""
        # Check if job already saved by this user using job_url or title+company
        job_url = job_request.job_data.get('job_url')
        job_title = job_request.job_data.get('title')
        job_company = job_request.job_data.get('company')
        
        # Query for existing jobs by this user
        existing_jobs = db.query(UserSavedJob).filter(UserSavedJob.user_id == user_id).all()
        
        for existing_job in existing_jobs:
            existing_job_url = existing_job.job_data.get('job_url')
            
            # If both jobs have URLs, only check by URL (most reliable)
            if job_url and existing_job_url:
                if existing_job_url == job_url:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Job already saved"
                    )
            # Only check by title + company if at least one job lacks a URL
            elif (job_title and job_company and 
                  existing_job.job_data.get('title') == job_title and 
                  existing_job.job_data.get('company') == job_company):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Job already saved"
                )
        
        db_job = UserSavedJob(
            user_id=user_id,
            job_data=job_request.job_data,
            notes=job_request.notes,
            tags=job_request.tags
        )
        
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job
    
    @staticmethod
    def get_saved_jobs(
        db: Session, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[UserSavedJob]:
        """Get user's saved jobs."""
        return db.query(UserSavedJob).filter(
            UserSavedJob.user_id == user_id
        ).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_saved_job(db: Session, user_id: str, job_id: str) -> Optional[UserSavedJob]:
        """Get a specific saved job."""
        return db.query(UserSavedJob).filter(
            UserSavedJob.user_id == user_id,
            UserSavedJob.id == job_id
        ).first()
    
    @staticmethod
    def update_saved_job(
        db: Session, 
        user_id: str, 
        job_id: str, 
        job_update: SavedJobUpdate
    ) -> Optional[UserSavedJob]:
        """Update a saved job."""
        db_job = db.query(UserSavedJob).filter(
            UserSavedJob.user_id == user_id,
            UserSavedJob.id == job_id
        ).first()
        
        if not db_job:
            return None
        
        # Update only provided fields
        update_data = job_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == 'applied' and value:
                setattr(db_job, 'applied_at', datetime.utcnow())
            setattr(db_job, field, value)
        
        db_job.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_job)
        return db_job
    
    @staticmethod
    def delete_saved_job(db: Session, user_id: str, job_id: str) -> bool:
        """Delete a saved job."""
        db_job = db.query(UserSavedJob).filter(
            UserSavedJob.user_id == user_id,
            UserSavedJob.id == job_id
        ).first()
        
        if not db_job:
            return False
        
        db.delete(db_job)
        db.commit()
        return True
    
    @staticmethod
    def get_categorized_jobs(db: Session, user_id: str) -> dict:
        """Get jobs categorized by status."""
        all_jobs = db.query(UserSavedJob).filter(UserSavedJob.user_id == user_id).all()
        
        categorized = {
            "all": all_jobs,
            "applied": [job for job in all_jobs if job.applied],
            "save_for_later": [job for job in all_jobs if job.save_for_later],
            "not_interested": [job for job in all_jobs if job.not_interested],
            "interview_scheduled": [job for job in all_jobs if job.interview_scheduled],
            "pending": [job for job in all_jobs if not any([
                job.applied, job.save_for_later, job.not_interested, job.interview_scheduled
            ])]
        }
        
        return categorized
    
    @staticmethod
    def add_search_history(
        db: Session, 
        user_id: str, 
        search_params: dict, 
        results_count: int = 0,
        search_duration: int = 0
    ):
        """Add a search to user's history."""
        search_history = SearchHistory(
            user_id=user_id,
            search_term=search_params.get('search_term', ''),
            sites=search_params.get('site_name', []),
            location=search_params.get('location'),
            distance=search_params.get('distance'),
            job_type=search_params.get('job_type'),
            is_remote=search_params.get('is_remote'),
            results_wanted=search_params.get('results_wanted'),
            company_filter=search_params.get('company_filter'),
            results_count=results_count,
            search_duration=search_duration
        )
        
        db.add(search_history)
        db.commit()
        return search_history
    
    @staticmethod
    def get_search_history(
        db: Session, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 50
    ) -> List[SearchHistory]:
        """Get user's search history."""
        return db.query(SearchHistory).filter(
            SearchHistory.user_id == user_id
        ).order_by(SearchHistory.searched_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def create_saved_search(
        db: Session, 
        user_id: str, 
        search_create: SavedSearchCreate
    ) -> SavedSearch:
        """Create a saved search template."""
        db_search = SavedSearch(
            user_id=user_id,
            **search_create.dict()
        )
        
        db.add(db_search)
        db.commit()
        db.refresh(db_search)
        return db_search
    
    @staticmethod
    def get_saved_searches(db: Session, user_id: str) -> List[SavedSearch]:
        """Get user's saved searches."""
        return db.query(SavedSearch).filter(SavedSearch.user_id == user_id).all()
    
    @staticmethod
    def update_saved_search(
        db: Session, 
        user_id: str, 
        search_id: str, 
        search_update: SavedSearchUpdate
    ) -> Optional[SavedSearch]:
        """Update a saved search."""
        db_search = db.query(SavedSearch).filter(
            SavedSearch.user_id == user_id,
            SavedSearch.id == search_id
        ).first()
        
        if not db_search:
            return None
        
        update_data = search_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_search, field, value)
        
        db_search.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_search)
        return db_search
    
    @staticmethod
    def delete_saved_search(db: Session, user_id: str, search_id: str) -> bool:
        """Delete a saved search."""
        db_search = db.query(SavedSearch).filter(
            SavedSearch.user_id == user_id,
            SavedSearch.id == search_id
        ).first()
        
        if not db_search:
            return False
        
        db.delete(db_search)
        db.commit()
        return True