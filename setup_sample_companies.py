#!/usr/bin/env python3
"""
Setup script to add sample companies to the database for testing admin interfaces
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

# Change to backend directory for database access
os.chdir(backend_path)

from database import SessionLocal, TargetCompany, create_tables
from datetime import datetime

def setup_sample_companies():
    """Add sample companies to the database"""
    
    # Ensure tables exist
    create_tables()
    
    # Sample companies with realistic data
    sample_companies = [
        {
            "name": "Google",
            "display_name": "Google",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "data scientist", "product manager", "swe"],
            "location_filters": ["USA", "Remote", "Mountain View, CA"]
        },
        {
            "name": "Microsoft",
            "display_name": "Microsoft",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "program manager", "data engineer"],
            "location_filters": ["USA", "Remote", "Redmond, WA", "Seattle, WA"]
        },
        {
            "name": "Amazon",
            "display_name": "Amazon",
            "preferred_sites": ["indeed"],
            "search_terms": ["software engineer", "sde", "software development engineer"],
            "location_filters": ["USA", "Remote", "Seattle, WA"]
        },
        {
            "name": "Apple",
            "display_name": "Apple",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "ios developer", "machine learning"],
            "location_filters": ["USA", "Remote", "Cupertino, CA"]
        },
        {
            "name": "Meta",
            "display_name": "Meta (Facebook)",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "data scientist", "product manager"],
            "location_filters": ["USA", "Remote", "Menlo Park, CA"]
        },
        {
            "name": "Netflix",
            "display_name": "Netflix",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "data engineer", "machine learning"],
            "location_filters": ["USA", "Remote", "Los Gatos, CA"]
        },
        {
            "name": "Uber",
            "display_name": "Uber",
            "preferred_sites": ["indeed"],
            "search_terms": ["software engineer", "data scientist", "product manager"],
            "location_filters": ["USA", "Remote", "San Francisco, CA"]
        },
        {
            "name": "Lyft",
            "display_name": "Lyft",
            "preferred_sites": ["indeed"],
            "search_terms": ["software engineer", "data scientist", "product manager"],
            "location_filters": ["USA", "Remote", "San Francisco, CA"]
        },
        {
            "name": "Airbnb",
            "display_name": "Airbnb",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "data scientist", "product manager"],
            "location_filters": ["USA", "Remote", "San Francisco, CA"]
        },
        {
            "name": "Stripe",
            "display_name": "Stripe",
            "preferred_sites": ["indeed", "linkedin"],
            "search_terms": ["software engineer", "product manager", "data engineer"],
            "location_filters": ["USA", "Remote", "San Francisco, CA"]
        }
    ]
    
    db = SessionLocal()
    
    try:
        added_count = 0
        updated_count = 0
        
        for company_data in sample_companies:
            # Check if company already exists
            existing = db.query(TargetCompany).filter(
                TargetCompany.name.ilike(f"%{company_data['name']}%")
            ).first()
            
            if existing:
                # Update existing company
                for key, value in company_data.items():
                    if key != 'name':  # Don't change the name
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
                updated_count += 1
                print(f"✅ Updated: {company_data['name']}")
            else:
                # Add new company
                new_company = TargetCompany(**company_data)
                db.add(new_company)
                added_count += 1
                print(f"➕ Added: {company_data['name']}")
        
        db.commit()
        
        print(f"\n🎉 Setup complete!")
        print(f"📊 Added: {added_count} new companies")
        print(f"🔄 Updated: {updated_count} existing companies")
        print(f"📈 Total companies in database: {db.query(TargetCompany).filter(TargetCompany.is_active == True).count()}")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Start the backend: python main.py")
        print(f"   2. Open database_viewer.html to see the companies")
        print(f"   3. Use scraping_interface.html to scrape jobs for these companies")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error setting up companies: {e}")
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    print("🏢 Setting up sample companies for admin interfaces...")
    print("=" * 60)
    
    success = setup_sample_companies()
    
    if success:
        print("=" * 60)
        print("✅ Sample companies setup completed successfully!")
    else:
        print("❌ Setup failed. Check the error messages above.")
        sys.exit(1)