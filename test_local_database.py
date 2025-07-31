#!/usr/bin/env python3
"""
Test Local Database Functionality

This script tests the core functionality of the local job database system:
1. Database schema creation
2. Job deduplication logic
3. Local search functionality
4. Company management
"""

import sys
import os
from datetime import datetime, timedelta

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import create_tables, get_db, TargetCompany, ScrapedJob, create_job_hash
from job_scraper import job_scraper


def test_database_creation():
    """Test database table creation."""
    print("🧪 Testing database creation...")
    try:
        create_tables()
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False


def test_job_hash_function():
    """Test job deduplication hash function."""
    print("\n🧪 Testing job hash function...")
    
    # Test identical jobs
    hash1 = create_job_hash("Software Engineer", "Google", "Mountain View", "https://example.com/job1")
    hash2 = create_job_hash("Software Engineer", "Google", "Mountain View", "https://example.com/job1")
    
    if hash1 == hash2:
        print("✅ Identical jobs produce same hash")
    else:
        print("❌ Identical jobs produce different hashes")
        return False
    
    # Test different jobs
    hash3 = create_job_hash("Data Scientist", "Microsoft", "Seattle", "https://example.com/job2")
    
    if hash1 != hash3:
        print("✅ Different jobs produce different hashes")
    else:
        print("❌ Different jobs produce same hash")
        return False
    
    # Test jobs without URL
    hash4 = create_job_hash("Product Manager", "Apple", "Cupertino", None)
    hash5 = create_job_hash("Product Manager", "Apple", "Cupertino", "")
    
    if hash4 == hash5:
        print("✅ Jobs without URL use title+company+location hash")
    else:
        print("❌ Jobs without URL produce inconsistent hashes")
        return False
    
    return True


def test_target_company_creation():
    """Test creating target companies."""
    print("\n🧪 Testing target company creation...")
    
    db = next(get_db())
    
    try:
        # Create test company
        test_company = TargetCompany(
            name="Test Company",
            display_name="Test Company Inc.",
            preferred_sites=["indeed"],
            search_terms=["software engineer", "developer"],
            location_filters=["USA", "Remote"]
        )
        
        db.add(test_company)
        db.commit()
        
        # Verify it was created
        created_company = db.query(TargetCompany).filter(
            TargetCompany.name == "Test Company"
        ).first()
        
        if created_company:
            print("✅ Target company created successfully")
            print(f"   ID: {created_company.id}")
            print(f"   Name: {created_company.name}")
            print(f"   Search Terms: {created_company.search_terms}")
            return True
        else:
            print("❌ Target company not found after creation")
            return False
            
    except Exception as e:
        print(f"❌ Target company creation failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def test_scraped_job_storage():
    """Test storing scraped jobs with deduplication."""
    print("\n🧪 Testing scraped job storage and deduplication...")
    
    db = next(get_db())
    
    try:
        # Sample job data
        job_data = {
            "title": "Senior Software Engineer",
            "company": "Test Corp",
            "location": "San Francisco, CA",
            "job_url": "https://example.com/test-job",
            "site": "indeed",
            "description": "Looking for a senior software engineer with 5+ years of experience...",
            "job_type": "fulltime",
            "is_remote": False,
            "min_amount": 120000,
            "max_amount": 180000,
            "interval": "yearly",
            "currency": "USD"
        }
        
        # Store job using the scraper service
        new_jobs, duplicates = job_scraper.store_jobs_in_database([job_data], db)
        
        if new_jobs == 1 and duplicates == 0:
            print("✅ First job stored successfully")
        else:
            print(f"❌ Expected 1 new job, got {new_jobs}")
            return False
        
        # Try to store the same job again (should be duplicate)
        new_jobs2, duplicates2 = job_scraper.store_jobs_in_database([job_data], db)
        
        if new_jobs2 == 0 and duplicates2 == 1:
            print("✅ Duplicate job correctly detected and skipped")
        else:
            print(f"❌ Expected 0 new jobs and 1 duplicate, got {new_jobs2} new and {duplicates2} duplicates")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Job storage test failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def test_local_search():
    """Test local job search functionality."""
    print("\n🧪 Testing local job search...")
    
    db = next(get_db())
    
    try:
        # Search for jobs we just stored
        jobs, total_count = job_scraper.search_local_jobs(
            db=db,
            search_term="software engineer",
            company_names=["Test Corp"],
            days_old=1,  # Last day
            limit=10
        )
        
        if total_count > 0:
            print(f"✅ Found {total_count} jobs in local search")
            print(f"   First job: {jobs[0].title} at {jobs[0].company}")
            return True
        else:
            print("❌ No jobs found in local search")
            return False
            
    except Exception as e:
        print(f"❌ Local search test failed: {e}")
        return False
    finally:
        db.close()


def test_experience_extraction():
    """Test experience years extraction from job descriptions."""
    print("\n🧪 Testing experience extraction...")
    
    test_cases = [
        ("Looking for 5+ years of experience", 5, None),
        ("Minimum 3 years experience required", 3, None),
        ("2-4 years of relevant experience", 2, 4),
        ("At least 7 years in software development", 7, None),
        ("No experience required", None, None),
        ("Must have 10 years experience", 10, None),
    ]
    
    passed = 0
    for description, expected_min, expected_max in test_cases:
        min_exp, max_exp = job_scraper.extract_experience_years(description)
        
        if min_exp == expected_min and max_exp == expected_max:
            print(f"✅ '{description}' → min: {min_exp}, max: {max_exp}")
            passed += 1
        else:
            print(f"❌ '{description}' → expected min: {expected_min}, max: {expected_max}, got min: {min_exp}, max: {max_exp}")
    
    if passed == len(test_cases):
        print("✅ All experience extraction tests passed")
        return True
    else:
        print(f"❌ {len(test_cases) - passed} experience extraction tests failed")
        return False


def main():
    """Run all tests."""
    print("🧪 JobSpy Local Database Test Suite")
    print("=" * 50)
    
    tests = [
        ("Database Creation", test_database_creation),
        ("Job Hash Function", test_job_hash_function),
        ("Target Company Creation", test_target_company_creation),
        ("Job Storage & Deduplication", test_scraped_job_storage),
        ("Local Search", test_local_search),
        ("Experience Extraction", test_experience_extraction),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_function in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        
        try:
            if test_function():
                passed += 1
                print(f"🎉 {test_name} PASSED")
            else:
                failed += 1
                print(f"💥 {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your local database is ready to use.")
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the setup.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)