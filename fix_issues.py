#!/usr/bin/env python3
"""
Fix script for database migration and bcrypt compatibility issues
"""

import sqlite3
import os
import sys
import subprocess

def fix_database_schema():
    """Fix the database schema by adding missing columns"""
    
    # Change to backend directory to ensure we're working with the right database
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
    
    database_path = "jobsearch.db"
    
    # Check if database exists
    if not os.path.exists(database_path):
        print(f"Database {database_path} not found. It will be created automatically.")
        return True
    
    print(f"🔍 Checking database schema in {database_path}...")
    
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    try:
        # Check if user_preferences table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("⚠️  user_preferences table doesn't exist. It will be created automatically.")
            return True
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(user_preferences)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 Current columns: {columns}")
        
        # Define required columns with their types
        required_columns = {
            'default_search_term': 'TEXT',
            'default_company_filter': 'TEXT', 
            'default_exclude_keywords': 'TEXT',
            'default_max_experience': 'INTEGER',
            'min_salary': 'INTEGER',
            'max_salary': 'INTEGER',
            'salary_currency': 'TEXT DEFAULT "USD"',
            'email_notifications': 'BOOLEAN DEFAULT 1',
            'job_alert_frequency': 'TEXT DEFAULT "daily"',
            'jobs_per_page': 'INTEGER DEFAULT 20',
            'default_sort': 'TEXT DEFAULT "date_posted"',
            'updated_at': 'DATETIME'
        }
        
        missing_columns = []
        for col_name, col_type in required_columns.items():
            if col_name not in columns:
                missing_columns.append((col_name, col_type))
        
        if not missing_columns:
            print("✅ All required columns exist!")
            return True
        
        print(f"🔧 Adding {len(missing_columns)} missing columns...")
        
        # Add missing columns one by one
        for col_name, col_type in missing_columns:
            try:
                cursor.execute(f"ALTER TABLE user_preferences ADD COLUMN {col_name} {col_type}")
                print(f"  ✅ Added {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"  ⚠️  {col_name} already exists")
                else:
                    print(f"  ❌ Error adding {col_name}: {e}")
                    return False
        
        conn.commit()
        print("✅ Database schema migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def fix_bcrypt_issue():
    """Fix bcrypt compatibility issue"""
    print("🔧 Fixing bcrypt compatibility issue...")
    
    try:
        # Try to upgrade bcrypt to a compatible version
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "bcrypt>=4.0.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ bcrypt upgraded successfully!")
            return True
        else:
            print(f"⚠️  bcrypt upgrade had issues: {result.stderr}")
            
            # Try alternative fix: install specific compatible version
            result2 = subprocess.run([
                sys.executable, "-m", "pip", "install", "bcrypt==4.1.2"
            ], capture_output=True, text=True)
            
            if result2.returncode == 0:
                print("✅ bcrypt installed with specific version!")
                return True
            else:
                print(f"❌ Failed to fix bcrypt: {result2.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Error fixing bcrypt: {e}")
        return False

def verify_dependencies():
    """Verify all required dependencies are installed"""
    print("🔍 Verifying dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'python-jobspy', 'pandas', 'pydantic',
        'requests', 'python-dotenv', 'sqlalchemy', 'python-jose',
        'passlib', 'python-multipart', 'email-validator', 'bcrypt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All missing packages installed!")
                return True
            else:
                print(f"❌ Package installation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
            return False
    else:
        print("✅ All dependencies are installed!")
        return True

def main():
    """Main function to run all fixes"""
    print("=" * 60)
    print("🔧 JobSpy Backend Fix Script")
    print("=" * 60)
    
    # Step 1: Fix database schema
    print("\n1️⃣  Fixing database schema...")
    db_fixed = fix_database_schema()
    
    # Step 2: Fix bcrypt issue  
    print("\n2️⃣  Fixing bcrypt compatibility...")
    bcrypt_fixed = fix_bcrypt_issue()
    
    # Step 3: Verify dependencies
    print("\n3️⃣  Verifying dependencies...")
    deps_ok = verify_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Database Schema: {'✅ FIXED' if db_fixed else '❌ FAILED'}")
    print(f"bcrypt Issue: {'✅ FIXED' if bcrypt_fixed else '❌ FAILED'}")
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ ISSUES'}")
    
    if db_fixed and bcrypt_fixed and deps_ok:
        print("\n🎉 ALL ISSUES FIXED! You can now start the backend server.")
        print("💡 Run: python backend/main.py")
    else:
        print("\n⚠️  Some issues remain. Please check the output above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()