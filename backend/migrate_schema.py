#!/usr/bin/env python3
"""
Database Schema Migration Script
Adds missing columns to production database:
 - scraping_runs.search_analytics (JSON)
 - scraping_runs.current_progress (JSON)
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from database import DATABASE_URL

def migrate_database():
    """Add missing columns to production database"""
    
    print(f"🔍 Connecting to database...")
    print(f"🌐 Database URL: {DATABASE_URL[:50]}...")
    
    # Convert postgresql:// to postgresql+psycopg:// if needed
    db_url = DATABASE_URL
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    
    try:
        engine = create_engine(db_url)
        
        # Check current schema
        inspector = inspect(engine)
        
        if 'scraping_runs' not in inspector.get_table_names():
            print("❌ scraping_runs table does not exist!")
            return False
            
        columns = [col['name'] for col in inspector.get_columns('scraping_runs')]
        print(f"📋 Current columns: {columns}")

        added_any = False

        # Add search_analytics if missing
        if 'search_analytics' not in columns:
            print("🔧 Adding search_analytics column...")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE scraping_runs ADD COLUMN search_analytics JSON"))
                conn.commit()
            print("✅ Successfully added search_analytics column")
            added_any = True
        else:
            print("✅ search_analytics column already exists")

        # Add current_progress if missing
        columns = [col['name'] for col in inspector.get_columns('scraping_runs')]
        if 'current_progress' not in columns:
            print("🔧 Adding current_progress column...")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE scraping_runs ADD COLUMN current_progress JSON"))
                conn.commit()
            print("✅ Successfully added current_progress column")
            added_any = True
        else:
            print("✅ current_progress column already exists")

        # Verify
        updated_columns = [col['name'] for col in inspector.get_columns('scraping_runs')]
        expected = {'search_analytics', 'current_progress'}
        if expected.issubset(set(updated_columns)):
            print("✅ All required columns present")
            return True
        else:
            print("❌ Column verification failed")
            print(f"📋 Final columns: {updated_columns}")
            return False
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database schema migration...")
    success = migrate_database()
    if success:
        print("🎉 Migration completed successfully!")
        sys.exit(0)
    else:
        print("💥 Migration failed!")
        sys.exit(1)