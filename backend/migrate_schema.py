#!/usr/bin/env python3
"""
Database Schema Migration Script
Adds the search_analytics column to production database
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from database import DATABASE_URL

def migrate_database():
    """Add missing search_analytics column to production database"""
    
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
        
        if 'search_analytics' in columns:
            print("✅ search_analytics column already exists")
            return True
            
        # Add the missing column
        print("🔧 Adding search_analytics column...")
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE scraping_runs ADD COLUMN search_analytics JSON"))
            conn.commit()
            
        print("✅ Successfully added search_analytics column")
        
        # Verify the column was added
        updated_columns = [col['name'] for col in inspector.get_columns('scraping_runs')]
        if 'search_analytics' in updated_columns:
            print("✅ Column addition verified")
            return True
        else:
            print("❌ Column addition failed verification")
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