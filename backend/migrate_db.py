#!/usr/bin/env python3
"""
Database migration script to add new columns to user_preferences table
"""
import sqlite3
import os

DATABASE_PATH = "jobsearch.db"

def migrate_database():
    """Add new columns to user_preferences table"""
    
    # Check if database exists
    if not os.path.exists(DATABASE_PATH):
        print(f"Database {DATABASE_PATH} not found. Creating new database...")
        # If no database exists, the create_tables() function will handle everything
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if new columns already exist
        cursor.execute("PRAGMA table_info(user_preferences)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Define all required columns with their types
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
        
        columns_to_add = []
        for col_name, col_type in required_columns.items():
            if col_name not in columns:
                columns_to_add.append((col_name, col_type))
        
        # Add missing columns
        for column_name, column_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE user_preferences ADD COLUMN {column_name} {column_type}")
                print(f"✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"⚠️  Column {column_name} already exists")
                else:
                    print(f"❌ Error adding column {column_name}: {e}")
        
        conn.commit()
        
        if columns_to_add:
            print(f"✅ Successfully migrated database with {len(columns_to_add)} new columns")
        else:
            print("ℹ️  No migration needed - all columns already exist")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("🔄 Starting database migration...")
    migrate_database()
    print("✅ Migration complete!")