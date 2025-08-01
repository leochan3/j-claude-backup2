#!/usr/bin/env python3
"""
Test script to verify the fixes are working
"""

import os
import sys
import sqlite3

def test_database_schema():
    """Test if database schema is correct"""
    print("🔍 Testing database schema...")
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
    
    database_path = "jobsearch.db"
    
    if not os.path.exists(database_path):
        print("  ⚠️  Database doesn't exist yet - will be created on first run")
        return True
    
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Check if user_preferences table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
        if not cursor.fetchone():
            print("  ⚠️  user_preferences table doesn't exist yet")
            return True
        
        # Check for required columns
        cursor.execute("PRAGMA table_info(user_preferences)")
        columns = [column[1] for column in cursor.fetchall()]
        
        required_columns = ['default_exclude_keywords', 'default_max_experience', 'min_salary', 'max_salary']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"  ❌ Missing columns: {missing_columns}")
            return False
        else:
            print("  ✅ All required columns present")
            return True
            
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False
    finally:
        conn.close()

def test_bcrypt_compatibility():
    """Test if bcrypt is working correctly"""
    print("🔍 Testing bcrypt compatibility...")
    
    try:
        # Add backend to path
        backend_path = os.path.join(os.path.dirname(__file__), 'backend')
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from auth import get_password_hash, verify_password
        
        # Test password hashing and verification
        test_password = "test123"
        hashed = get_password_hash(test_password)
        
        if not hashed:
            print("  ❌ Password hashing failed")
            return False
        
        if not verify_password(test_password, hashed):
            print("  ❌ Password verification failed")
            return False
        
        if verify_password("wrong_password", hashed):
            print("  ❌ Password verification incorrectly accepted wrong password")
            return False
        
        print("  ✅ bcrypt working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ bcrypt test failed: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    required_imports = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'uvicorn'),
        ('pandas', 'pd'),
        ('sqlalchemy', 'sqlalchemy'),
        ('passlib', 'passlib'),
        ('bcrypt', 'bcrypt'),
        ('jose', 'jose'),
        ('pydantic', 'pydantic')
    ]
    
    failed_imports = []
    
    for module_name, alias in required_imports:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"  ❌ Failed to import: {failed_imports}")
        return False
    else:
        print("  ✅ All modules imported successfully")
        return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 Running Fix Verification Tests")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Schema", test_database_schema),
        ("bcrypt Compatibility", test_bcrypt_compatibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Backend should start successfully.")
        print("💡 Next steps:")
        print("   1. Run: python backend/main.py")
        print("   2. Open browser to: http://localhost:8000/docs")
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the output above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()