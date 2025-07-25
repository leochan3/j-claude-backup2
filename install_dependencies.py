#!/usr/bin/env python3
"""
Install script for JobSpy account system dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    print("🔧 Installing JobSpy Account System Dependencies")
    print("=" * 50)
    
    # Required packages for the account system
    packages = [
        "sqlalchemy==2.0.23",
        "python-jose[cryptography]==3.3.0", 
        "passlib[bcrypt]==1.7.4",
        "python-multipart==0.0.6"
    ]
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    success_count = 0
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
            success_count += 1
        else:
            print(f"❌ Failed to install {package}")
    
    print(f"\n📊 Installation Summary:")
    print(f"   Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("\n🎉 All dependencies installed successfully!")
        print("✨ You can now run the JobSpy application with account features.")
        print("\n🚀 Next steps:")
        print("   1. Run: python backend/main.py")
        print("   2. Open: http://localhost:8000")
        print("   3. Register a new account and start using the enhanced features!")
    else:
        print(f"\n⚠️  {len(packages) - success_count} packages failed to install.")
        print("Please check the error messages above and try installing manually:")
        for package in packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()