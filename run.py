#!/usr/bin/env python3
"""
JobSpy Application Runner

This script helps you start the JobSpy application easily.
It will start the backend API server and optionally open the frontend.
"""

import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import jobspy
        import pandas
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    backend_path = Path("backend/main.py")
    if not backend_path.exists():
        print("❌ Backend file not found. Make sure you're in the project root directory.")
        return None
    
    print("🚀 Starting JobSpy API server...")
    try:
        # Start the backend server
        process = subprocess.Popen([
            sys.executable, str(backend_path)
        ], cwd=".")
        
        # Give the server time to start
        time.sleep(3)
        
        # Check if server is running
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend server is running on http://localhost:8000")
                print("📚 API Documentation: http://localhost:8000/docs")
                return process
            else:
                print("❌ Backend server responded but not healthy")
                return None
        except requests.exceptions.RequestException:
            print("❌ Could not connect to backend server")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def open_frontend():
    """Open the frontend in the default browser"""
    frontend_path = Path("frontend/index.html")
    if not frontend_path.exists():
        print("❌ Frontend file not found")
        return False
    
    print("🌐 Opening frontend in browser...")
    try:
        frontend_url = f"file://{frontend_path.absolute()}"
        webbrowser.open(frontend_url)
        print(f"✅ Frontend opened: {frontend_url}")
        return True
    except Exception as e:
        print(f"❌ Failed to open frontend: {e}")
        print(f"Please manually open: {frontend_path.absolute()}")
        return False

def main():
    """Main application runner"""
    print("🔍 JobSpy Application Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        sys.exit(1)
    
    # Open frontend
    open_frontend()
    
    print("\n🎯 JobSpy Application is ready!")
    print("=" * 40)
    print("Backend API: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Frontend: Check your browser")
    print("\n💡 Tips:")
    print("- Try searching for 'python developer' to test the application")
    print("- Check the API documentation for advanced usage")
    print("- Press Ctrl+C to stop the application")
    
    try:
        # Keep the script running
        print("\n⏳ Application running... Press Ctrl+C to stop")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Application stopped successfully")

if __name__ == "__main__":
    main() 