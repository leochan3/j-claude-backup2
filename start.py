#!/usr/bin/env python3
"""
JobSpy Application Launcher
Starts both backend and frontend servers with proper error handling
"""

import subprocess
import time
import os
import sys
import signal
from pathlib import Path

def check_port(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def kill_process_on_port(port):
    """Kill any process running on the specified port"""
    try:
        if sys.platform == "darwin" or sys.platform == "linux":
            subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, capture_output=True)
        elif sys.platform == "win32":
            subprocess.run(f"netstat -ano | findstr :{port}", shell=True, capture_output=True)
    except:
        pass

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    
    # Kill any existing process on port 8000
    kill_process_on_port(8000)
    time.sleep(1)
    
    backend_path = Path(__file__).parent / "backend"
    original_dir = os.getcwd()
    os.chdir(backend_path)
    
    # Start backend server
    backend_process = subprocess.Popen([
        sys.executable, "main.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for server to start
    time.sleep(4)
    
    # Check if process is still running
    if backend_process.poll() is not None:
        # Process died, get error output
        stdout, stderr = backend_process.communicate()
        print("❌ Backend failed to start. Error:")
        if stderr:
            print(f"   {stderr.decode()}")
        if stdout:
            print(f"   {stdout.decode()}")
        os.chdir(original_dir)
        return None
    
    # Check if backend is running on port
    if check_port(8000):
        print("❌ Backend failed to start on port 8000")
        stdout, stderr = backend_process.communicate()
        if stderr:
            print(f"   Error: {stderr.decode()}")
        os.chdir(original_dir)
        return None
    else:
        print("✅ Backend server running on http://localhost:8000")
        os.chdir(original_dir)
        return backend_process

def start_frontend():
    """Start the frontend HTTP server"""
    print("🚀 Starting frontend server...")
    
    # Kill any existing process on port 3000
    kill_process_on_port(3000)
    time.sleep(1)
    
    frontend_path = Path(__file__).parent / "frontend"
    os.chdir(frontend_path)
    
    # Start frontend server
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "http.server", "3000"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Check if frontend is running
    if check_port(3000):
        print("❌ Frontend failed to start on port 3000")
        return None
    else:
        print("✅ Frontend server running on http://localhost:3000")
        return frontend_process

def open_browser():
    """Open the application in default browser"""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "http://localhost:3000"])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", "http://localhost:3000"])
        elif sys.platform == "win32":
            subprocess.run(["start", "http://localhost:3000"], shell=True)
        print("🌐 Opening application in browser...")
    except:
        print("💡 Please manually open http://localhost:3000 in your browser")

def main():
    """Main function to start the application"""
    print("=" * 50)
    print("🔍 JobSpy Web Application Launcher")
    print("=" * 50)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if .env exists, if not copy from .env.example
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("📝 Creating .env file from .env.example...")
            with open(".env.example", "r", encoding='utf-8') as src, open(".env", "w", encoding='utf-8') as dst:
                dst.write(src.read())
            print("⚠️  Please edit .env file with your actual API keys")
        else:
            print("⚠️  No .env file found. OpenAI features will not be available.")
    else:
        # Check if .env file has encoding issues and fix them
        try:
            with open(".env", "r", encoding='utf-8') as f:
                content = f.read()
                if '\x00' in content:  # Check for null bytes
                    print("🔧 Fixing .env file encoding issues...")
                    # Recreate from example
                    with open(".env.example", "r", encoding='utf-8') as src:
                        with open(".env", "w", encoding='utf-8') as dst:
                            dst.write(src.read())
        except UnicodeDecodeError:
            print("🔧 Fixing .env file encoding issues...")
            # Recreate from example
            with open(".env.example", "r", encoding='utf-8') as src:
                with open(".env", "w", encoding='utf-8') as dst:
                    dst.write(src.read())
    
    processes = []
    
    try:
        # Start backend
        backend = start_backend()
        if backend:
            processes.append(backend)
        
        # Start frontend
        frontend = start_frontend()
        if frontend:
            processes.append(frontend)
        
        if not processes:
            print("❌ Failed to start any servers")
            return
        
        print("\\n" + "=" * 50)
        print("🎉 Application started successfully!")
        print("📱 Frontend: http://localhost:3000")
        print("🔗 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("💡 Press Ctrl+C to stop all servers")
        print("=" * 50)
        
        # Open browser
        open_browser()
        
        # Wait for user interruption
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down servers...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        # Kill any remaining processes on the ports
        kill_process_on_port(8000)
        kill_process_on_port(3000)
        
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()