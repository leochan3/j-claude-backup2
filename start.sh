#!/bin/bash

# JobSpy Application Startup Script
# Simple script to start both backend and frontend

echo "🔍 JobSpy Web Application Launcher"
echo "=================================="

# Change to project directory
cd "$(dirname "$0")"

# Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📝 Creating .env file from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your actual API keys"
    else
        echo "⚠️  No .env file found. OpenAI features will not be available."
    fi
fi

# Function to kill processes on ports
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    
    # Kill processes on ports 8000 and 3000
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    fi
    
    echo "✅ All servers stopped"
    exit 0
}

# Set up trap for cleanup on Ctrl+C
trap cleanup INT TERM

# Start backend
echo "🚀 Starting backend server..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "🚀 Starting frontend server..."
cd frontend
python -m http.server 3000 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 2

echo ""
echo "=================================="
echo "🎉 Application started successfully!"
echo "📱 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "=================================="
echo "💡 Press Ctrl+C to stop all servers"
echo "=================================="

# Open browser (macOS)
if command -v open >/dev/null 2>&1; then
    echo "🌐 Opening application in browser..."
    open http://localhost:3000
fi

# Wait for user input to keep script running
wait