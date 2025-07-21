# 🚀 Quick Setup Guide

## 🎯 One-Command Startup

### Option 1: Python Launcher (Recommended)
```bash
python start.py
```

### Option 2: Shell Script (Unix/macOS/Linux)
```bash
./start.sh
```

Both scripts will:
- ✅ Create `.env` file from `.env.example` if needed
- ✅ Kill any existing processes on ports 8000/3000
- ✅ Start backend server (http://localhost:8000)
- ✅ Start frontend server (http://localhost:3000)
- ✅ Open your browser automatically
- ✅ Handle graceful shutdown with Ctrl+C

## 🔧 Manual Setup (If needed)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings (optional)
nano .env
```

### 3. Start Services Manually

#### Backend:
```bash
cd backend
python main.py
```

#### Frontend:
```bash
cd frontend
python -m http.server 3000
```

## 🔑 API Key Configuration (Optional)

The application works without API keys, but AI filtering features require OpenAI:

1. Get API key from: https://platform.openai.com/api-keys
2. Edit `.env` file:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## 🐙 Git Usage (No More API Key Issues!)

The `.gitignore` is now properly configured to exclude:
- `.env` files (your API keys stay local)
- Python cache files
- OS-specific files
- IDE configurations

**Safe to push to GitHub:**
```bash
git add .
git commit -m "Update project"
git push
```

Your API keys in `.env` will never be committed!

## 🛠️ Troubleshooting

### "Port already in use"
The startup scripts automatically kill existing processes. If issues persist:
```bash
# Kill processes manually
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

### "Backend not responding"
1. Check if backend started successfully
2. Verify dependencies are installed: `pip install -r requirements.txt`
3. Check logs in terminal

### "Frontend not loading"
1. Try http://127.0.0.1:3000 instead of localhost
2. Clear browser cache
3. Try different browser

## 📱 Access Points

Once running:
- **Web App**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔄 Updating the Project

```bash
git pull origin main
pip install -r requirements.txt  # Update dependencies if needed
python start.py                  # Restart with new changes
```