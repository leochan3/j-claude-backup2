@echo off
echo 🔧 JobSpy Account System Setup
echo ===============================

echo.
echo 📦 Installing Python dependencies...
python install_dependencies.py

echo.
echo 🗄️ Setting up database...
python -c "from backend.database import create_tables; create_tables(); print('✅ Database tables created successfully')"

echo.
echo 🎉 Setup complete! 
echo.
echo 🚀 To start the application:
echo    python backend/main.py
echo.
echo 🌐 Then open: http://localhost:8000
echo.
pause