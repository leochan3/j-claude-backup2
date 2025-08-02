# 🛠️ Admin Interfaces Guide

Your JobSpy application now has **4 fully functional frontend interfaces**:

## 🔍 **Main Job Search Interface**
- **File**: `frontend/index.html`
- **Purpose**: Primary job search interface for end users
- **Features**: User registration, login, job search, save jobs, manage preferences

## 📊 **Database Viewer (Admin)**
- **File**: `database_viewer.html`
- **Purpose**: View database statistics and stored job data
- **Features**: 
  - Real-time database statistics
  - View target companies
  - Search local job database
  - Top companies by job count

## 🤖 **Scraping Interface (Admin)**
- **File**: `scraping_interface.html` 
- **Purpose**: Manage job scraping operations
- **Features**:
  - Bulk scrape jobs from multiple companies
  - Configure scraping parameters
  - Real-time scraping progress
  - Results summary

## 👥 **User Management (Admin)**
- **File**: `user_management.html`
- **Purpose**: Monitor and manage registered users
- **Features**:
  - View all registered users and their information
  - User statistics (total, active, recent registrations)
  - Detailed user profiles with activity history
  - Saved jobs tracking and categorization
  - Search history and saved searches monitoring
  - Most active users analytics

---

## 🚀 Quick Setup

### 1. **Start the Backend**
```bash
cd "C:\Users\S0C0VYB\Documents\Cursor Project\j-claude2 backup"
python backend/main.py
```

### 2. **Setup Sample Companies (Optional)**
```bash
python setup_sample_companies.py
```

### 3. **Access the Interfaces**
- **Main Job Search**: `frontend/index.html`
- **Database Viewer**: `database_viewer.html`
- **Scraping Interface**: `scraping_interface.html`
- **User Management**: `user_management.html`
- **Test Dashboard**: `test_admin_frontends.html`

---

## 🔧 Backend API Endpoints

### **Public Endpoints** (No Authentication Required)
```
GET  /health                     - Health check
GET  /supported-sites           - Available job sites
GET  /supported-countries       - Supported countries
POST /search-jobs-public        - Public job search
POST /search-jobs-local-public  - Search local database
GET  /database-stats-public     - Database statistics (for admin UI)
GET  /target-companies-public   - Get companies (for admin UI)
POST /scrape-bulk-public        - Bulk scraping (for admin UI)
GET  /admin/users-public        - Get all users (for admin UI)
GET  /admin/user-details-public/{user_id} - Get user details (for admin UI)
GET  /admin/users-stats-public  - Get user statistics (for admin UI)
```

### **Authenticated Endpoints** (Require Login)
```
POST /auth/register             - User registration
POST /auth/login               - User login
POST /search-jobs              - Authenticated job search
GET  /user/preferences         - User preferences
POST /user/save-job           - Save job
GET  /user/saved-jobs         - Get saved jobs
```

### **Admin Endpoints** (Require Authentication)
```
POST /admin/target-companies   - Manage companies
POST /admin/scrape-bulk       - Authenticated bulk scraping
GET  /admin/database-stats    - Admin database stats
GET  /admin/scraping-runs     - View scraping history
```

---

## 🎯 Usage Workflows

### **For End Users**
1. Open `frontend/index.html`
2. Register/Login
3. Search for jobs
4. Save interesting jobs
5. Manage preferences

### **For Administrators**

#### **View Database Statistics**
1. Open `database_viewer.html`
2. View real-time statistics
3. Browse target companies
4. Search local job database

#### **Scrape New Jobs**
1. Open `scraping_interface.html`
2. Enter company names (one per line)
3. Configure search parameters
4. Click "Start Scraping"
5. Wait for completion (may take several minutes)

#### **Monitor Progress**
1. Use `test_admin_frontends.html` to test all interfaces
2. Check backend health and endpoint status
3. Verify scraping functionality

---

## 📈 Sample Companies Included

After running `setup_sample_companies.py`, you'll have these companies:
- Google, Microsoft, Amazon, Apple, Meta
- Netflix, Uber, Lyft, Airbnb, Stripe

Each company includes:
- Preferred job sites (Indeed, LinkedIn)
- Common search terms (software engineer, data scientist, etc.)
- Location filters (USA, Remote, specific cities)

---

## 🔍 Troubleshooting

### **Backend Not Responding**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Restart backend
python backend/main.py
```

### **Database Issues**
```bash
# Fix database schema
python backend/migrate_db.py

# Or run comprehensive fix
python fix_issues.py
```

### **Admin Interfaces Not Loading**
1. Ensure backend is running on port 8000
2. Check browser console for errors
3. Test endpoints using `test_admin_frontends.html`

### **No Companies Showing**
```bash
# Add sample companies
python setup_sample_companies.py
```

### **Scraping Fails**
- Check internet connection
- Try fewer companies at once
- Reduce `results_per_company` setting
- Use only Indeed (most reliable)

---

## 🌟 Features Overview

### **Database Viewer Features**
- ✅ Real-time statistics dashboard
- ✅ Company management view
- ✅ Local job database search
- ✅ Top companies ranking
- ✅ Responsive design

### **Scraping Interface Features**
- ✅ Bulk company scraping
- ✅ Configurable search parameters
- ✅ Multiple job sites support
- ✅ Real-time progress feedback
- ✅ Duplicate detection
- ✅ Results summary

### **Main Interface Features**
- ✅ User authentication system
- ✅ Advanced job search filters
- ✅ Job saving and categorization
- ✅ Search history tracking
- ✅ User preferences management
- ✅ AI-powered job filtering

---

## 🚨 Important Notes

1. **Authentication**: Admin interfaces use public endpoints for simplicity. In production, add authentication.

2. **Rate Limiting**: Be respectful when scraping. Indeed has no rate limits, but LinkedIn does.

3. **Data Storage**: All scraped jobs are stored locally in SQLite database with intelligent deduplication.

4. **Performance**: Scraping can take time. Be patient, especially with multiple companies.

5. **Reliability**: Indeed is the most reliable job site. Use it as primary source.

---

## 📚 API Documentation

For complete API documentation, visit: http://localhost:8000/docs

This interactive documentation shows all available endpoints, request/response formats, and allows testing directly in the browser.

---

🎉 **Your JobSpy application now has comprehensive admin capabilities!**