# 🗄️ Local Job Database Guide

This guide explains the new **Local Job Database** feature that allows you to:
- ⚡ **Search jobs instantly** without hitting external APIs every time
- 🤖 **Proactively scrape** jobs from target companies 
- 🔄 **Avoid duplicates** with intelligent deduplication
- 📊 **Track scraping** runs and database statistics

## 🚀 Quick Start

### 1. **Setup Your Local Database**
```bash
# Run the setup script to get started
python setup_job_database.py
```

This script will:
- ✅ Create database tables
- 🏢 Add 20 popular tech companies (Google, Microsoft, Amazon, etc.)
- 🔍 Run initial scraping for 5 companies
- 📖 Show usage examples

### 2. **Start the Server**
```bash
python backend/main.py
```

### 3. **Search Jobs Locally (Fast!)**
```bash
curl -X POST "http://localhost:8000/search-jobs-local-public" \
  -H "Content-Type: application/json" \
  -d '{
    "search_term": "software engineer",
    "company_names": ["Google", "Microsoft"], 
    "locations": ["USA"],
    "days_old": 30,
    "limit": 20
  }'
```

## 🏗️ Architecture Overview

### **Before (Slow):**
```
User Search → JobSpy API → Indeed/LinkedIn → Results
     ↑_______________(2-10 seconds delay)______________↑
```

### **After (Fast):**
```
Proactive Scraping → JobSpy API → Indeed/LinkedIn → Local Database
                                                          ↓
User Search → Local Database → Instant Results (< 100ms)
```

## 📊 Database Schema

### **Target Companies**
Companies you want to scrape regularly:
```sql
- id, name, display_name
- preferred_sites (indeed, linkedin, etc.)
- search_terms (software engineer, data scientist, etc.)  
- location_filters (USA, Remote, etc.)
- last_scraped, total_jobs_found
```

### **Scraped Jobs**
All job data with deduplication:
```sql
- job_url, job_hash (for deduplication)
- title, company, location, site
- description, job_type, is_remote  
- salary info, date_posted, date_scraped
- experience_years (extracted from description)
```

### **Scraping Runs**
Track all scraping sessions:
```sql
- run_type, status, duration
- companies_scraped, sites_used
- total_jobs_found, new_jobs_added, duplicates_skipped
```

## 🔍 API Endpoints

### **Job Search (Local Database)**

#### Search Jobs (Authenticated)
```http
POST /search-jobs-local
Authorization: Bearer YOUR_TOKEN
```

#### Search Jobs (Public)
```http  
POST /search-jobs-local-public
Content-Type: application/json

{
  "search_term": "data scientist",
  "company_names": ["Google", "Netflix"],
  "locations": ["USA", "Remote"],
  "job_types": ["fulltime"],
  "is_remote": true,
  "min_salary": 100000,
  "max_salary": 200000,
  "max_experience_years": 5,
  "sites": ["indeed"],
  "days_old": 30,
  "limit": 50,
  "offset": 0
}
```

### **Company Management**

#### Add Target Company
```http
POST /admin/target-companies
Authorization: Bearer YOUR_TOKEN

{
  "name": "OpenAI",
  "display_name": "OpenAI",
  "preferred_sites": ["indeed", "linkedin"],
  "search_terms": ["software engineer", "ML engineer"],
  "location_filters": ["USA", "Remote"]
}
```

#### List Target Companies
```http
GET /admin/target-companies
Authorization: Bearer YOUR_TOKEN
```

### **Job Scraping**

#### Bulk Scrape Companies
```http
POST /admin/scrape-bulk
Authorization: Bearer YOUR_TOKEN

{
  "company_names": ["Stripe", "Coinbase", "Databricks"],
  "search_terms": ["software engineer", "senior engineer"],
  "sites": ["indeed"],
  "locations": ["USA", "Remote"],
  "results_per_company": 100,
  "hours_old": 720
}
```

#### View Scraping History
```http
GET /admin/scraping-runs?limit=10&offset=0
Authorization: Bearer YOUR_TOKEN
```

#### Database Statistics
```http
GET /admin/database-stats
Authorization: Bearer YOUR_TOKEN
```

## 🔄 Deduplication Strategy

Jobs are deduplicated using a **hash-based approach**:

1. **Primary Key**: `job_url` (if available)
2. **Secondary Key**: `title + company + location` (if no URL)
3. **Hash Storage**: MD5 hash stored in `job_hash` field
4. **Uniqueness**: Database constraint prevents duplicate hashes

### Example:
```python
# Two identical jobs from different searches
Job 1: "Software Engineer at Google in Mountain View"
Job 2: "Software Engineer at Google in Mountain View" 

# Same hash → Only stored once ✅
hash = md5("software engineer|google|mountain view")
```

## ⚡ Performance Benefits

### **Search Speed Comparison:**
- **External API**: 2-10 seconds per search
- **Local Database**: < 100ms per search
- **Improvement**: **20-100x faster**

### **API Rate Limiting:**
- **External**: Limited by Indeed/LinkedIn rate limits
- **Local**: No limits, search as much as you want

### **Cost:**
- **External**: Potential API costs/blocks
- **Local**: Free unlimited searching

## 🕒 Scheduling Regular Scraping

### **Option 1: Cron Job (Linux/Mac)**
```bash
# Add to crontab: scrape daily at 2 AM
0 2 * * * cd /path/to/jobspy && python -c "
import asyncio
from backend.job_scraper import job_scraper
from backend.database import get_db
from backend.models import BulkScrapingRequest

async def daily_scrape():
    db = next(get_db())
    companies = ['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta']
    request = BulkScrapingRequest(
        company_names=companies,
        search_terms=['software engineer', 'data scientist'],
        sites=['indeed'],
        locations=['USA'],
        results_per_company=100
    )
    await job_scraper.bulk_scrape_companies(request, db)

asyncio.run(daily_scrape())
"
```

### **Option 2: Task Scheduler (Windows)**
Create a batch file and schedule it:
```batch
@echo off
cd C:\path\to\jobspy
python daily_scrape.py
```

### **Option 3: Background Service**
Create a Python service that runs scraping on a schedule.

## 📈 Monitoring & Maintenance

### **Database Statistics**
Monitor your database health:
```bash
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/admin/database-stats
```

### **Cleaning Old Jobs**
```sql
-- Remove jobs older than 60 days
UPDATE scraped_jobs 
SET is_active = false 
WHERE date_scraped < NOW() - INTERVAL 60 DAY;
```

### **Database Size Management**
```sql
-- Check database size
SELECT 
  COUNT(*) as total_jobs,
  COUNT(CASE WHEN is_active THEN 1 END) as active_jobs,
  AVG(LENGTH(description)) as avg_description_length
FROM scraped_jobs;
```

## 🛠️ Troubleshooting

### **Common Issues:**

#### **1. No Jobs Found in Local Search**
```bash
# Check if database has jobs
curl http://localhost:8000/admin/database-stats

# If empty, run initial scraping
python setup_job_database.py
```

#### **2. Scraping Fails**
```bash
# Check scraping run logs
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/admin/scraping-runs

# Look for error_message field
```

#### **3. Duplicate Jobs Still Appearing**
```sql
-- Check for hash collisions
SELECT job_hash, COUNT(*) 
FROM scraped_jobs 
GROUP BY job_hash 
HAVING COUNT(*) > 1;
```

#### **4. Slow Local Searches**
```sql
-- Rebuild indexes
REINDEX TABLE scraped_jobs;

-- Check query plan
EXPLAIN QUERY PLAN 
SELECT * FROM scraped_jobs 
WHERE title LIKE '%engineer%';
```

## 🔧 Advanced Configuration

### **Custom Company Search Terms**
```python
# Add company with specialized search terms
{
  "name": "Palantir",
  "search_terms": [
    "software engineer",
    "forward deployed engineer", 
    "deployment strategist"
  ]
}
```

### **Multiple Location Targets**
```python
# Target specific locations
{
  "name": "Uber",
  "location_filters": [
    "San Francisco, CA",
    "New York, NY", 
    "Austin, TX",
    "Remote"
  ]
}
```

### **Site-Specific Scraping**
```python
# Different sites for different companies
{
  "name": "LinkedIn",
  "preferred_sites": ["linkedin"],  # Scrape LinkedIn from LinkedIn
  "name": "Google", 
  "preferred_sites": ["indeed", "glassdoor"]  # Multiple sources
}
```

## 🎯 Best Practices

### **1. Start Small**
- Begin with 5-10 companies
- Use 50-100 results per company initially
- Monitor database size and performance

### **2. Strategic Company Selection**
- Choose companies you're actually interested in
- Include both large (Google, Microsoft) and small (startups) companies
- Add companies from your target industries

### **3. Smart Search Terms**
- Use specific terms relevant to your skills
- Include both junior and senior level positions
- Add role variations (engineer, developer, scientist)

### **4. Regular Maintenance**
- Scrape weekly or bi-weekly
- Clean old jobs monthly
- Monitor database size
- Update company lists quarterly

### **5. Search Strategy**
- Use local database for quick filtering
- Use external API for real-time/fresh results
- Combine both approaches for comprehensive search

## 🚀 What's Next?

The local database enables many advanced features:

- **🤖 AI-Powered Recommendations**: Analyze your saved jobs to suggest similar positions
- **📧 Smart Alerts**: Get notified when jobs matching your criteria are found
- **📊 Market Analysis**: Track salary trends, company hiring patterns
- **🎯 Application Tracking**: Connect scraped jobs with your application status
- **🔍 Advanced Filtering**: ML-based filtering on job descriptions

---

**🎉 Congratulations!** You now have a blazing-fast local job database that will save you hours of waiting for API responses. Search faster, find more opportunities, and land your dream job! 🚀