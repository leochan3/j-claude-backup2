-- JobSpy Database SQL Queries
-- Use these queries with SQLite browser or command line

-- ===========================================
-- BASIC STATISTICS
-- ===========================================

-- Total jobs in database
SELECT COUNT(*) as total_jobs FROM scraped_jobs WHERE is_active = 1;

-- Jobs by company (top 10)
SELECT company, COUNT(*) as job_count 
FROM scraped_jobs 
WHERE is_active = 1 
GROUP BY company 
ORDER BY job_count DESC 
LIMIT 10;

-- Jobs by site
SELECT site, COUNT(*) as job_count 
FROM scraped_jobs 
WHERE is_active = 1 
GROUP BY site 
ORDER BY job_count DESC;

-- Jobs added in last 7 days
SELECT COUNT(*) as recent_jobs 
FROM scraped_jobs 
WHERE is_active = 1 
AND date_scraped >= datetime('now', '-7 days');

-- ===========================================
-- TARGET COMPANIES
-- ===========================================

-- List all target companies
SELECT 
    name,
    display_name,
    total_jobs_found,
    last_scraped,
    is_active
FROM target_companies 
ORDER BY total_jobs_found DESC;

-- Companies that haven't been scraped recently
SELECT 
    name,
    last_scraped,
    CASE 
        WHEN last_scraped IS NULL THEN 'Never scraped'
        WHEN last_scraped < datetime('now', '-7 days') THEN 'Over a week ago'
        ELSE 'Recently scraped'
    END as status
FROM target_companies 
WHERE is_active = 1
ORDER BY last_scraped ASC;

-- ===========================================
-- JOB DETAILS
-- ===========================================

-- Recent jobs with full details
SELECT 
    title,
    company,
    location,
    site,
    min_amount,
    max_amount,
    salary_interval,
    min_experience_years,
    is_remote,
    date_posted,
    date_scraped
FROM scraped_jobs 
WHERE is_active = 1 
ORDER BY date_scraped DESC 
LIMIT 20;

-- High-paying jobs (over $150k)
SELECT 
    title,
    company,
    location,
    min_amount,
    max_amount,
    salary_interval
FROM scraped_jobs 
WHERE is_active = 1 
AND (min_amount >= 150000 OR max_amount >= 150000)
ORDER BY COALESCE(max_amount, min_amount) DESC;

-- Remote jobs
SELECT 
    title,
    company,
    location,
    min_amount,
    max_amount
FROM scraped_jobs 
WHERE is_active = 1 
AND is_remote = 1
ORDER BY date_scraped DESC
LIMIT 20;

-- Entry-level jobs (0-2 years experience)
SELECT 
    title,
    company,
    location,
    min_experience_years,
    max_experience_years
FROM scraped_jobs 
WHERE is_active = 1 
AND (min_experience_years <= 2 OR min_experience_years IS NULL)
ORDER BY date_scraped DESC
LIMIT 20;

-- ===========================================
-- SCRAPING RUNS
-- ===========================================

-- Recent scraping runs
SELECT 
    run_type,
    status,
    total_jobs_found,
    new_jobs_added,
    duplicate_jobs_skipped,
    started_at,
    completed_at,
    duration_seconds
FROM scraping_runs 
ORDER BY started_at DESC 
LIMIT 10;

-- Success rate of scraping runs
SELECT 
    status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM scraping_runs), 2) as percentage
FROM scraping_runs 
GROUP BY status;

-- ===========================================
-- SEARCH QUERIES
-- ===========================================

-- Search for specific job titles
SELECT 
    title,
    company,
    location,
    min_amount,
    max_amount,
    date_scraped
FROM scraped_jobs 
WHERE is_active = 1 
AND title LIKE '%software engineer%'
ORDER BY date_scraped DESC
LIMIT 10;

-- Search for specific companies
SELECT 
    title,
    company,
    location,
    min_amount,
    max_amount,
    site
FROM scraped_jobs 
WHERE is_active = 1 
AND company LIKE '%Google%'
ORDER BY date_scraped DESC;

-- Jobs in specific locations
SELECT 
    title,
    company,
    location,
    min_amount,
    max_amount
FROM scraped_jobs 
WHERE is_active = 1 
AND location LIKE '%San Francisco%'
ORDER BY date_scraped DESC
LIMIT 10;

-- ===========================================
-- DATA QUALITY CHECKS
-- ===========================================

-- Jobs without salary information
SELECT COUNT(*) as jobs_without_salary
FROM scraped_jobs 
WHERE is_active = 1 
AND min_amount IS NULL 
AND max_amount IS NULL;

-- Jobs without job URLs
SELECT COUNT(*) as jobs_without_url
FROM scraped_jobs 
WHERE is_active = 1 
AND (job_url IS NULL OR job_url = '');

-- Duplicate job detection (by hash)
SELECT 
    job_hash,
    COUNT(*) as duplicate_count
FROM scraped_jobs 
WHERE is_active = 1
GROUP BY job_hash 
HAVING COUNT(*) > 1;

-- ===========================================
-- CLEANUP QUERIES (USE WITH CAUTION)
-- ===========================================

-- Mark old jobs as inactive (older than 60 days)
-- UPDATE scraped_jobs 
-- SET is_active = 0 
-- WHERE date_scraped < datetime('now', '-60 days');

-- Delete failed scraping runs
-- DELETE FROM scraping_runs 
-- WHERE status = 'failed' 
-- AND started_at < datetime('now', '-30 days');

-- ===========================================
-- EXPORT QUERIES
-- ===========================================

-- Export all active jobs to CSV format
SELECT 
    title,
    company,
    location,
    site,
    job_type,
    is_remote,
    min_amount,
    max_amount,
    salary_interval,
    min_experience_years,
    date_posted,
    date_scraped,
    job_url
FROM scraped_jobs 
WHERE is_active = 1 
ORDER BY date_scraped DESC;