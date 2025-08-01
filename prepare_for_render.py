#!/usr/bin/env python3
"""
Prepare frontend files for Render deployment by updating API endpoints
"""

import os
import re
from pathlib import Path

def update_api_base_url(file_path, new_api_base):
    """Update API_BASE constant in HTML/JS files"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patterns to match different API base configurations
        patterns_and_replacements = [
            # Pattern 1: const API_BASE = 'http://localhost:8000'
            (r"const API_BASE\s*=\s*['\"]http://localhost:\d+['\"]", f"const API_BASE = '{new_api_base}'"),
            # Pattern 2: const API_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : window.location.origin;
            (r"const API_BASE_URL\s*=\s*window\.location\.protocol[^;]+;", f"const API_BASE_URL = '{new_api_base}';"),
            # Pattern 3: Direct hardcoded URLs like 'http://localhost:8000/endpoint'
            (r"['\"]http://localhost:\d+(/[^'\"]*)?['\"]", lambda m: f"'{new_api_base}{m.group(1) or ''}'")
        ]
        
        # Try each pattern and apply replacements
        updated_content = content
        pattern_matched = False
        
        for pattern, replacement in patterns_and_replacements:
            if re.search(pattern, updated_content):
                updated_content = re.sub(pattern, replacement, updated_content)
                pattern_matched = True
                print(f"✅ Applied pattern to {file_path}: {pattern[:30]}...")
        
        if pattern_matched:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ Updated {file_path}")
            return True
        else:
            print(f"⚠️  No API_BASE patterns found in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def prepare_for_render():
    """Prepare all frontend files for Render deployment"""
    
    print("🚀 Preparing frontend files for Render deployment...")
    print("=" * 60)
    
    # Get the production API URL (will be the same as the web URL)
    render_url = "https://j-claude-backup2-1.onrender.com"
    
    # Files to update
    frontend_files = [
        "frontend/index.html",
        "database_viewer.html", 
        "scraping_interface.html",
        "test_admin_frontends.html",
        "test_frontend.html"
    ]
    
    updated_count = 0
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            if update_api_base_url(file_path, render_url):
                updated_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("=" * 60)
    print(f"📊 Updated {updated_count} files with production API URL")
    print(f"🌐 API Base URL: {render_url}")
    
    # Also create a production info file
    with open("PRODUCTION_URLS.md", "w", encoding='utf-8') as f:
        f.write(f"""# Production URLs

## Your Live Application
🌐 **Main URL**: {render_url}

## Available Interfaces
- **API Documentation**: {render_url}/docs
- **Main Job Search**: {render_url}/app
- **Database Viewer**: {render_url}/database-viewer
- **Scraping Interface**: {render_url}/scraping-interface
- **Health Check**: {render_url}/health

## API Endpoints
All your API endpoints are available at: {render_url}

Example:
- `GET {render_url}/supported-sites`
- `POST {render_url}/search-jobs-public`
- `GET {render_url}/database-stats-public`

## Next Steps
1. Push these changes to your repository
2. Render will automatically redeploy
3. Your frontends will work with the production backend!
""")
    
    print(f"📝 Created PRODUCTION_URLS.md with all your live URLs")
    print("=" * 60)
    print("🎉 Ready for Render deployment!")
    print()
    print("Next steps:")
    print("1. Commit and push these changes to your repo")
    print("2. Render will automatically redeploy")
    print("3. Access your app at: " + render_url)

if __name__ == "__main__":
    prepare_for_render()