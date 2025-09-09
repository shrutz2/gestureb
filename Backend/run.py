#!/usr/bin/env python3
"""
Sign Language Recognition API Server
"""

import os
from app import create_app
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask API app
app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Sign Language Recognition API Server")
    print("=" * 60)
    print("📱 Frontend (React): http://localhost:3000")
    print("⚡ Backend API: http://localhost:5000/api")
    print("🔍 Health Check: http://localhost:5000/api/health")
    print("=" * 60)
    
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )