import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

VIDEO_DIR = os.path.join(basedir, "videos")

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR, exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "gizli_anahtar"
    
    # SQLite Database (आपके existing ML models के लिए)
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(
        basedir, "sign-language-recognition.sqlite"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # MongoDB Configuration (Authentication के लिए)
    MONGO_URI = os.environ.get("MONGO_URI") or "mongodb://localhost:27017/gesture_auth"
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or "jwt-secret-string"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Celery Configuration
    CELERY_BROKER_URL = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/0"