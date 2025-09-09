import os
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.config import Config
from app.celery_utils import make_celery

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()
celery = make_celery()

def create_app():
    """Create Flask API server - Clean CORS (No Duplicates)"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # CLEAN CORS Configuration - Only use Flask-CORS
    CORS(app, 
         origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization'],
         supports_credentials=True)
    
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    
    # Initialize MongoDB (with error handling)
    try:
        from app.auth_models import mongo
        mongo.init_app(app)
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"⚠️ MongoDB connection warning: {e}")
        print("💡 Make sure MongoDB is running!")
    
    # Initialize Celery
    celery.conf.update(
        broker_url=app.config["CELERY_BROKER_URL"],
        result_backend=app.config["CELERY_RESULT_BACKEND"],
    )

    # Ensure the SQLite database exists (for existing ML models)
    db_path = os.path.join(app.root_path, "sign-language-recognition.sqlite")
    
    with app.app_context():
        try:
            # Import existing models
            from app import models
            
            if not os.path.exists(db_path):
                db.create_all()
                print("✅ SQLite Database created for ML data.")
            else:
                print("✅ SQLite Database exists for ML data.")
        except Exception as e:
            print(f"⚠️ SQLite setup warning: {e}")

        # Register auth routes FIRST
        try:
            from app.auth_routes import auth_bp
            app.register_blueprint(auth_bp)
            print("✅ Authentication routes registered.")
        except Exception as e:
            print(f"⚠️ Auth routes warning: {e}")
        
        # Register search routes
        try:
            from app.search_routes import search_bp
            app.register_blueprint(search_bp)
            print("✅ Search routes registered.")
        except Exception as e:
            print(f"⚠️ Search routes warning: {e}")
        
        # Import existing routes LAST (with error handling)  
        try:
            from app import routes
            print("✅ Existing ML routes imported.")
        except Exception as e:
            print(f"⚠️ ML routes warning: {e}")

    # JWT Error handlers (without extra CORS headers)
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'Token has expired'}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'Invalid token'}), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({'error': 'Token is required'}), 401

    # Simple test endpoint
    @app.route('/api/test', methods=['GET'])
    def test_api():
        return jsonify({
            'message': 'API is working!',
            'cors': 'enabled',
            'timestamp': str(request.headers.get('Origin', 'No origin'))
        })

    print("🚀 Flask API server initialized!")
    print("🔍 Search API: /api/search (POST)")
    print("🔐 Auth API: /api/auth/* (POST/GET)")
    print("🧪 Test API: /api/test (GET)")
    print("🎯 CORS: Clean configuration for localhost:3000")
    
    return app