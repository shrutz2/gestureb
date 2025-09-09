from datetime import datetime
from bson import ObjectId
import bcrypt
from flask_pymongo import PyMongo

mongo = PyMongo()

class User:
    def __init__(self, username=None, email=None, password=None, _id=None):
        self.username = username
        self.email = email
        self.password = password
        self._id = _id
        self.created_at = datetime.utcnow()
        self.stats = {
            'level': 1,
            'total_points': 0,
            'words_learned': 0,
            'practice_sessions': 0,
            'accuracy_percentage': 0.0
        }
    
    @staticmethod
    def hash_password(password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    @staticmethod
    def check_password(password, hashed_password):
        """Check if password matches hashed password"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
    
    def save(self):
        """Save user to MongoDB"""
        user_data = {
            'username': self.username,
            'email': self.email,
            'password': self.hash_password(self.password),
            'created_at': self.created_at,
            'stats': self.stats
        }
        result = mongo.db.users.insert_one(user_data)
        self._id = result.inserted_id
        return self
    
    @staticmethod
    def find_by_email(email):
        """Find user by email"""
        user_data = mongo.db.users.find_one({'email': email})
        if user_data:
            user = User()
            user._id = user_data['_id']
            user.username = user_data['username']
            user.email = user_data['email']
            user.password = user_data['password']
            user.created_at = user_data.get('created_at', datetime.utcnow())
            user.stats = user_data.get('stats', {
                'level': 1,
                'total_points': 0,
                'words_learned': 0,
                'practice_sessions': 0,
                'accuracy_percentage': 0.0
            })
            return user
        return None
    
    @staticmethod
    def find_by_id(user_id):
        """Find user by ID"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        
        user_data = mongo.db.users.find_one({'_id': user_id})
        if user_data:
            user = User()
            user._id = user_data['_id']
            user.username = user_data['username']
            user.email = user_data['email']
            user.password = user_data['password']
            user.created_at = user_data.get('created_at', datetime.utcnow())
            user.stats = user_data.get('stats', {
                'level': 1,
                'total_points': 0,
                'words_learned': 0,
                'practice_sessions': 0,
                'accuracy_percentage': 0.0
            })
            return user
        return None
    
    @staticmethod
    def email_exists(email):
        """Check if email already exists"""
        return mongo.db.users.find_one({'email': email}) is not None
    
    @staticmethod
    def username_exists(username):
        """Check if username already exists"""
        return mongo.db.users.find_one({'username': username}) is not None
    
    def update_stats(self, stats_update):
        """Update user stats"""
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$set': {'stats': stats_update}}
        )
        self.stats = stats_update
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': str(self._id),
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'stats': self.stats
        }