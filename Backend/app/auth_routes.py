from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, 
    jwt_required, 
    get_jwt_identity,
    create_refresh_token
)
from app.auth_models import User
import re

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    return len(password) >= 6

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not username or len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Please enter a valid email address'}), 400
        
        if not validate_password(password):
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        if User.email_exists(email):
            return jsonify({'error': 'Email is already registered'}), 400
        
        if User.username_exists(username):
            return jsonify({'error': 'Username is already taken'}), 400
        
        # Create new user
        user = User(username=username, email=email, password=password)
        user.save()
        
        # Create tokens
        access_token = create_access_token(identity=str(user._id))
        refresh_token = create_refresh_token(identity=str(user._id))
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'token': access_token,
            'refresh_token': refresh_token
        }), 201
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'An error occurred during registration'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.find_by_email(email)
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Check password
        if not User.check_password(password, user.password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Create tokens
        access_token = create_access_token(identity=str(user._id))
        refresh_token = create_refresh_token(identity=str(user._id))
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': access_token,
            'refresh_token': refresh_token
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'An error occurred during login'}), 500

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()}), 200
        
    except Exception as e:
        print(f"Get user error: {e}")
        return jsonify({'error': 'An error occurred'}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    try:
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        new_token = create_access_token(identity=current_user_id)
        
        return jsonify({
            'token': new_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        print(f"Refresh token error: {e}")
        return jsonify({'error': 'An error occurred'}), 500

@auth_bp.route('/update-stats', methods=['POST'])
@jwt_required()
def update_user_stats():
    try:
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        stats_update = data.get('stats', {})
        
        # Merge with existing stats
        current_stats = user.stats.copy()
        current_stats.update(stats_update)
        
        user.update_stats(current_stats)
        
        return jsonify({
            'message': 'Stats updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        print(f"Update stats error: {e}")
        return jsonify({'error': 'An error occurred'}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    # JWT tokens को server-side invalidate करने के लिए
    # आप token blacklist implement कर सकते हैं
    # अभी के लिए सिर्फ success response भेज रहे हैं
    return jsonify({'message': 'Logged out successfully'}), 200