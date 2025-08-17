#!/usr/bin/env python3
"""
Complete API Server for Sign Language Recognition
Provides all endpoints needed by the frontend
"""
import os
import sys
import time
import json
import cv2
import base64
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from collections import deque, defaultdict
import threading
import queue

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from simplified_config import config, logger
    from video_processor import VideoProcessor  
    from simplified_model import SimplifiedSignModel
    from label_map_parser import LabelMapParser
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure all required files are present...")
    sys.exit(1)

# Create Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Global variables for ML components
video_processor: Optional[VideoProcessor] = None
model: Optional[SimplifiedSignModel] = None
available_words: List[str] = []

# User session management
user_sessions = defaultdict(lambda: {
    'points': 0,
    'streak': 0,
    'correct_attempts': 0,
    'total_attempts': 0,
    'last_activity': time.time()
})

class GestureBuffer:
    """Manages gesture sequence buffering for real-time prediction"""
    
    def __init__(self, max_length: int = 30):
        self.max_length = max_length
        self.frames = deque(maxlen=max_length)
        self.confidences = deque(maxlen=max_length)
        self.lock = threading.Lock()
    
    def add_frame(self, landmarks: np.ndarray, confidence: float):
        with self.lock:
            self.frames.append(landmarks.copy())
            self.confidences.append(confidence)
    
    def get_sequence(self) -> Tuple[np.ndarray, List[float]]:
        with self.lock:
            if len(self.frames) == 0:
                return np.array([]), []
            
            frames_array = np.array(list(self.frames))
            confidences_list = list(self.confidences)
            return frames_array, confidences_list
    
    def clear(self):
        with self.lock:
            self.frames.clear()
            self.confidences.clear()

# Global gesture buffer for real-time processing
gesture_buffer = GestureBuffer()

def load_available_words() -> List[str]:
    """Load available words from videos directory and label map"""
    parser = LabelMapParser()
    words = parser.get_all_available_words()
    
    if not words:
        # Fallback common words
        words = [
            'hello', 'thanks', 'please', 'sorry', 'yes', 'no', 'good', 'bad',
            'help', 'more', 'water', 'food', 'love', 'family', 'friend', 'happy',
            'sad', 'angry', 'tired', 'hungry', 'cold', 'hot', 'big', 'small'
        ]
        logger.warning("Using fallback word list")
    
    logger.info(f"✅ Loaded {len(words)} available words")
    return words

def initialize_ml_components():
    """Initialize ML components (video processor and model)"""
    global video_processor, model
    
    try:
        # Initialize video processor
        logger.info("🔧 Initializing video processor...")
        video_processor = VideoProcessor()
        logger.info("✅ Video processor initialized")
        
        # Try to load trained model
        logger.info("🤖 Loading trained model...")
        if config.MODEL_PATH.exists() and config.LABEL_ENCODER_PATH.exists():
            # Get number of classes from available words
            num_classes = len(available_words)
            model = SimplifiedSignModel(num_classes=num_classes)
            
            if model.load_model():
                logger.info("✅ Trained model loaded successfully")
            else:
                logger.warning("⚠️ Failed to load model, using mock predictions")
                model = None
        else:
            logger.warning("⚠️ No trained model found, using mock predictions")
            logger.info("   Train model first: python simplified_trainer.py")
            model = None
            
    except Exception as e:
        logger.error(f"❌ Error initializing ML components: {e}")
        video_processor = None
        model = None

def startup():
    """Initialize application on startup"""
    logger.info("🚀 Starting Sign Language Recognition API...")
    global available_words
    
    # Load available words
    available_words = load_available_words()
    
    # Initialize ML components
    initialize_ml_components()
    
    # Create required directories
    config.VIDEOS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.DATA_DIR.mkdir(exist_ok=True)
    
    logger.info("🎉 API startup complete!")

# Initialize on import (replaces @app.before_first_request)
startup()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sign Language API is running',
        'timestamp': time.time(),
        'components': {
            'video_processor': video_processor is not None,
            'model': model is not None,
            'available_words': len(available_words),
            'videos_directory': config.VIDEOS_DIR.exists()
        }
    })

@app.route('/api/search', methods=['POST'])
def search_word():
    """Search for a word and check if video exists"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query'].lower().strip()
        if not query:
            return jsonify({'error': 'Empty query'}), 400
        
        # Check if word is available
        if query in available_words:
            video_path = config.VIDEOS_DIR / f"{query}.mp4"
            if video_path.exists():
                return jsonify({
                    'found': True,
                    'word': query,
                    'message': f'Ready to practice "{query}"',
                    'video_available': True,
                    'video_url': f'/videos/{query}.mp4'
                })
            else:
                return jsonify({
                    'found': True,
                    'word': query,
                    'message': f'Word "{query}" available but no video found',
                    'video_available': False
                })
        
        # Find similar words
        similar_words = find_similar_words(query, available_words[:10])
        
        return jsonify({
            'found': False,
            'message': f'Word "{query}" not found',
            'suggestions': similar_words
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_gesture():
    """Predict gesture from sequence of frames"""
    try:
        data = request.get_json()
        if not data or 'frames' not in data:
            return jsonify({'error': 'No frames provided'}), 400
        
        frames = data['frames']
        target_word = data.get('target_word', '').lower().strip()
        user_id = data.get('user_id', 'default_user')
        
        if not frames:
            return jsonify({'error': 'Empty frames list'}), 400
        
        # Process frames to extract landmarks
        landmarks_sequence = []
        detection_confidences = []
        
        for frame_data in frames:
            try:
                # Decode base64 image
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                
                image_bytes = base64.b64decode(frame_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Extract landmarks
                if video_processor:
                    landmarks, detected, confidence = video_processor.extract_landmarks_from_frame(frame)
                    if detected and confidence > 0.5:
                        landmarks_sequence.append(landmarks)
                        detection_confidences.append(confidence)
                else:
                    # Mock landmark extraction for testing
                    mock_landmarks = np.random.rand(config.FEATURES_PER_FRAME) * 0.1
                    landmarks_sequence.append(mock_landmarks)
                    detection_confidences.append(0.8)
                    
            except Exception as e:
                logger.warning(f"Frame processing error: {e}")
                continue
        
        if len(landmarks_sequence) < 5:
            return jsonify({
                'is_correct': False,
                'message': 'Not enough valid frames detected. Please ensure good lighting and clear hand visibility.',
                'confidence': 0.0,
                'points': 0,
                'predicted_word': '',
                'hands_detected_count': len(landmarks_sequence)
            })
        
        # Normalize sequence length
        normalized_sequence = normalize_sequence_length(landmarks_sequence)
        
        # Make prediction
        if model and len(landmarks_sequence) >= 10:
            try:
                predicted_word, confidence, top_predictions = model.predict_single_sequence(normalized_sequence)
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                # Fallback to mock prediction
                predicted_word, confidence, top_predictions = mock_prediction(target_word)
        else:
            # Mock prediction for testing
            predicted_word, confidence, top_predictions = mock_prediction(target_word)
        
        # Check if prediction is correct
        is_correct = (predicted_word.lower() == target_word.lower() and 
                     confidence > config.CONFIDENCE_THRESHOLD)
        
        # Calculate points
        points = 0
        if is_correct:
            base_points = 10
            confidence_bonus = int((confidence - config.CONFIDENCE_THRESHOLD) * 20)
            points = base_points + confidence_bonus
            
            # Update user session
            session = user_sessions[user_id]
            session['points'] += points
            session['correct_attempts'] += 1
            session['streak'] += 1
        else:
            user_sessions[user_id]['streak'] = 0
        
        user_sessions[user_id]['total_attempts'] += 1
        user_sessions[user_id]['last_activity'] = time.time()
        
        # Prepare response
        response = {
            'is_correct': is_correct,
            'predicted_word': predicted_word,
            'confidence': confidence,
            'points': points,
            'message': get_feedback_message(is_correct, predicted_word, target_word, confidence),
            'top_predictions': top_predictions,
            'hands_detected_count': len(landmarks_sequence),
            'user_stats': {
                'total_points': user_sessions[user_id]['points'],
                'streak': user_sessions[user_id]['streak'],
                'accuracy': (user_sessions[user_id]['correct_attempts'] / 
                           max(1, user_sessions[user_id]['total_attempts'])) * 100
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'is_correct': False,
            'confidence': 0.0,
            'points': 0
        }), 500

@app.route('/api/verify', methods=['POST'])
def verify_gesture():
    """Verify if predicted gesture matches target - alias for predict"""
    return predict_gesture()

@app.route('/api/score', methods=['POST'])
def update_score():
    """Update user score"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        points = data.get('points', 0)
        is_correct = data.get('is_correct', False)
        
        session = user_sessions[user_id]
        
        if is_correct:
            session['points'] += points
            session['correct_attempts'] += 1
            session['streak'] += 1
        else:
            session['streak'] = 0
        
        session['total_attempts'] += 1
        session['last_activity'] = time.time()
        
        return jsonify({
            'success': True,
            'user_stats': {
                'total_points': session['points'],
                'streak': session['streak'],
                'accuracy': (session['correct_attempts'] / max(1, session['total_attempts'])) * 100
            }
        })
        
    except Exception as e:
        logger.error(f"Score update error: {e}")
        return jsonify({'error': 'Score update failed'}), 500

@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files to frontend"""
    try:
        return send_from_directory(config.VIDEOS_DIR, filename)
    except Exception as e:
        logger.error(f"Video serving error: {e}")
        return jsonify({'error': 'Video not found'}), 404

@app.route('/api/words', methods=['GET'])
def get_words():
    """Get list of all available words"""
    return jsonify({
        'words': available_words,
        'count': len(available_words)
    })

@app.route('/api/user/<user_id>/stats', methods=['GET'])
def get_user_stats(user_id):
    """Get user statistics"""
    session = user_sessions[user_id]
    return jsonify({
        'points': session['points'],
        'streak': session['streak'],
        'correct_attempts': session['correct_attempts'],
        'total_attempts': session['total_attempts'],
        'accuracy': (session['correct_attempts'] / max(1, session['total_attempts'])) * 100,
        'last_activity': session['last_activity']
    })

@app.route('/api/user/<user_id>/reset', methods=['POST'])
def reset_user_stats(user_id):
    """Reset user statistics"""
    user_sessions[user_id] = {
        'points': 0,
        'streak': 0,
        'correct_attempts': 0,
        'total_attempts': 0,
        'last_activity': time.time()
    }
    return jsonify({'message': 'Stats reset successfully'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get detailed system status"""
    return jsonify({
        'api_version': '2.0',
        'model_loaded': model is not None,
        'processor_loaded': video_processor is not None,
        'available_words': len(available_words),
        'total_users': len(user_sessions),
        'model_path': str(config.MODEL_PATH) if config.MODEL_PATH.exists() else None,
        'videos_count': len(list(config.VIDEOS_DIR.glob('*.mp4'))) if config.VIDEOS_DIR.exists() else 0
    })

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_similar_words(query: str, word_list: List[str]) -> List[str]:
    """Find words similar to query using simple string matching"""
    similar = []
    
    # Exact substring matches
    for word in word_list:
        if query in word or word in query:
            similar.append(word)
    
    # Levenshtein distance for fuzzy matching
    if len(similar) < 5:
        from difflib import get_close_matches
        close_matches = get_close_matches(query, word_list, n=5, cutoff=0.6)
        similar.extend([w for w in close_matches if w not in similar])
    
    return similar[:5]

def normalize_sequence_length(sequence: List[np.ndarray], target_length: int = None) -> np.ndarray:
    """Normalize sequence to target length"""
    if target_length is None:
        target_length = config.SEQUENCE_LENGTH
    
    sequence_array = np.array(sequence)
    
    if len(sequence_array) == target_length:
        return sequence_array
    elif len(sequence_array) > target_length:
        # Uniformly sample frames
        indices = np.linspace(0, len(sequence_array) - 1, target_length, dtype=int)
        return sequence_array[indices]
    else:
        # Pad by repeating frames
        padding_needed = target_length - len(sequence_array)
        if len(sequence_array) > 0:
            # Repeat sequence cyclically
            repeated_indices = np.tile(np.arange(len(sequence_array)), 
                                     padding_needed // len(sequence_array) + 1)[:padding_needed]
            padding = sequence_array[repeated_indices]
            return np.vstack([sequence_array, padding])
        else:
            # Fallback if no frames
            return np.zeros((target_length, config.FEATURES_PER_FRAME))

def mock_prediction(target_word: str) -> Tuple[str, float, List[Dict]]:
    """Generate mock predictions for testing when model is not available"""
    import random
    
    # Simulate model accuracy - higher chance of correct prediction
    if target_word and random.random() < 0.7:  # 70% accuracy simulation
        predicted_word = target_word
        confidence = random.uniform(0.75, 0.95)
    else:
        # Random incorrect prediction
        predicted_word = random.choice(available_words[:10])
        confidence = random.uniform(0.3, 0.7)
    
    # Mock top predictions
    top_predictions = [
        {'word': predicted_word, 'confidence': confidence, 'percentage': f"{confidence*100:.1f}%"}
    ]
    
    # Add some random alternatives
    alternatives = [w for w in available_words[:5] if w != predicted_word]
    for word in alternatives[:4]:
        fake_conf = random.uniform(0.1, confidence * 0.8)
        top_predictions.append({
            'word': word, 
            'confidence': fake_conf,
            'percentage': f"{fake_conf*100:.1f}%"
        })
    
    return predicted_word, confidence, top_predictions

def get_feedback_message(is_correct: bool, predicted_word: str, 
                        target_word: str, confidence: float) -> str:
    """Generate appropriate feedback message"""
    if is_correct:
        if confidence > 0.9:
            return f"🎉 Perfect! Great job signing '{target_word}'!"
        elif confidence > 0.8:
            return f"✅ Correct! Good signing of '{target_word}'."
        else:
            return f"✅ Correct! You signed '{target_word}' - try for more clarity next time."
    else:
        if confidence > config.CONFIDENCE_THRESHOLD:
            return f"❌ I detected '{predicted_word}' but you were practicing '{target_word}'. Try again!"
        else:
            return f"🤔 I'm not confident about this gesture. Please sign more clearly and try again."

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({'error': 'Request too large - reduce video quality'}), 413

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    logger.info(f"🚀 Starting Sign Language API on {config.API_HOST}:{config.API_PORT}")
    
    try:
        app.run(
            host=config.API_HOST,
            port=config.API_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()