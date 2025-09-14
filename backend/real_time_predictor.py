#!/usr/bin/env python3
"""
Real-time Sign Language Recognition API
Instant detection with MediaPipe - No delays, no recording
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import base64
from collections import deque
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeSignRecognizer:
    def __init__(self, model_path='models/sign_language_model.h5'):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.word_mappings = None
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Recognition parameters
        self.sequence_length = 30
        self.feature_dim = 158
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.prediction_threshold = 0.7
        
        # Load model and artifacts
        self.load_model_artifacts()
        
        # Initialize MediaPipe
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        logger.info("Real-time sign recognizer initialized successfully")
    
    def load_model_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load scaler
            scaler_path = self.model_path.parent / 'scaler.pkl'
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
            
            # Load word mappings
            mappings_path = Path('dataset/word_mappings.json')
            with open(mappings_path, 'r') as f:
                self.word_mappings = json.load(f)
            logger.info(f"Loaded {len(self.word_mappings['words'])} word mappings")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def extract_frame_features(self, frame):
        """Extract MediaPipe features from frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            
            features = []
            
            # Hand features (126 features)
            if hand_results.multi_hand_landmarks:
                for hand_idx in range(2):
                    if hand_idx < len(hand_results.multi_hand_landmarks):
                        landmarks = hand_results.multi_hand_landmarks[hand_idx].landmark
                        for lm in landmarks:
                            features.extend([lm.x, lm.y, lm.z])
                    else:
                        features.extend([0.0] * 63)
            else:
                features.extend([0.0] * 126)
            
            # Pose features (32 features)
            pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                for idx in pose_indices:
                    lm = landmarks[idx]
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0.0] * 32)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def predict_sign(self, frame):
        """Predict sign from current frame"""
        try:
            # Extract features
            features = self.extract_frame_features(frame)
            
            # Add to buffer
            self.frame_buffer.append(features)
            
            # Need enough frames for prediction
            if len(self.frame_buffer) < self.sequence_length:
                return {
                    'predicted_word': None,
                    'confidence': 0.0,
                    'status': 'buffering',
                    'buffer_progress': len(self.frame_buffer) / self.sequence_length
                }
            
            # Prepare sequence
            sequence = np.array(list(self.frame_buffer))
            sequence = sequence.reshape(1, self.sequence_length, self.feature_dim)
            
            # Normalize
            sequence_flat = sequence.reshape(-1, self.feature_dim)
            sequence_normalized = self.scaler.transform(sequence_flat)
            sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.feature_dim)
            
            # Predict
            predictions = self.model.predict(sequence_normalized, verbose=0)[0]
            
            # Get top prediction
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            
            if confidence >= self.prediction_threshold:
                predicted_word = self.word_mappings['idx_to_word'][str(predicted_idx)]
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions)[-3:][::-1]
                top_predictions = [
                    {
                        'word': self.word_mappings['idx_to_word'][str(idx)],
                        'confidence': float(predictions[idx])
                    }
                    for idx in top_indices
                ]
                
                return {
                    'predicted_word': predicted_word,
                    'confidence': confidence,
                    'status': 'detected',
                    'top_predictions': top_predictions
                }
            else:
                return {
                    'predicted_word': None,
                    'confidence': confidence,
                    'status': 'low_confidence'
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'predicted_word': None,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def reset_buffer(self):
        """Reset frame buffer"""
        self.frame_buffer.clear()

# FastAPI app
app = FastAPI(title="Sign Language Recognition API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recognizer
recognizer = RealTimeSignRecognizer()

@app.get("/")
async def root():
    """API root"""
    return {
        "message": "Sign Language Recognition API",
        "status": "active",
        "total_words": len(recognizer.word_mappings['words']),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": recognizer.model is not None,
        "total_words": len(recognizer.word_mappings['words'])
    }

@app.get("/words")
async def get_supported_words():
    """Get list of supported words"""
    return {
        "words": recognizer.word_mappings['words'],
        "total_count": len(recognizer.word_mappings['words'])
    }

@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time recognition"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            
            try:
                # Parse frame data (base64 encoded image)
                frame_data = json.loads(data)
                
                if frame_data.get('type') == 'frame':
                    # Decode base64 image
                    image_data = base64.b64decode(frame_data['image'].split(',')[1])
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Predict sign
                        result = recognizer.predict_sign(frame)
                        
                        # Send result back
                        await websocket.send_text(json.dumps({
                            'type': 'prediction',
                            'data': result,
                            'timestamp': asyncio.get_event_loop().time()
                        }))
                
                elif frame_data.get('type') == 'reset':
                    # Reset buffer
                    recognizer.reset_buffer()
                    await websocket.send_text(json.dumps({
                        'type': 'reset_complete',
                        'data': {'status': 'buffer_reset'}
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'data': {'error': 'Invalid JSON format'}
                }))
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'data': {'error': str(e)}
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.post("/predict")
async def predict_from_image(request: dict):
    """HTTP endpoint for single image prediction"""
    try:
        if 'image' not in request:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64 image
        image_data = base64.b64decode(request['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Predict
        result = recognizer.predict_sign(frame)
        
        return {
            'success': True,
            'prediction': result
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_recognition():
    """Reset recognition buffer"""
    recognizer.reset_buffer()
    return {'success': True, 'message': 'Buffer reset successfully'}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )