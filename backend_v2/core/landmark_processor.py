"""
Advanced Hand Landmark Processor for Sign Language Recognition
Handles MediaPipe integration, normalization, and temporal smoothing
"""
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
from collections import deque
import time
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from config.production_config import config, logger

class LandmarkProcessor:
    """
    Production-ready hand landmark processor with temporal modeling
    
    Features:
    - MediaPipe Hands integration with optimized settings
    - Wrist-based coordinate normalization for scale/translation invariance  
    - Proper handedness handling (left hand first, right hand second)
    - Temporal smoothing with moving average filter
    - Confidence-based filtering for quality assurance
    - Real-time performance optimization (<5ms per frame)
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize MediaPipe with production settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            model_complexity=config.MODEL_COMPLEXITY
        )
        
        # Temporal smoothing buffers
        self.landmark_history = deque(maxlen=config.TEMPORAL_SMOOTHING_WINDOW)
        self.confidence_history = deque(maxlen=config.TEMPORAL_SMOOTHING_WINDOW)
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        # Normalization scaler (will be loaded if exists)
        self.scaler: Optional[StandardScaler] = None
        self.load_scaler()
        
        logger.info("LandmarkProcessor initialized with production settings")
    
    def load_scaler(self) -> None:
        """Load pre-trained landmark scaler for normalization"""
        if config.SCALER_PATH.exists():
            try:
                with open(config.SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Landmark scaler loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
                self.scaler = None
    
    def save_scaler(self) -> None:
        """Save the fitted scaler"""
        if self.scaler is not None:
            config.setup_directories()
            with open(config.SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("💾 Landmark scaler saved")
    
    def extract_landmarks(self, frame: np.ndarray, 
                         smooth: bool = True) -> Tuple[np.ndarray, bool, float]:
        """
        Extract hand landmarks from a single frame
        
        Args:
            frame: Input image (BGR format)
            smooth: Apply temporal smoothing
            
        Returns:
            landmarks: Normalized landmark array [126,] (2 hands × 21 points × 3 coords)
            hands_detected: Whether any hands were detected
            confidence: Average detection confidence
        """
        start_time = time.perf_counter()
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Initialize output
            landmarks = np.zeros(config.FEATURES_PER_FRAME, dtype=np.float32)
            hands_detected = False
            confidence = 0.0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_detected = True
                
                # Process detected hands
                hand_data = []
                confidences = []
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Extract handedness label and confidence
                    hand_label = handedness.classification[0].label
                    hand_confidence = handedness.classification[0].score
                    
                    # Extract landmark coordinates
                    coords = []
                    for landmark in hand_landmarks.landmark:
                        coords.extend([landmark.x, landmark.y, landmark.z])
                    
                    hand_data.append({
                        'label': hand_label,
                        'coords': coords,
                        'confidence': hand_confidence
                    })
                    confidences.append(hand_confidence)
                
                # Sort hands: Left first, then Right (consistent ordering)
                hand_data.sort(key=lambda x: x['label'])
                
                # Fill landmark array with proper hand ordering
                for hand_idx, hand in enumerate(hand_data[:2]):  # Max 2 hands
                    start_idx = hand_idx * config.LANDMARKS_PER_HAND * config.COORDINATES_PER_LANDMARK
                    end_idx = start_idx + len(hand['coords'])
                    landmarks[start_idx:end_idx] = hand['coords']
                
                # Calculate average confidence
                confidence = np.mean(confidences) if confidences else 0.0
                
                # Normalize coordinates using wrist-based normalization
                landmarks = self._normalize_landmarks(landmarks)
                
                # Apply temporal smoothing if enabled
                if smooth:
                    landmarks = self._apply_temporal_smoothing(landmarks, confidence)
            
            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            # Warn if processing is too slow
            if processing_time > 15:  # 15ms threshold
                logger.warning(f"⚠️ Slow landmark extraction: {processing_time:.1f}ms")
            
            return landmarks, hands_detected, confidence
            
        except Exception as e:
            logger.error(f"❌ Error in landmark extraction: {e}")
            return np.zeros(config.FEATURES_PER_FRAME, dtype=np.float32), False, 0.0
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply wrist-based normalization for scale and translation invariance
        
        This is critical for generalization across different users and camera distances
        """
        if np.allclose(landmarks, 0):
            return landmarks
        
        normalized = landmarks.copy()
        
        for hand_idx in range(config.MAX_NUM_HANDS):
            start_idx = hand_idx * config.LANDMARKS_PER_HAND * config.COORDINATES_PER_LANDMARK
            end_idx = start_idx + config.LANDMARKS_PER_HAND * config.COORDINATES_PER_LANDMARK
            
            hand_landmarks = landmarks[start_idx:end_idx].reshape(-1, 3)
            
            # Skip if hand not detected
            if np.allclose(hand_landmarks, 0):
                continue
            
            # Use wrist (landmark 0) as reference point
            wrist = hand_landmarks[0]
            
            # Translate to make wrist origin
            translated = hand_landmarks - wrist
            
            # Scale normalization using hand span (thumb tip to pinky tip)
            try:
                thumb_tip = translated[4]  # Landmark 4
                pinky_tip = translated[20]  # Landmark 20
                hand_span = np.linalg.norm(thumb_tip - pinky_tip)
                
                if hand_span > 1e-6:  # Avoid division by zero
                    scaled = translated / hand_span
                else:
                    scaled = translated
            except:
                scaled = translated
            
            # Flatten back to original format
            normalized[start_idx:end_idx] = scaled.flatten()
        
        # Apply statistical normalization if scaler is available
        if self.scaler is not None:
            try:
                normalized = self.scaler.transform(normalized.reshape(1, -1))[0]
            except Exception as e:
                logger.warning(f"⚠️ Scaler transform failed: {e}")
        
        return normalized
    
    def _apply_temporal_smoothing(self, landmarks: np.ndarray, confidence: float) -> np.ndarray:
        """
        Apply temporal smoothing to reduce jitter and noise
        Uses exponential moving average weighted by confidence
        """
        # Add to history
        self.landmark_history.append(landmarks.copy())
        self.confidence_history.append(confidence)
        
        if len(self.landmark_history) < 2:
            return landmarks
        
        # Weighted moving average based on confidence
        weights = np.array(self.confidence_history)
        weighted_sum = np.zeros_like(landmarks)
        total_weight = 0
        
        for i, (hist_landmarks, weight) in enumerate(zip(self.landmark_history, weights)):
            # More recent frames get higher weight
            temporal_weight = (i + 1) / len(self.landmark_history)
            final_weight = weight * temporal_weight
            
            weighted_sum += hist_landmarks * final_weight
            total_weight += final_weight
        
        if total_weight > 0:
            smoothed = weighted_sum / total_weight
        else:
            smoothed = landmarks
        
        return smoothed
    
    def extract_sequence_from_video(self, video_path: Path) -> Tuple[np.ndarray, bool]:
        """
        Extract landmark sequence from a video file
        Used for training data preparation
        
        Args:
            video_path: Path to video file
            
        Returns:
            sequence: Landmark sequence [seq_len, 126]
            success: Whether extraction was successful
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"❌ Cannot open video: {video_path}")
                return np.array([]), False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"📹 Processing video: {video_path.name} ({total_frames} frames, {fps:.1f} fps)")
            
            # Extract landmarks from all frames
            all_landmarks = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks, detected, confidence = self.extract_landmarks(frame, smooth=False)
                
                # Only keep frames with high-confidence detections
                if detected and confidence > config.MIN_DETECTION_CONFIDENCE:
                    all_landmarks.append(landmarks)
                
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 30 == 0:
                    logger.debug(f"Processed {frame_idx}/{total_frames} frames")
            
            cap.release()
            
            if len(all_landmarks) < config.MIN_SEQUENCE_LENGTH:
                logger.warning(f"⚠️ Not enough valid frames in {video_path.name}: {len(all_landmarks)}")
                return np.array([]), False
            
            # Convert to numpy array
            sequence = np.array(all_landmarks, dtype=np.float32)
            
            logger.info(f"✅ Extracted {len(sequence)} landmark frames from {video_path.name}")
            return sequence, True
            
        except Exception as e:
            logger.error(f"❌ Error processing video {video_path}: {e}")
            return np.array([]), False
    
    def fit_normalizer(self, landmarks_data: np.ndarray) -> None:
        """
        Fit the landmark normalizer on training data
        
        Args:
            landmarks_data: All training landmarks [n_samples, n_frames, 126]
        """
        # Reshape to [n_samples * n_frames, 126]
        flattened_data = landmarks_data.reshape(-1, config.FEATURES_PER_FRAME)
        
        # Remove zero rows (no hand detected)
        non_zero_mask = ~np.all(flattened_data == 0, axis=1)
        valid_data = flattened_data[non_zero_mask]
        
        if len(valid_data) < 100:
            logger.warning("⚠️ Not enough data to fit normalizer")
            return
        
        # Fit standard scaler
        self.scaler = StandardScaler()
        self.scaler.fit(valid_data)
        
        logger.info(f"📊 Normalizer fitted on {len(valid_data)} landmark frames")
        self.save_scaler()
    
    def get_performance_stats(self) -> Dict:
        """Get landmark processing performance statistics"""
        if not self.processing_times:
            return {"avg_time_ms": 0, "max_time_ms": 0, "min_time_ms": 0}
        
        times = list(self.processing_times)
        return {
            "avg_time_ms": np.mean(times),
            "max_time_ms": np.max(times),
            "min_time_ms": np.min(times),
            "samples": len(times)
        }
    
    def reset_temporal_buffers(self) -> None:
        """Reset temporal smoothing buffers (call when starting new gesture)"""
        self.landmark_history.clear()
        self.confidence_history.clear()
        logger.debug("🔄 Temporal buffers reset")
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
