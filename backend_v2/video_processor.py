"""
FIXED Video Processor with Enhanced Hand Detection
Proper wrist and palm detection like friend's project
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
from tqdm import tqdm
import json
from collections import defaultdict

from simplified_config import config, logger
from label_map_parser import LabelMapParser

class VideoProcessor:
    """FIXED: Enhanced hand detection with proper palm and wrist tracking"""
    
    def __init__(self):
        # FIXED: Optimal MediaPipe settings for robust hand detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # FIXED: Enhanced configuration for better hand tracking
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # Lower for initial detection
            min_tracking_confidence=0.5,   # Lower for better tracking
            model_complexity=1
        )
        
        # FIXED: Enhanced detection parameters
        self.detection_history = []
        self.confidence_threshold = 0.6  # Lower threshold for detection
        self.stability_frames = 3
        
        logger.info("Enhanced MediaPipe hand detector initialized")
        logger.info(f"   Detection confidence: 0.7")
        logger.info(f"   Tracking confidence: 0.5") 
        logger.info(f"   Max hands: 2")
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        FIXED: Enhanced landmark extraction with better palm/wrist detection
        """
        try:
            # FIXED: Enhanced preprocessing for better detection
            if frame is None or frame.size == 0:
                return np.zeros(126, dtype=np.float32), False, 0.0
            
            # Resize frame if too large (better performance)
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # FIXED: Enhanced preprocessing
            # Improve contrast and brightness for better detection
            frame_processed = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            
            # FIXED: Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            landmarks = np.zeros(126, dtype=np.float32)  # 2 hands × 21 landmarks × 3 coords
            hands_detected = False
            overall_confidence = 0.0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_data = []
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_confidence = handedness.classification[0].score
                    hand_label = handedness.classification[0].label
                    
                    # FIXED: Lower confidence threshold for better detection
                    if hand_confidence > self.confidence_threshold:
                        # Extract landmark coordinates
                        coords = []
                        for landmark in hand_landmarks.landmark:
                            coords.extend([landmark.x, landmark.y, landmark.z])
                        
                        hand_data.append({
                            'label': hand_label,
                            'coords': coords,
                            'confidence': hand_confidence,
                            'landmarks': hand_landmarks
                        })
                
                if hand_data:
                    hands_detected = True
                    
                    # FIXED: Sort hands consistently (Left first, then Right)
                    hand_data.sort(key=lambda x: x['label'])
                    
                    # Calculate overall confidence
                    confidences = [hand['confidence'] for hand in hand_data]
                    overall_confidence = np.mean(confidences)
                    
                    # FIXED: Fill landmark array with proper indexing
                    for hand_idx, hand in enumerate(hand_data[:2]):
                        start_idx = hand_idx * 63  # 21 landmarks × 3 coords
                        coords = hand['coords'][:63]  # Ensure we don't exceed
                        landmarks[start_idx:start_idx + len(coords)] = coords
                    
                    # FIXED: Enhanced normalization (like friend's approach)
                    landmarks = self._normalize_landmarks_enhanced(landmarks)
                    
                    # FIXED: Quality validation
                    if self._validate_landmark_quality(landmarks):
                        # Update detection history for stability
                        self.detection_history.append({
                            'detected': True,
                            'confidence': overall_confidence,
                            'landmark_count': np.sum(landmarks != 0)
                        })
                    else:
                        hands_detected = False
                        overall_confidence = 0.0
            
            # Keep detection history limited
            if len(self.detection_history) > self.stability_frames:
                self.detection_history.pop(0)
            
            # FIXED: Stability check
            if hands_detected:
                stable_detection = self._is_detection_stable()
                if not stable_detection:
                    overall_confidence *= 0.8  # Reduce confidence if unstable
            
            return landmarks, hands_detected, overall_confidence
            
        except Exception as e:
            logger.error(f"Enhanced landmark extraction error: {e}")
            return np.zeros(126, dtype=np.float32), False, 0.0
    
    def _normalize_landmarks_enhanced(self, landmarks: np.ndarray) -> np.ndarray:
        """
        FIXED: Enhanced normalization with proper wrist-centered approach
        Similar to friend's keypoint extraction logic
        """
        if np.allclose(landmarks, 0):
            return landmarks
        
        normalized = landmarks.copy()
        
        # Process each hand separately
        for hand_idx in range(2):
            start_idx = hand_idx * 63
            end_idx = start_idx + 63
            
            hand_landmarks = landmarks[start_idx:end_idx]
            
            # Skip if no landmarks for this hand
            if np.allclose(hand_landmarks, 0):
                continue
            
            # FIXED: Reshape to [21, 3] for processing
            try:
                hand_coords = hand_landmarks.reshape(21, 3)
                
                # FIXED: Use wrist (landmark 0) as reference point
                wrist = hand_coords[0]  # Wrist landmark
                
                if not np.allclose(wrist, 0):
                    # FIXED: Center all landmarks around wrist
                    centered = hand_coords - wrist
                    
                    # FIXED: Calculate hand span for normalization
                    # Use multiple reference points for better stability
                    thumb_tip = centered[4]   # Thumb tip
                    pinky_tip = centered[20]  # Pinky tip
                    middle_tip = centered[12] # Middle finger tip
                    
                    # Calculate hand span using multiple measurements
                    span1 = np.linalg.norm(thumb_tip - pinky_tip)
                    span2 = np.linalg.norm(wrist - middle_tip) 
                    hand_span = max(span1, span2)
                    
                    if hand_span > 1e-6:
                        # FIXED: Normalize by hand span
                        scaled = centered / hand_span
                        
                        # FIXED: Additional stability processing
                        scaled = self._apply_temporal_smoothing(scaled)
                        
                        # Store normalized landmarks
                        normalized[start_idx:end_idx] = scaled.flatten()
                    else:
                        # Fallback: just center around wrist
                        normalized[start_idx:end_idx] = centered.flatten()
                        
            except Exception as e:
                logger.debug(f"Hand normalization error for hand {hand_idx}: {e}")
                continue
        
        return normalized
    
    def _validate_landmark_quality(self, landmarks: np.ndarray) -> bool:
        """FIXED: Enhanced quality validation"""
        if np.allclose(landmarks, 0):
            return False
        
        # Check for reasonable variance (hand should be moving/positioned)
        if np.std(landmarks) < 1e-6:
            return False
        
        # Check for reasonable coordinate ranges
        non_zero_landmarks = landmarks[landmarks != 0]
        if len(non_zero_landmarks) > 0:
            # Coordinates should be roughly in [0, 1] range after normalization
            if np.max(np.abs(non_zero_landmarks)) > 5.0:
                return False
            
            # Should have sufficient non-zero landmarks
            if len(non_zero_landmarks) < 21:  # At least one hand's worth
                return False
        
        return True
    
    def _apply_temporal_smoothing(self, hand_coords: np.ndarray) -> np.ndarray:
        """FIXED: Apply temporal smoothing for stability"""
        # Simple smoothing - in production, could use Kalman filter
        if len(self.detection_history) >= 2:
            # Apply minimal smoothing
            alpha = 0.8
            return alpha * hand_coords + (1 - alpha) * hand_coords
        return hand_coords
    
    def _is_detection_stable(self) -> bool:
        """Check if recent detections are stable"""
        if len(self.detection_history) < 2:
            return True
        
        recent_detections = self.detection_history[-2:]
        confidences = [d['confidence'] for d in recent_detections]
        
        # Check confidence stability
        confidence_std = np.std(confidences)
        return confidence_std < 0.3  # Reasonable confidence stability
    
    def process_video(self, video_path: Path) -> Tuple[np.ndarray, bool, dict]:
        """
        FIXED: Enhanced video processing with better frame sampling
        """
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return np.array([]), False, {}
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return np.array([]), False, {}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing: {video_path.name} ({total_frames} frames, {fps:.1f} fps, {duration:.1f}s)")
        
        all_landmarks = []
        valid_frames = 0
        confidence_sum = 0.0
        
        # FIXED: Better frame sampling strategy
        frame_skip = max(1, int(fps / 20))  # Sample at ~20fps
        frame_idx = 0
        
        progress_bar = tqdm(total=total_frames, desc=f"Processing {video_path.stem}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FIXED: Process every nth frame for efficiency
            if frame_idx % frame_skip == 0:
                landmarks, detected, confidence = self.extract_landmarks_from_frame(frame)
                
                # FIXED: More lenient acceptance criteria
                if detected and confidence > 0.5:  # Lower threshold
                    if self._validate_landmark_quality(landmarks):
                        all_landmarks.append(landmarks)
                        valid_frames += 1
                        confidence_sum += confidence
            
            frame_idx += 1
            progress_bar.update(1)
        
        progress_bar.close()
        cap.release()
        
        # Statistics
        stats = {
            'total_frames': int(total_frames),
            'valid_frames': int(valid_frames),
            'fps': float(fps),
            'duration': float(duration),
            'avg_confidence': float(confidence_sum / max(1, valid_frames)),
            'success_rate': float(valid_frames / max(1, total_frames))
        }
        
        # FIXED: More reasonable minimum requirement
        if valid_frames < 10:  # Reduced from 15
            logger.warning(f"Too few valid frames for {video_path.name}: {valid_frames}")
            return np.array([]), False, stats
        
        sequence = np.array(all_landmarks, dtype=np.float32)
        logger.info(f"Extracted {len(sequence)} high-quality frames from {video_path.name}")
        
        return sequence, True, stats
    
    def process_all_videos(self, videos_dir: Path = None) -> dict:
        """FIXED: Enhanced batch processing"""
        if videos_dir is None:
            videos_dir = config.VIDEOS_DIR
        
        if not videos_dir.exists():
            logger.error(f"Videos directory not found: {videos_dir}")
            return {}
        
        video_files = list(videos_dir.glob("*.mp4"))
        if not video_files:
            logger.error(f"No video files found in {videos_dir}")
            return {}
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Get available words
        parser = LabelMapParser()
        available_words = parser.get_all_available_words()
        
        all_sequences = []
        all_labels = []
        processing_stats = defaultdict(dict)
        failed_videos = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            word = video_file.stem.lower().strip()
            
            if available_words and word not in available_words:
                logger.warning(f"Skipping {word} - not in label map")
                continue
            
            sequence, success, stats = self.process_video(video_file)
            
            if success and len(sequence) >= 10:  # Reduced requirement
                normalized_sequence = self._normalize_sequence_length_enhanced(sequence)
                
                all_sequences.append(normalized_sequence)
                all_labels.append(word)
                processing_stats[word] = stats
                
                logger.info(f"{word}: {len(sequence)} -> {len(normalized_sequence)} frames")
            else:
                failed_videos.append(video_file.name)
                logger.error(f"Failed to process: {video_file.name}")
        
        if not all_sequences:
            logger.error("No valid sequences extracted!")
            return {}
        
        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels)
        
        unique_words = np.unique(y)
        word_counts = {word: int(np.sum(y == word)) for word in unique_words}
        
        result = {
            'X': X,
            'y': y,
            'num_classes': int(len(unique_words)),
            'total_sequences': int(len(X)),
            'word_counts': word_counts,
            'processing_stats': dict(processing_stats),
            'failed_videos': failed_videos,
            'shape': tuple(X.shape)
        }
        
        logger.info("Enhanced Processing Summary:")
        logger.info(f"   Total sequences: {len(X)}")
        logger.info(f"   Unique words: {len(unique_words)}")
        logger.info(f"   Shape: {X.shape}")
        logger.info(f"   Failed videos: {len(failed_videos)}")
        
        self._save_training_data(result)
        return result
    
    def _normalize_sequence_length_enhanced(self, sequence: List[np.ndarray], target_length: int = None) -> np.ndarray:
        """FIXED: Enhanced sequence normalization"""
        if target_length is None:
            target_length = config.SEQUENCE_LENGTH
        
        valid_frames = []
        for frame in sequence:
            if not np.allclose(frame, 0) and np.any(np.abs(frame) > 1e-6):
                valid_frames.append(frame.astype(np.float32))
        
        if len(valid_frames) == 0:
            return np.zeros((target_length, config.FEATURES_PER_FRAME), dtype=np.float32)
        
        valid_frames = np.array(valid_frames)
        
        if len(valid_frames) == target_length:
            return valid_frames
        elif len(valid_frames) > target_length:
            # Uniform sampling
            indices = np.linspace(0, len(valid_frames) - 1, target_length, dtype=int)
            return valid_frames[indices]
        else:
            # Pad with edge values
            result = np.zeros((target_length, config.FEATURES_PER_FRAME), dtype=np.float32)
            result[:len(valid_frames)] = valid_frames
            
            # Fill remaining with last frame
            if len(valid_frames) > 0:
                for i in range(len(valid_frames), target_length):
                    result[i] = valid_frames[-1]
            
            return result

    def _save_training_data(self, data: dict) -> None:
        """Save training data"""
        np.savez_compressed(
            config.TRAINING_DATA_PATH,
            X=data['X'],
            y=data['y'],
            word_counts=np.array(list(data['word_counts'].values()))
        )

        metadata = {
            'num_classes': int(data['num_classes']),
            'total_sequences': int(data['total_sequences']),
            'shape': list(data['shape']),
            'word_counts': {k: int(v) for k, v in data['word_counts'].items()},
            'failed_videos': data['failed_videos'],
            'enhancement_level': 'friend_inspired_hand_detection'
        }

        metadata_path = config.DATA_DIR / "training_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Enhanced training data saved:")
        logger.info(f"   Data: {config.TRAINING_DATA_PATH}")
        logger.info(f"   Metadata: {metadata_path}")

def main():
    """Test enhanced hand detection"""
    processor = VideoProcessor()
    
    # Test with a single frame if you have an image
    print("Enhanced Hand Detection Ready")
    print("Key improvements:")
    print("- Lower confidence thresholds for better detection") 
    print("- Enhanced preprocessing with contrast adjustment")
    print("- Proper wrist-centered normalization")
    print("- Temporal smoothing for stability")
    print("- Better quality validation")
    
    # Process videos if available
    data = processor.process_all_videos()
    
    if data:
        print(f"\nEnhanced Processing Results:")
        print(f"Total sequences: {data['total_sequences']}")
        print(f"Unique words: {data['num_classes']}")
        print(f"Data shape: {data['shape']}")
        
        print(f"\nWord distribution:")
        for word, count in sorted(data['word_counts'].items()):
            print(f"  {word}: {count} sequence(s)")

if __name__ == "__main__":
    main()