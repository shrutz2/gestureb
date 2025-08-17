"""
Video Processor for Sign Language Training Data
Extracts hand landmarks from your 299 videos and prepares training data
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
    """Extract hand landmarks from sign language videos"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            model_complexity=1  # Balance speed vs accuracy
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        logger.info("MediaPipe initialized")  # Removed emoji
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Extract hand landmarks from a single frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            landmarks: Flattened landmark array [126,] (2 hands × 21 points × 3 coords)
            hands_detected: Whether hands were detected
            confidence: Detection confidence
        """
        # Convert BGR to RGB
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
                hand_confidence = handedness.classification[0].score
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                
                # Extract coordinates
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
            
            # Fill landmark array
            for hand_idx, hand in enumerate(hand_data[:2]):  # Max 2 hands
                start_idx = hand_idx * 21 * 3  # 21 landmarks × 3 coordinates
                end_idx = start_idx + len(hand['coords'])
                landmarks[start_idx:end_idx] = hand['coords']
            
            confidence = np.mean(confidences) if confidences else 0.0
            
            # Simple normalization - center around wrist
            landmarks = self._normalize_landmarks(landmarks)
        
        return landmarks, hands_detected, confidence
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Simple wrist-centered normalization"""
        if np.allclose(landmarks, 0):
            return landmarks
        
        normalized = landmarks.copy()
        
        # Process each hand
        for hand_idx in range(2):  # 2 hands max
            start_idx = hand_idx * 21 * 3
            end_idx = start_idx + 21 * 3
            
            hand_landmarks = landmarks[start_idx:end_idx]
            
            if np.allclose(hand_landmarks, 0):
                continue
            
            # Reshape to [21, 3] for easier processing
            hand_coords = hand_landmarks.reshape(21, 3)
            
            # Use wrist (landmark 0) as reference
            wrist = hand_coords[0]
            
            if not np.allclose(wrist, 0):
                # Center around wrist
                centered = hand_coords - wrist
                
                # Scale by hand span (thumb tip to pinky tip)
                thumb_tip = centered[4]
                pinky_tip = centered[20]
                hand_span = np.linalg.norm(thumb_tip - pinky_tip)
                
                if hand_span > 1e-6:
                    scaled = centered / hand_span
                else:
                    scaled = centered
                
                # Store back
                normalized[start_idx:end_idx] = scaled.flatten()
        
        return normalized
    
    def process_video(self, video_path: Path) -> Tuple[np.ndarray, bool, dict]:
        """
        Extract landmark sequence from a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            sequence: Landmark sequence [n_frames, 126]
            success: Whether processing was successful
            stats: Processing statistics
        """
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")  # Removed emoji
            return np.array([]), False, {}
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")  # Removed emoji
            return np.array([]), False, {}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing: {video_path.name} ({total_frames} frames, {fps:.1f} fps, {duration:.1f}s)")  # Removed emoji
        
        all_landmarks = []
        valid_frames = 0
        confidence_sum = 0.0
        
        frame_idx = 0
        progress_bar = tqdm(total=total_frames, desc=f"Processing {video_path.stem}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            landmarks, detected, confidence = self.extract_landmarks_from_frame(frame)
            
            # Only keep frames with good detections
            if detected and confidence > 0.5:
                all_landmarks.append(landmarks)
                valid_frames += 1
                confidence_sum += confidence
            
            frame_idx += 1
            progress_bar.update(1)
        
        progress_bar.close()
        cap.release()
        
        # Statistics
        stats = {
            'total_frames': int(total_frames),  # Convert to Python int
            'valid_frames': int(valid_frames),  # Convert to Python int
            'fps': float(fps),  # Convert to Python float
            'duration': float(duration),  # Convert to Python float
            'avg_confidence': float(confidence_sum / max(1, valid_frames)),  # Convert to Python float
            'success_rate': float(valid_frames / max(1, total_frames))  # Convert to Python float
        }
        
        if valid_frames < 10:  # Minimum frames needed
            logger.warning(f"Too few valid frames for {video_path.name}: {valid_frames}")  # Removed emoji
            return np.array([]), False, stats
        
        sequence = np.array(all_landmarks, dtype=np.float32)
        logger.info(f"Extracted {len(sequence)} frames from {video_path.name}")  # Removed emoji
        
        return sequence, True, stats
    
    def process_all_videos(self, videos_dir: Path = None) -> dict:
        """
        Process all videos in directory and extract training data
        
        Args:
            videos_dir: Directory containing videos
            
        Returns:
            Dictionary with all extracted data
        """
        if videos_dir is None:
            videos_dir = config.VIDEOS_DIR
        
        if not videos_dir.exists():
            logger.error(f"Videos directory not found: {videos_dir}")  # Removed emoji
            return {}
        
        # Find all video files
        video_files = list(videos_dir.glob("*.mp4"))
        if not video_files:
            logger.error(f"No video files found in {videos_dir}")  # Removed emoji
            return {}
        
        logger.info(f"Found {len(video_files)} video files to process")  # Removed emoji
        
        # Get available words from label map
        parser = LabelMapParser()
        available_words = parser.get_all_available_words()
        
        all_sequences = []
        all_labels = []
        processing_stats = defaultdict(dict)
        failed_videos = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            word = video_file.stem.lower().strip()
            
            # Skip if word not in available words
            if available_words and word not in available_words:
                logger.warning(f"Skipping {word} - not in label map")  # Removed emoji
                continue
            
            # Process video
            sequence, success, stats = self.process_video(video_file)
            
            if success and len(sequence) >= 10:
                # Normalize sequence length
                normalized_sequence = self._normalize_sequence_length(sequence)
                
                all_sequences.append(normalized_sequence)
                all_labels.append(word)
                processing_stats[word] = stats
                
                logger.info(f"{word}: {len(sequence)} -> {len(normalized_sequence)} frames")  # Removed emoji
            else:
                failed_videos.append(video_file.name)
                logger.error(f"Failed to process: {video_file.name}")  # Removed emoji
        
        if not all_sequences:
            logger.error("No valid sequences extracted!")  # Removed emoji
            return {}
        
        # Convert to numpy arrays
        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels)
        
        # Summary statistics
        unique_words = np.unique(y)
        word_counts = {word: int(np.sum(y == word)) for word in unique_words}  # Convert to Python int
        
        result = {
            'X': X,
            'y': y,
            'num_classes': int(len(unique_words)),  # Convert to Python int
            'total_sequences': int(len(X)),  # Convert to Python int
            'word_counts': word_counts,
            'processing_stats': dict(processing_stats),
            'failed_videos': failed_videos,
            'shape': tuple(X.shape)  # Convert to tuple for JSON serialization
        }
        
        logger.info("Processing Summary:")  # Removed emoji
        logger.info(f"   Total sequences: {len(X)}")
        logger.info(f"   Unique words: {len(unique_words)}")
        logger.info(f"   Shape: {X.shape}")
        logger.info(f"   Failed videos: {len(failed_videos)}")
        
        # Save the data
        self._save_training_data(result)
        
        return result
    
    def _normalize_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence to fixed length"""
        target_length = config.SEQUENCE_LENGTH
        
        if len(sequence) == target_length:
            return sequence
        elif len(sequence) > target_length:
            # Uniformly sample frames
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return sequence[indices]
        else:
            # Pad by repeating frames
            padding_needed = target_length - len(sequence)
            
            if len(sequence) > 0:
                # Repeat sequence cyclically
                repeated_indices = np.tile(np.arange(len(sequence)), 
                                         padding_needed // len(sequence) + 1)[:padding_needed]
                padding = sequence[repeated_indices]
                return np.vstack([sequence, padding])
            else:
                return np.zeros((target_length, config.FEATURES_PER_FRAME), dtype=np.float32)
    
    def _save_training_data(self, data: dict) -> None:
        """Save training data to disk"""
        # Save as NPZ (compressed numpy format)
        np.savez_compressed(
            config.TRAINING_DATA_PATH,
            X=data['X'],
            y=data['y'],
            word_counts=np.array(list(data['word_counts'].values()))  # Convert dict to array for numpy
        )
        
        # Save metadata as JSON with proper type conversion
        metadata = {
            'num_classes': int(data['num_classes']),
            'total_sequences': int(data['total_sequences']),
            'shape': list(data['shape']),  # Convert tuple to list
            'word_counts': {k: int(v) for k, v in data['word_counts'].items()},  # Ensure all values are Python ints
            'failed_videos': data['failed_videos']
        }
        
        metadata_path = config.DATA_DIR / "training_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
            json.dump(metadata, f, indent=2, ensure_ascii=False)  # Allow unicode characters
        
        logger.info(f"Training data saved:")  # Removed emoji
        logger.info(f"   Data: {config.TRAINING_DATA_PATH}")
        logger.info(f"   Metadata: {metadata_path}")

def main():
    """Process all videos and prepare training data"""
    processor = VideoProcessor()
    
    # Process all videos
    data = processor.process_all_videos()
    
    if data:
        print("\nTraining Data Summary:")  # Removed emoji
        print("=" * 50)
        print(f"Total sequences: {data['total_sequences']}")
        print(f"Unique words: {data['num_classes']}")
        print(f"Data shape: {data['shape']}")
        
        print(f"\nWord distribution:")
        for word, count in sorted(data['word_counts'].items()):
            print(f"  {word}: {count} sequence(s)")
        
        if data['failed_videos']:
            print(f"\nFailed videos ({len(data['failed_videos'])}):")  # Removed emoji
            for video in data['failed_videos'][:10]:  # Show first 10
                print(f"  {video}")
            if len(data['failed_videos']) > 10:
                print(f"  ... and {len(data['failed_videos']) - 10} more")
    
    print(f"\nVideo processing complete!")  # Removed emoji
    print(f"Next step: Run 'python simplified_trainer.py' to train the model")

if __name__ == "__main__":
    main()