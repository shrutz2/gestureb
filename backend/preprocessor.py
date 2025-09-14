#!/usr/bin/env python3
"""
Professional Real-time Sign Language Dataset Processor
Handles 200 words with MediaPipe feature extraction
Optimized for real-time inference performance
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignLanguageProcessor:
    def __init__(self, dataset_path='dataset/videos', output_path='models/'):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Optimized for real-time performance
        self.sequence_length = 20  # Frames for temporal modeling
        self.hand_landmarks = 21
        self.pose_landmarks = 6    # Key upper body points
        self.feature_dim = (2 * self.hand_landmarks * 3) + (self.pose_landmarks * 3)  # 144 features
        
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        logger.info(f"Real-time Sign Language Processor initialized")
        logger.info(f"Target: 200 words from your dataset")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Sequence length: {self.sequence_length}")
    
    def create_mediapipe_models(self):
        """Create MediaPipe models for processing"""
        hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        return hands, pose
    
    def extract_features_from_frame(self, frame, hands, pose):
        """Extract hand and pose landmarks from a single frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)
            
            features = []
            
            # Extract hand landmarks (126 features for 2 hands)
            if hand_results.multi_hand_landmarks:
                for hand_idx in range(2):
                    if hand_idx < len(hand_results.multi_hand_landmarks):
                        landmarks = hand_results.multi_hand_landmarks[hand_idx].landmark
                        for lm in landmarks:
                            features.extend([lm.x, lm.y, lm.z])
                    else:
                        features.extend([0.0] * 63)  # Zero padding for missing hand
            else:
                features.extend([0.0] * 126)  # No hands detected
            
            # Extract key pose landmarks (18 features)
            # Focus on shoulders, elbows, wrists for sign language context
            pose_indices = [11, 12, 13, 14, 15, 16]  # Upper body keypoints
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                for idx in pose_indices:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        features.extend([lm.x, lm.y, lm.z])
                    else:
                        features.extend([0.0] * 3)
            else:
                features.extend([0.0] * 18)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def process_video(self, video_path, word_label):
        """Process a single video and extract feature sequence"""
        try:
            hands, pose = self.create_mediapipe_models()
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path}")
                return None
            
            frames_features = []
            frame_count = 0
            max_frames = 60  # Limit processing for efficiency
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for consistency
                frame = cv2.resize(frame, (640, 480))
                
                # Extract features
                features = self.extract_features_from_frame(frame, hands, pose)
                
                # Only keep frames with detected landmarks
                if np.sum(np.abs(features)) > 0:
                    frames_features.append(features)
                
                frame_count += 1
            
            cap.release()
            hands.close()
            pose.close()
            
            # Need minimum frames for temporal modeling
            if len(frames_features) < 5:
                logger.warning(f"Insufficient features for {video_path}")
                return None
            
            # Normalize sequence to fixed length
            normalized_sequence = self.normalize_sequence(frames_features)
            
            return {
                'features': normalized_sequence,
                'label': word_label,
                'video_path': str(video_path),
                'original_frames': len(frames_features)
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return None
    
    def normalize_sequence(self, sequence):
        """Normalize sequence to fixed length for LSTM input"""
        sequence = np.array(sequence)
        current_length = len(sequence)
        
        if current_length == self.sequence_length:
            return sequence
        elif current_length > self.sequence_length:
            # Sample uniformly across the sequence
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            return sequence[indices]
        else:
            # Pad with zeros and repeat last frame
            padded = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
            padded[:current_length] = sequence
            # Fill remaining with last frame to maintain temporal consistency
            if current_length > 0:
                padded[current_length:] = sequence[-1]
            return padded
    
    def load_and_process_dataset(self):
        """Load and process the complete dataset"""
        logger.info("Loading your sign language dataset...")
        
        # Get all video files organized by word
        word_videos = {}
        video_extensions = ['.mp4', '.avi', '.mov', '.webm']
        
        for word_dir in self.dataset_path.iterdir():
            if word_dir.is_dir():
                word = word_dir.name.lower().strip()
                videos = []
                for ext in video_extensions:
                    videos.extend(list(word_dir.glob(f'*{ext}')))
                
                if videos:
                    word_videos[word] = videos
        
        logger.info(f"Found {len(word_videos)} words in dataset")
        
        # Process all videos
        processed_data = []
        failed_count = 0
        
        # Sequential processing to avoid MediaPipe conflicts
        for word, video_paths in tqdm(word_videos.items(), desc="Processing words"):
            for video_path in video_paths:
                result = self.process_video(video_path, word)
                if result is not None:
                    processed_data.append(result)
                else:
                    failed_count += 1
        
        success_rate = len(processed_data) / (len(processed_data) + failed_count) * 100
        logger.info(f"Processing complete:")
        logger.info(f"  Successful: {len(processed_data)} videos")
        logger.info(f"  Failed: {failed_count} videos")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        
        return processed_data, list(word_videos.keys())
    
    def prepare_training_data(self, processed_data):
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        if not processed_data:
            raise ValueError("No processed data available!")
        
        # Extract features and labels
        X = np.array([item['features'] for item in processed_data])
        y = [item['label'] for item in processed_data]
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data (70% train, 15% val, 15% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        # Create word mappings for real-time inference
        word_to_idx = {word: idx for idx, word in enumerate(label_encoder.classes_)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        dataset = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'label_encoder': label_encoder,
            'num_classes': len(label_encoder.classes_),
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'class_names': label_encoder.classes_.tolist()
        }
        
        logger.info(f"Training data prepared:")
        logger.info(f"  Total words: {dataset['num_classes']}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Test samples: {len(X_test)}")
        
        return dataset
    
    def save_processed_dataset(self, dataset):
        """Save the processed dataset"""
        # Save complete dataset
        dataset_path = self.output_path / 'processed_dataset.pkl'
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save word mappings for real-time inference
        mappings = {
            'word_to_idx': dataset['word_to_idx'],
            'idx_to_word': dataset['idx_to_word'],
            'num_classes': dataset['num_classes'],
            'class_names': dataset['class_names'],
            'feature_dim': dataset['feature_dim'],
            'sequence_length': dataset['sequence_length']
        }
        
        mappings_path = self.output_path / 'word_mappings.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f"Dataset saved to: {dataset_path}")
        logger.info(f"Mappings saved to: {mappings_path}")
        
        return dataset_path

def main():
    """Main preprocessing pipeline"""
    print("🔥 REAL-TIME SIGN LANGUAGE PREPROCESSING")
    print("=" * 60)
    print("📊 Processing your 200-word dataset")
    print("🎯 Target: 90%+ accuracy for real-time recognition")
    print("=" * 60)
    
    try:
        # Initialize processor
        processor = SignLanguageProcessor()
        
        # Process dataset
        processed_data, word_list = processor.load_and_process_dataset()
        
        if not processed_data:
            logger.error("No data processed successfully!")
            return
        
        # Prepare training data
        dataset = processor.prepare_training_data(processed_data)
        
        # Save dataset
        dataset_path = processor.save_processed_dataset(dataset)
        
        print("\n✅ PREPROCESSING COMPLETED!")
        print(f"📈 Ready to train on {dataset['num_classes']} words")
        print(f"🎬 Total videos processed: {len(processed_data)}")
        print(f"📊 Training samples: {len(dataset['X_train'])}")
        print(f"✅ Validation samples: {len(dataset['X_val'])}")
        print(f"🧪 Test samples: {len(dataset['X_test'])}")
        print(f"\n🚀 Next: python src/model_trainer.py")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()