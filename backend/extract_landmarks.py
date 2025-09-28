#!/usr/bin/env python3
"""
IMPROVED Landmark Extraction for Sign Language Recognition
Fixed for better accuracy and robust landmark detection
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedLandmarkExtractor:
    """Enhanced landmark extractor with better hand detection and normalization"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # MORE ROBUST hand detection settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # Increased from 0.5
            min_tracking_confidence=0.5,
            model_complexity=1  # Use full model
        )
        
        # Add pose detection for normalization reference
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_hand_landmarks(self, image):
        """
        Extract hand landmarks with better normalization
        Returns: normalized landmark vector (126 dims for 2 hands)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        hands_results = self.hands.process(rgb_image)
        # Detect pose for shoulder reference
        pose_results = self.pose.process(rgb_image)
        
        # Initialize landmark arrays
        left_hand_landmarks = np.zeros(63)  # 21 landmarks * 3 coords
        right_hand_landmarks = np.zeros(63)
        
        # Get image dimensions for normalization
        h, w = image.shape[:2]
        
        # Extract shoulder positions for better normalization
        shoulder_center = np.array([w/2, h/3])  # Default fallback
        if pose_results.pose_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[11]
            right_shoulder = pose_results.pose_landmarks.landmark[12]
            shoulder_center = np.array([
                (left_shoulder.x + right_shoulder.x) / 2 * w,
                (left_shoulder.y + right_shoulder.y) / 2 * h
            ])
        
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # Determine if left or right hand
                hand_label = handedness.classification[0].label
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # Convert to pixel coordinates
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z  # Relative depth
                    
                    # Normalize relative to shoulder center and image size
                    x_norm = (x - shoulder_center[0]) / w
                    y_norm = (y - shoulder_center[1]) / h
                    z_norm = z  # Keep relative depth as-is
                    
                    landmarks.extend([x_norm, y_norm, z_norm])
                
                # Assign to correct hand
                if hand_label == 'Left':
                    left_hand_landmarks = np.array(landmarks)
                else:  # Right hand
                    right_hand_landmarks = np.array(landmarks)
        
        # Combine both hands (left + right = 126 features)
        combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])
        
        return combined_landmarks
    
    def draw_landmarks_on_image(self, image, landmarks_vector):
        """Draw landmarks for visualization - creates green dots and connections"""
        if np.all(landmarks_vector == 0):
            return image
            
        h, w = image.shape[:2]
        annotated_image = image.copy()
        
        # Reconstruct shoulder center for denormalization
        shoulder_center = np.array([w/2, h/3])
        
        # Process both hands
        for hand_idx, hand_start in enumerate([0, 63]):  # Left: 0-62, Right: 63-125
            hand_landmarks = landmarks_vector[hand_start:hand_start+63]
            
            if np.any(hand_landmarks != 0):
                # Convert back to pixel coordinates
                points = []
                for i in range(0, 63, 3):
                    x_norm, y_norm, z_norm = hand_landmarks[i:i+3]
                    x_pixel = int(x_norm * w + shoulder_center[0])
                    y_pixel = int(y_norm * h + shoulder_center[1])
                    points.append((x_pixel, y_pixel))
                
                # Draw landmarks as green dots
                for point in points:
                    if 0 <= point[0] < w and 0 <= point[1] < h:
                        cv2.circle(annotated_image, point, 3, (0, 255, 0), -1)
                
                # Draw hand connections
                self._draw_hand_connections(annotated_image, points)
        
        return annotated_image
    
    def _draw_hand_connections(self, image, points):
        """Draw connections between hand landmarks"""
        # Hand landmark connections (standard MediaPipe hand model)
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17)  # Palm
        ]
        
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                
                # Only draw if both points are valid
                h, w = image.shape[:2]
                if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                    0 <= end_point[0] < w and 0 <= end_point[1] < h):
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    def process_video(self, video_path, max_frames=30):
        """
        Process video and extract landmark sequences
        Returns: list of landmark vectors (max_frames length)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []
        
        landmarks_sequence = []
        frame_count = 0
        
        # Get total frames for sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If video is longer than max_frames, sample uniformly
        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process frames we want to sample
            if current_frame in frame_indices:
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (640, 480))
                
                # Extract landmarks
                landmarks = self.extract_hand_landmarks(frame)
                landmarks_sequence.append(landmarks)
                
                frame_count += 1
                if frame_count >= max_frames:
                    break
            
            current_frame += 1
        
        cap.release()
        
        # Pad sequence if too short
        while len(landmarks_sequence) < max_frames:
            # Repeat last frame or use zeros
            if landmarks_sequence:
                landmarks_sequence.append(landmarks_sequence[-1])
            else:
                landmarks_sequence.append(np.zeros(126))
        
        return landmarks_sequence[:max_frames]


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from sign language videos")
    parser.add_argument('--videos_dir', type=str, default='dataset/videos', 
                       help='Directory containing word folders with videos')
    parser.add_argument('--output_dir', type=str, default='data/landmarks',
                       help='Output directory for landmark files')
    parser.add_argument('--csv_out', type=str, default='data/data.csv',
                       help='Output CSV manifest file')
    parser.add_argument('--max_frames', type=int, default=30,
                       help='Maximum frames to extract per video')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    
    # Check if videos directory exists, if not try alternative paths
    if not videos_dir.exists():
        alternative_paths = [
            Path('dataset') / 'videos',
            Path('main') / 'backend' / 'dataset' / 'videos',
            Path('backend') / 'dataset' / 'videos',
            Path('data') / 'videos'
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                videos_dir = alt_path
                logger.info(f"Found dataset at: {videos_dir.absolute()}")
                break
        else:
            logger.error(f"❌ Videos directory not found!")
            logger.error(f"Tried paths:")
            logger.error(f"  - {args.videos_dir}")
            for alt_path in alternative_paths:
                logger.error(f"  - {alt_path}")
            logger.error(f"\nCurrent working directory: {Path.cwd()}")
            logger.error(f"Please ensure your dataset structure is:")
            logger.error(f"  dataset/videos/")
            logger.error(f"    📁 word1/")
            logger.error(f"      📼 word1 (1).mp4")
            logger.error(f"      📼 word1 (2).mp4")
            return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
    
    extractor = ImprovedLandmarkExtractor()
    
    # Collect all video files
    video_files = []
    for word_folder in videos_dir.iterdir():
        if word_folder.is_dir():
            word = word_folder.name
            for video_file in word_folder.glob('*.mp4'):
                video_files.append({
                    'video_path': video_file,
                    'word': word,
                    'video_id': video_file.stem
                })
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Process videos and create manifest
    manifest_data = []
    
    for video_info in tqdm(video_files, desc="Processing videos"):
        video_path = video_info['video_path']
        word = video_info['word']
        video_id = video_info['video_id']
        
        # Extract landmarks
        landmarks_sequence = extractor.process_video(video_path, args.max_frames)
        
        if landmarks_sequence:
            # Save landmark sequence
            landmarks_file = output_dir / f"{word}_{video_id}.npy"
            np.save(landmarks_file, np.array(landmarks_sequence))
            
            # Add to manifest
            manifest_data.append({
                'filepath': str(landmarks_file.relative_to(Path('.'))),
                'label': word,
                'video_id': video_id,
                'num_frames': len(landmarks_sequence),
                'source_type': 'landmark_sequence'
            })
            
            # Optional: Save visualization
            if args.visualize and landmarks_sequence:
                # Use middle frame for visualization
                middle_idx = len(landmarks_sequence) // 2
                
                # Read the middle frame again for visualization
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    viz_image = extractor.draw_landmarks_on_image(frame, landmarks_sequence[middle_idx])
                    viz_path = viz_dir / f"{word}_{video_id}_landmarks.jpg"
                    cv2.imwrite(str(viz_path), viz_image)
    
    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(args.csv_out, index=False)
    
    logger.info(f"✅ Extraction completed!")
    logger.info(f"📊 Processed {len(manifest_data)} videos")
    logger.info(f"📁 Landmarks saved to: {output_dir}")
    logger.info(f"📋 Manifest saved to: {args.csv_out}")
    
    # Print class distribution
    class_counts = manifest_df['label'].value_counts()
    logger.info(f"📈 Class distribution:")
    for word, count in class_counts.head(10).items():
        logger.info(f"   {word}: {count} videos")


if __name__ == '__main__':
    main()