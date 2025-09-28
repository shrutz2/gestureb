#!/usr/bin/env python3
"""
ENHANCED Data Preprocessing for Sign Language Recognition
Improved for better accuracy and robust data handling
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)

class EnhancedSignLanguagePreprocessor:
    """Enhanced preprocessing with better normalization and augmentation"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Use RobustScaler instead of StandardScaler for better outlier handling
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        
        set_seed(random_state)
        
    def load_landmark_sequences(self, landmark_files, max_sequence_length=30):
        """
        Load landmark sequences with improved error handling and filtering
        """
        sequences = []
        labels = []
        metadata = []
        
        logger.info(f"Loading {len(landmark_files)} landmark files...")
        
        valid_count = 0
        invalid_count = 0
        
        for file_info in landmark_files:
            filepath = file_info['filepath']
            label = file_info['label']
            
            try:
                # Load numpy array
                landmarks_seq = np.load(filepath)
                
                # Validate sequence shape and content
                if self._is_valid_sequence(landmarks_seq, max_sequence_length):
                    # Ensure exact sequence length
                    landmarks_seq = self._standardize_sequence_length(landmarks_seq, max_sequence_length)
                    
                    sequences.append(landmarks_seq)
                    labels.append(label)
                    metadata.append(file_info)
                    valid_count += 1
                else:
                    invalid_count += 1
                    logger.debug(f"Invalid sequence: {filepath} - shape: {landmarks_seq.shape}")
                    
            except Exception as e:
                invalid_count += 1
                logger.debug(f"Error loading {filepath}: {e}")
        
        logger.info(f"✅ Loaded {valid_count} valid sequences, {invalid_count} invalid/failed")
        
        if not sequences:
            raise ValueError("No valid sequences found! Check your landmark files.")
            
        return np.array(sequences), np.array(labels), metadata
    
    def _is_valid_sequence(self, seq, max_length):
        """Check if sequence is valid for training"""
        if seq.ndim != 2:
            return False
        if seq.shape[0] == 0 or seq.shape[1] != 126:  # 126 features for 2 hands
            return False
        if seq.shape[0] > max_length * 2:  # Too long
            return False
        # Check if sequence has some actual hand detection (not all zeros)
        if np.all(seq == 0):
            return False
        return True
    
    def _standardize_sequence_length(self, seq, target_length):
        """Ensure all sequences have exact target length"""
        current_length = seq.shape[0]
        
        if current_length == target_length:
            return seq
        elif current_length > target_length:
            # Downsample uniformly
            indices = np.linspace(0, current_length-1, target_length, dtype=int)
            return seq[indices]
        else:
            # Upsample by repeating frames
            repeat_factor = target_length // current_length
            remainder = target_length % current_length
            
            repeated_seq = np.repeat(seq, repeat_factor, axis=0)
            if remainder > 0:
                # Add extra frames from the beginning
                extra_frames = seq[:remainder]
                repeated_seq = np.concatenate([repeated_seq, extra_frames], axis=0)
            
            return repeated_seq[:target_length]
    
    def apply_data_augmentation(self, X, y, augment_factor=2):
        """
        Apply landmark-specific data augmentation
        """
        logger.info(f"Applying data augmentation with factor {augment_factor}...")
        
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            sequence = X[i]
            label = y[i]
            
            # Original sequence
            augmented_X.append(sequence)
            augmented_y.append(label)
            
            # Generate augmented versions
            for _ in range(augment_factor - 1):
                aug_sequence = self._augment_sequence(sequence)
                augmented_X.append(aug_sequence)
                augmented_y.append(label)
        
        logger.info(f"✅ Augmentation complete: {len(X)} → {len(augmented_X)} sequences")
        return np.array(augmented_X), np.array(augmented_y)
    
    def _augment_sequence(self, sequence):
        """Apply augmentation to a single sequence"""
        aug_seq = sequence.copy()
        
        # 1. Add small Gaussian noise
        noise = np.random.normal(0, 0.01, aug_seq.shape)
        aug_seq += noise
        
        # 2. Small rotation around wrist (approximate)
        angle = np.random.uniform(-5, 5) * np.pi / 180  # ±5 degrees
        aug_seq = self._rotate_landmarks(aug_seq, angle)
        
        # 3. Small scaling
        scale_factor = np.random.uniform(0.95, 1.05)
        aug_seq *= scale_factor
        
        # 4. Temporal augmentation (slight speed variation)
        if np.random.random() > 0.5:
            # Randomly drop 1-2 frames and duplicate others
            frames_to_drop = np.random.choice(len(aug_seq), size=min(2, len(aug_seq)//10), replace=False)
            mask = np.ones(len(aug_seq), dtype=bool)
            mask[frames_to_drop] = False
            
            # Remove frames and pad by duplicating random frames
            aug_seq_filtered = aug_seq[mask]
            while len(aug_seq_filtered) < len(aug_seq):
                idx_to_duplicate = np.random.randint(len(aug_seq_filtered))
                aug_seq_filtered = np.insert(aug_seq_filtered, idx_to_duplicate, aug_seq_filtered[idx_to_duplicate], axis=0)
            
            aug_seq = aug_seq_filtered[:len(aug_seq)]
        
        return aug_seq
    
    def _rotate_landmarks(self, sequence, angle):
        """Apply small rotation to hand landmarks"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_seq = sequence.copy()
        
        # Apply rotation to x,y coordinates (skip z)
        for frame_idx in range(len(sequence)):
            for hand_start in [0, 63]:  # Left and right hand
                for joint_idx in range(21):  # 21 joints per hand
                    coord_start = hand_start + joint_idx * 3
                    x, y = sequence[frame_idx, coord_start:coord_start+2]
                    
                    # Apply rotation
                    rotated_xy = rotation_matrix @ np.array([x, y])
                    rotated_seq[frame_idx, coord_start:coord_start+2] = rotated_xy
        
        return rotated_seq
    
    def filter_classes_by_sample_count(self, X, y, min_samples=5):
        """Remove classes with insufficient samples"""
        label_counts = Counter(y)
        valid_labels = {label for label, count in label_counts.items() if count >= min_samples}
        
        if len(valid_labels) < len(label_counts):
            logger.info(f"Filtering classes: {len(label_counts)} → {len(valid_labels)} classes")
            
            # Create mask for valid samples
            valid_mask = np.array([label in valid_labels for label in y])
            X_filtered = X[valid_mask]
            y_filtered = y[valid_mask]
            
            logger.info(f"Samples after filtering: {len(X)} → {len(X_filtered)}")
            return X_filtered, y_filtered
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """Enhanced stratified splitting"""
        logger.info(f"Splitting data: train/val/test with test_size={test_size}, val_size={val_size}")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=self.random_state
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_labels(self, y_train, y_val, y_test):
        """Encode string labels to integers"""
        logger.info("Encoding labels...")
        
        # Fit encoder on training data only
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Classes: {self.label_encoder.classes_[:10]}...")  # Show first 10
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def normalize_features(self, X_train, X_val, X_test):
        """Enhanced feature normalization"""
        logger.info("Normalizing features...")
        
        # Reshape for per-feature normalization across all frames
        original_shape = X_train.shape
        
        # Flatten temporal dimension: (samples, time, features) → (samples*time, features)
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data only
        X_train_normalized = self.scaler.fit_transform(X_train_flat)
        X_val_normalized = self.scaler.transform(X_val_flat)
        X_test_normalized = self.scaler.transform(X_test_flat)
        
        # Reshape back to original
        X_train_normalized = X_train_normalized.reshape(original_shape)
        X_val_normalized = X_val_normalized.reshape(X_val.shape)
        X_test_normalized = X_test_normalized.reshape(X_test.shape)
        
        logger.info("✅ Normalization completed")
        return X_train_normalized, X_val_normalized, X_test_normalized
    
    def compute_class_weights(self, y_encoded):
        """Compute class weights for imbalanced dataset"""
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_encoded), 
            y=y_encoded
        )
        
        self.class_weights = dict(zip(np.unique(y_encoded), class_weights))
        logger.info(f"Class weights computed for {len(self.class_weights)} classes")
        
        return self.class_weights
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
        """Save all processed data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_path}")
        
        # Save arrays
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "X_val.npy", X_val)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "y_val.npy", y_val)
        np.save(output_path / "y_test.npy", y_test)
        
        logger.info("✅ Processed data saved")
        
        return {
            'X_train': output_path / "X_train.npy",
            'X_val': output_path / "X_val.npy", 
            'X_test': output_path / "X_test.npy",
            'y_train': output_path / "y_train.npy",
            'y_val': output_path / "y_val.npy",
            'y_test': output_path / "y_test.npy"
        }
    
    def save_preprocessing_artifacts(self, artifacts_dir, metadata):
        """Save preprocessing artifacts"""
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving preprocessing artifacts to {artifacts_path}")
        
        # Save scaler and encoder
        with open(artifacts_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(artifacts_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save class weights
        if self.class_weights:
            with open(artifacts_path / 'class_weights.json', 'w') as f:
                json.dump({str(k): float(v) for k, v in self.class_weights.items()}, f, indent=2)
        
        # Save labels mapping
        labels_mapping = {
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': len(self.label_encoder.classes_)
        }
        with open(artifacts_path / 'labels.json', 'w') as f:
            json.dump(labels_mapping, f, indent=2)
        
        logger.info("✅ Preprocessing artifacts saved")
        
        return {
            'scaler': artifacts_path / 'scaler.pkl',
            'label_encoder': artifacts_path / 'label_encoder.pkl',
            'class_weights': artifacts_path / 'class_weights.json',
            'labels': artifacts_path / 'labels.json'
        }


def main():
    parser = argparse.ArgumentParser(description="Enhanced preprocessing for sign language recognition")
    parser.add_argument('--csv', type=str, default='data/data.csv', help='Input CSV manifest file')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory for processed data')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', help='Output directory for preprocessing artifacts')
    parser.add_argument('--max_sequence_length', type=int, default=30, help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set proportion')
    parser.add_argument('--min_samples_per_class', type=int, default=5, help='Minimum samples per class')
    parser.add_argument('--augment_factor', type=int, default=3, help='Data augmentation factor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load the manifest
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"Manifest file not found: {csv_path}")
        logger.error("Please run the landmark extraction script first.")
        return

    manifest_df = pd.read_csv(csv_path)
    landmark_files = manifest_df.to_dict('records')
    logger.info(f"Loaded {len(landmark_files)} entries from {args.csv}")
    
    preprocessor = EnhancedSignLanguagePreprocessor(random_state=args.seed)
    
    # Load sequences
    X, y, metadata = preprocessor.load_landmark_sequences(
        landmark_files, 
        args.max_sequence_length
    )
    
    # Filter classes with insufficient samples
    X, y = preprocessor.filter_classes_by_sample_count(X, y, args.min_samples_per_class)
    
    # Apply data augmentation to increase dataset size
    if args.augment_factor > 1:
        X, y = preprocessor.apply_data_augmentation(X, y, args.augment_factor)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, 
        test_size=args.test_size, 
        val_size=args.val_size
    )
    
    # Encode labels
    y_train_enc, y_val_enc, y_test_enc = preprocessor.encode_labels(y_train, y_val, y_test)
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm = preprocessor.normalize_features(
        X_train, X_val, X_test
    )
    
    # Compute class weights
    class_weights = preprocessor.compute_class_weights(y_train_enc)
    
    # Save processed data
    data_paths = preprocessor.save_processed_data(
        X_train_norm, X_val_norm, X_test_norm, 
        y_train_enc, y_val_enc, y_test_enc,
        args.output_dir
    )
    
    # Save preprocessing artifacts
    artifact_paths = preprocessor.save_preprocessing_artifacts(
        args.artifacts_dir, metadata
    )
    
    logger.info("🎉 Enhanced preprocessing completed successfully!")
    
    # Print comprehensive summary
    print(f"\n" + "="*60)
    print("📊 PREPROCESSING SUMMARY")
    print("="*60)
    print(f"📁 Data directory: {args.output_dir}")
    print(f"📁 Artifacts directory: {args.artifacts_dir}")
    print(f"🎯 Total sequences (after filtering & augmentation): {len(X)}")
    print(f"⏱️  Sequence length: {args.max_sequence_length} frames")
    print(f"🔢 Feature dimension: {X.shape[-1]} (126 = 2 hands × 21 landmarks × 3 coords)")
    print(f"🏷️  Total classes: {len(preprocessor.label_encoder.classes_)}")
    print(f"📊 Train/Val/Test splits: {len(X_train)}/{len(X_val)}/{len(X_test)} samples")
    print(f"🔄 Augmentation factor: {args.augment_factor}x")
    print("\n🚀 Next step: Run the training script:")
    print("   python train_model.py --data_dir data/processed --artifacts_dir artifacts")
    print("="*60)


if __name__ == '__main__':
    main()