"""
Sign Language Model Trainer - Fixed version
Handles single sample per class scenario with data augmentation
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import argparse
from collections import Counter

from simplified_config import config, logger

class DataAugmentor:
    """Enhanced data augmentation for single samples per class"""
    
    def __init__(self):
        logger.info("Data augmentation initialized")
        self.noise_factor = 0.02
        self.rotation_range = 10  # degrees
        self.scale_range = 0.1
        self.temporal_shift_range = 0.1
    
    def add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to sequence"""
        noise = np.random.normal(0, self.noise_factor, sequence.shape)
        return sequence + noise
    
    def temporal_shift(self, sequence: np.ndarray) -> np.ndarray:
        """Shift sequence temporally"""
        seq_len = len(sequence)
        shift = int(seq_len * self.temporal_shift_range * np.random.uniform(-1, 1))
        
        if shift > 0:
            # Shift right, pad left
            padded = np.pad(sequence, ((shift, 0), (0, 0)), mode='edge')
            return padded[:seq_len]
        elif shift < 0:
            # Shift left, pad right
            padded = np.pad(sequence, ((0, -shift), (0, 0)), mode='edge')
            return padded[-seq_len:]
        return sequence
    
    def scale_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Scale the entire sequence"""
        scale_factor = 1 + np.random.uniform(-self.scale_range, self.scale_range)
        return sequence * scale_factor
    
    def augment_sequence(self, sequence: np.ndarray, num_augmentations: int = 4) -> np.ndarray:
        """Generate multiple augmented versions of a sequence"""
        augmented = [sequence]  # Original
        
        for _ in range(num_augmentations):
            aug_seq = sequence.copy()
            
            # Apply random combination of augmentations
            if np.random.random() > 0.3:
                aug_seq = self.add_noise(aug_seq)
            if np.random.random() > 0.3:
                aug_seq = self.temporal_shift(aug_seq)
            if np.random.random() > 0.3:
                aug_seq = self.scale_sequence(aug_seq)
            
            augmented.append(aug_seq)
        
        return np.array(augmented)
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, target_samples_per_class: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Augment entire dataset to have target samples per class"""
        logger.info(f"Augmenting dataset to {target_samples_per_class} samples per class...")
        
        augmented_X = []
        augmented_y = []
        
        unique_classes = np.unique(y)
        
        for class_label in unique_classes:
            # Get all samples for this class
            class_mask = y == class_label
            class_samples = X[class_mask]
            
            current_count = len(class_samples)
            logger.info(f"Class '{class_label}': {current_count} -> {target_samples_per_class} samples")
            
            if current_count >= target_samples_per_class:
                # Already have enough samples, just take what we need
                selected_samples = class_samples[:target_samples_per_class]
                augmented_X.extend(selected_samples)
                augmented_y.extend([class_label] * target_samples_per_class)
            else:
                # Need to generate more samples
                needed_samples = target_samples_per_class - current_count
                augmentations_per_sample = max(1, needed_samples // current_count)
                
                # Add original samples
                augmented_X.extend(class_samples)
                augmented_y.extend([class_label] * current_count)
                
                # Generate augmented samples
                samples_generated = 0
                for sample in class_samples:
                    if samples_generated >= needed_samples:
                        break
                    
                    # Generate augmentations for this sample
                    aug_samples = self.augment_sequence(sample, num_augmentations=augmentations_per_sample)
                    
                    # Skip the original (index 0) and add augmented ones
                    for aug_sample in aug_samples[1:]:
                        if samples_generated >= needed_samples:
                            break
                        augmented_X.append(aug_sample)
                        augmented_y.append(class_label)
                        samples_generated += 1
        
        final_X = np.array(augmented_X, dtype=np.float32)
        final_y = np.array(augmented_y)
        
        logger.info(f"Dataset augmented: {X.shape} -> {final_X.shape}")
        return final_X, final_y

class SignLanguageTrainer:
    """Enhanced trainer for sign language recognition with single samples"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        self.augmentor = DataAugmentor()
        
        # Set memory growth for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
    
    def load_or_process_data(self, force_reprocess: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Load or process training data"""
        if config.TRAINING_DATA_PATH.exists() and not force_reprocess:
            logger.info("Loading existing training data...")
            
            data = np.load(config.TRAINING_DATA_PATH, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            logger.info(f"Loaded: {X.shape}, {len(y)} labels")
            return X, y
        else:
            logger.info("Processing videos to create training data...")
            from video_processor import VideoProcessor
            
            processor = VideoProcessor()
            result = processor.process_all_videos()
            
            if not result:
                raise ValueError("No training data could be generated!")
            
            return result['X'], result['y']
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, augment: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with proper augmentation"""
        logger.info("Preparing training data...")
        
        # Check class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        logger.info(f"Original dataset: {len(unique_classes)} classes, {len(X)} total samples")
        
        min_samples = np.min(counts)
        max_samples = np.max(counts)
        logger.info(f"Samples per class - Min: {min_samples}, Max: {max_samples}")
        
        if augment and min_samples < 5:  # Need augmentation
            logger.info("Augmenting dataset due to insufficient samples per class...")
            target_samples = max(10, max_samples * 2)  # At least 10 samples per class
            X, y = self.augmentor.augment_dataset(X, y, target_samples_per_class=target_samples)
            
            # Check new distribution
            unique_classes, counts = np.unique(y, return_counts=True)
            logger.info(f"After augmentation: {len(unique_classes)} classes, {len(X)} total samples")
            logger.info(f"New samples per class - Min: {np.min(counts)}, Max: {np.max(counts)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data - no stratification if still too few samples
        min_samples_per_class = np.min(np.bincount(y_encoded))
        
        if min_samples_per_class >= 2:
            # Can use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.2, 
                random_state=42,
                stratify=y_encoded
            )
            logger.info("Used stratified split")
        else:
            # Simple random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.2, 
                random_state=42
            )
            logger.info("Used random split (not enough samples for stratification)")
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_model(self, input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
        """Create improved LSTM model for sign language recognition"""
        logger.info(f"Creating model for {num_classes} classes...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Masking for variable length sequences
            layers.Masking(mask_value=0.0),
            
            # LSTM layers with dropout
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            layers.LSTM(96, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with appropriate settings for multi-class
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        return model
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train the model with proper callbacks"""
        
        num_classes = len(np.unique(y_train))
        input_shape = X_train.shape[1:]
        
        # Create model
        self.model = self.create_model(input_shape, num_classes)
        
        # Calculate class weights to handle imbalance
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        logger.info("Calculated class weights for balanced training")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.MODEL_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting model training...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,  # Increased for better learning
            batch_size=min(32, len(X_train) // 4),  # Adaptive batch size
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final model
        test_loss, test_accuracy, test_top_k = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for detailed analysis
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Training summary
        best_val_acc = max(self.history.history['val_accuracy'])
        
        results = {
            'test_accuracy': float(test_accuracy),
            'test_top_k_accuracy': float(test_top_k),
            'best_val_accuracy': float(best_val_acc),
            'total_epochs': len(self.history.history['loss']),
            'num_classes': int(num_classes),
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test accuracy: {test_accuracy:.3f}")
        logger.info(f"Best validation accuracy: {best_val_acc:.3f}")
        
        return results
    
    def save_model_artifacts(self, results: Dict[str, Any]) -> None:
        """Save model and associated artifacts"""
        
        # Save label encoder
        encoder_path = config.DATA_DIR / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save training results
        results_path = config.DATA_DIR / "training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save class mapping
        class_mapping = {
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': len(self.label_encoder.classes_)
        }
        
        mapping_path = config.DATA_DIR / "class_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info("Model artifacts saved:")
        logger.info(f"  Model: {config.MODEL_PATH}")
        logger.info(f"  Label encoder: {encoder_path}")
        logger.info(f"  Results: {results_path}")
        logger.info(f"  Class mapping: {mapping_path}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of video data')
    parser.add_argument('--no-augment', action='store_true',
                       help='Skip data augmentation')
    args = parser.parse_args()
    
    try:
        logger.info("Sign Language Model Training Pipeline")
        logger.info("=" * 50)
        
        # Initialize trainer
        trainer = SignLanguageTrainer()
        
        # Load data
        logger.info("Step 1: Loading training data...")
        X, y = trainer.load_or_process_data(force_reprocess=args.force_reprocess)
        
        if len(X) == 0:
            logger.error("No training data available!")
            return 1
        
        # Data overview
        unique_words = np.unique(y)
        word_counts = Counter(y)
        
        logger.info("Dataset Overview:")
        logger.info(f"   Total samples: {len(X)}")
        logger.info(f"   Unique words: {len(unique_words)}")
        logger.info(f"   Sequence length: {X.shape[1]}")
        logger.info(f"   Features per frame: {X.shape[2]}")
        
        logger.info("Word distribution:")
        for word, count in sorted(word_counts.items()):
            logger.info(f"   {word}: {count} sample(s)")
        
        # Prepare data
        logger.info("Step 2: Preparing training data...")
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            X, y, augment=not args.no_augment
        )
        
        # Train model
        logger.info("Step 3: Training model...")
        results = trainer.train_model(X_train, X_test, y_train, y_test)
        
        # Save everything
        logger.info("Step 4: Saving model and artifacts...")
        trainer.save_model_artifacts(results)
        
        # Final summary
        logger.info("Training Summary:")
        logger.info("=" * 50)
        logger.info(f"Final test accuracy: {results['test_accuracy']:.1%}")
        logger.info(f"Top-k accuracy: {results['test_top_k_accuracy']:.1%}")
        logger.info(f"Total classes: {results['num_classes']}")
        logger.info(f"Training samples: {results['training_samples']}")
        logger.info(f"Test samples: {results['test_samples']}")
        logger.info(f"Training epochs: {results['total_epochs']}")
        
        if results['test_accuracy'] > 0.8:
            logger.info("Great! Model achieved good accuracy")
        elif results['test_accuracy'] > 0.6:
            logger.info("Decent accuracy. Consider more data or training")
        else:
            logger.info("Low accuracy. More data augmentation needed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Check your data and configuration")
        return 1

if __name__ == "__main__":
    exit(main())