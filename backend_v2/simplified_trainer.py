"""
FIXED Sign Language Model Trainer
Enhanced training pipeline similar to friend's approach with video data handling
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from simplified_config import config, logger
from simplified_model import FixedSignLanguageModel

class DataAugmentor:
    """FIXED: Enhanced data augmentation for video-based sign language data"""
    
    def __init__(self):
        logger.info("🔄 Data augmentation initialized")
        # FIXED: More conservative augmentation parameters for stability
        self.noise_factor = 0.01        # Reduced noise
        self.rotation_range = 5         # Reduced rotation  
        self.scale_range = 0.05         # Reduced scaling
        self.temporal_shift_range = 0.1 # Keep temporal shift
    
    def add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """FIXED: Add controlled Gaussian noise"""
        noise = np.random.normal(0, self.noise_factor, sequence.shape)
        return sequence + noise
    
    def temporal_shift(self, sequence: np.ndarray) -> np.ndarray:
        """FIXED: Shift sequence temporally (similar to friend's approach)"""
        seq_len = len(sequence)
        shift = int(seq_len * self.temporal_shift_range * np.random.uniform(-1, 1))
        
        if shift > 0:
            # Shift right, pad left with first frame
            padded = np.pad(sequence, ((shift, 0), (0, 0)), mode='edge')
            return padded[:seq_len]
        elif shift < 0:
            # Shift left, pad right with last frame
            padded = np.pad(sequence, ((0, -shift), (0, 0)), mode='edge')
            return padded[-seq_len:]
        return sequence
    
    def scale_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """FIXED: Scale the sequence (hand size variation)"""
        scale_factor = 1 + np.random.uniform(-self.scale_range, self.scale_range)
        return sequence * scale_factor
    
    def augment_sequence(self, sequence: np.ndarray, num_augmentations: int = 3) -> np.ndarray:
        """FIXED: Generate augmented versions (conservative approach)"""
        augmented = [sequence]  # Original
        
        for _ in range(num_augmentations):
            aug_seq = sequence.copy()
            
            # FIXED: Apply random combination of augmentations (more conservative)
            if np.random.random() > 0.5:  # 50% chance
                aug_seq = self.add_noise(aug_seq)
            if np.random.random() > 0.5:  # 50% chance
                aug_seq = self.temporal_shift(aug_seq)
            if np.random.random() > 0.7:  # 30% chance (less scaling)
                aug_seq = self.scale_sequence(aug_seq)
            
            augmented.append(aug_seq)
        
        return np.array(augmented)
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, target_samples_per_class: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Conservative dataset augmentation (similar to friend's 30 sequences per class)"""
        logger.info(f"🔄 Augmenting dataset to {target_samples_per_class} samples per class...")
        
        augmented_X = []
        augmented_y = []
        
        unique_classes = np.unique(y)
        
        for class_label in unique_classes:
            # Get all samples for this class
            class_mask = y == class_label
            class_samples = X[class_mask]
            
            current_count = len(class_samples)
            logger.info(f"   {class_label}: {current_count} -> {target_samples_per_class} samples")
            
            if current_count >= target_samples_per_class:
                # Already have enough samples, just take what we need
                selected_samples = class_samples[:target_samples_per_class]
                augmented_X.extend(selected_samples)
                augmented_y.extend([class_label] * target_samples_per_class)
            else:
                # Add original samples
                augmented_X.extend(class_samples)
                augmented_y.extend([class_label] * current_count)
                
                # Generate additional samples through augmentation
                needed_samples = target_samples_per_class - current_count
                samples_generated = 0
                
                while samples_generated < needed_samples:
                    for sample in class_samples:
                        if samples_generated >= needed_samples:
                            break
                        
                        # Generate one augmentation
                        aug_samples = self.augment_sequence(sample, num_augmentations=1)
                        
                        # Add the augmented sample (skip original at index 0)
                        if len(aug_samples) > 1:
                            augmented_X.append(aug_samples[1])
                            augmented_y.append(class_label)
                            samples_generated += 1
        
        final_X = np.array(augmented_X, dtype=np.float32)
        final_y = np.array(augmented_y)
        
        logger.info(f"✅ Dataset augmented: {X.shape} -> {final_X.shape}")
        return final_X, final_y

class SignLanguageTrainer:
    """FIXED: Enhanced trainer similar to friend's approach"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        self.augmentor = DataAugmentor()
        
        # FIXED: Set memory growth for GPU (if available)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"⚠️ GPU configuration error: {e}")
    
    def load_or_process_data(self, force_reprocess: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Load or process training data"""
        if config.TRAINING_DATA_PATH.exists() and not force_reprocess:
            logger.info("📂 Loading existing training data...")
            
            data = np.load(config.TRAINING_DATA_PATH, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            logger.info(f"✅ Loaded: {X.shape}, {len(y)} labels")
            return X, y
        else:
            logger.info("🎬 Processing videos to create training data...")
            from video_processor import VideoProcessor
            
            processor = VideoProcessor()
            result = processor.process_all_videos()
            
            if not result:
                raise ValueError("❌ No training data could be generated!")
            
            return result['X'], result['y']
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, augment: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """FIXED: Prepare data similar to friend's approach"""
        logger.info("🔧 Preparing training data...")
        
        # Check class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        logger.info(f"📊 Original dataset: {len(unique_classes)} classes, {len(X)} total samples")
        
        min_samples = np.min(counts)
        max_samples = np.max(counts)
        logger.info(f"📊 Samples per class - Min: {min_samples}, Max: {max_samples}")
        
        # FIXED: Apply augmentation if needed (similar to friend's 30 sequences approach)
        if augment and min_samples < 5:
            logger.info("🔄 Augmenting dataset due to insufficient samples per class...")
            target_samples = max(8, max_samples)  # Conservative target
            X, y = self.augmentor.augment_dataset(X, y, target_samples_per_class=target_samples)
            
            # Check new distribution
            unique_classes, counts = np.unique(y, return_counts=True)
            logger.info(f"📊 After augmentation: {len(unique_classes)} classes, {len(X)} total samples")
            logger.info(f"📊 New samples per class - Min: {np.min(counts)}, Max: {np.max(counts)}")
        
        # FIXED: Encode labels (similar to friend's label_map approach)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # FIXED: Convert to categorical (like friend's to_categorical)
        y_categorical = to_categorical(y_encoded)
        
        # FIXED: Split data with proper stratification
        min_samples_per_class = np.min(np.bincount(y_encoded))
        
        if min_samples_per_class >= 2:
            # Can use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_categorical, 
                test_size=0.15,  # Smaller test set like friend's 10%
                random_state=42,
                stratify=y_encoded
            )
            logger.info("✅ Used stratified split")
        else:
            # Simple random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_categorical, 
                test_size=0.15, 
                random_state=42
            )
            logger.info("✅ Used random split")
        
        logger.info(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """FIXED: Train the model with friend's approach"""
        
        num_classes = len(np.unique(self.label_encoder.classes_))
        
        # FIXED: Create model with proper architecture
        self.model = FixedSignLanguageModel(num_classes=num_classes)
        
        logger.info(f"🚀 Training model with {num_classes} classes...")
        
        # FIXED: Train model (similar to friend's 100 epochs)
        results = self.model.train_model(
            X_train, y_train,
            X_test, y_test,
            epochs=100  # Same as friend's model
        )
        
        # FIXED: Evaluate predictions (like friend's accuracy calculation)
        y_pred = self.model.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        
        # FIXED: Get class names for reporting
        class_names = self.label_encoder.classes_
        
        # Generate classification report
        report = classification_report(
            y_test_classes, y_pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # FIXED: Enhanced results dictionary
        results.update({
            'test_accuracy': float(accuracy),
            'num_classes': int(num_classes),
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'class_names': list(class_names),
            'classification_report': report
        })
        
        logger.info("🎊 Training completed successfully!")
        logger.info(f"   Final test accuracy: {accuracy:.3f}")
        logger.info(f"   Classes: {list(class_names)}")
        
        return results
    
    def save_model_artifacts(self, results: Dict[str, Any]) -> None:
        """FIXED: Save model and associated artifacts"""
        
        # Save the trained model
        if not self.model.save_model():
            logger.error("❌ Failed to save model")
            return
        
        # Save label encoder
        if not self.model.save_label_encoder():
            logger.error("❌ Failed to save label encoder")
            return
        
        # Save training results
        results_path = config.DATA_DIR / "training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save class mapping
        class_mapping = {
            'classes': list(self.label_encoder.classes_),
            'num_classes': len(self.label_encoder.classes_),
            'class_to_index': {cls: int(idx) for idx, cls in enumerate(self.label_encoder.classes_)},
            'index_to_class': {int(idx): cls for idx, cls in enumerate(self.label_encoder.classes_)}
        }
        
        mapping_path = config.DATA_DIR / "class_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Model artifacts saved:")
        logger.info(f"   Model: {config.MODEL_PATH}")
        logger.info(f"   Label encoder: {config.DATA_DIR / 'label_encoder.pkl'}")
        logger.info(f"   Results: {results_path}")
        logger.info(f"   Class mapping: {mapping_path}")
    
    def plot_training_history(self):
        """FIXED: Plot training history"""
        if self.model and self.model.history:
            history = self.model.history.history
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            ax1.plot(history['categorical_accuracy'], label='Training Accuracy')
            ax1.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot loss
            ax2.plot(history['loss'], label='Training Loss')
            ax2.plot(history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(config.DATA_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Training history plot saved: {config.DATA_DIR / 'training_history.png'}")


def main():
    """FIXED: Main training pipeline similar to friend's approach"""
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of video data')
    parser.add_argument('--no-augment', action='store_true',
                       help='Skip data augmentation')
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Sign Language Model Training Pipeline")
        logger.info("=" * 50)
        
        # Initialize trainer
        trainer = SignLanguageTrainer()
        
        # FIXED: Load data (similar to friend's data loading)
        logger.info("📂 Step 1: Loading training data...")
        X, y = trainer.load_or_process_data(force_reprocess=args.force_reprocess)
        
        if len(X) == 0:
            logger.error("❌ No training data available!")
            return 1
        
        # Data overview
        unique_words = np.unique(y)
        word_counts = Counter(y)
        
        logger.info("📊 Dataset Overview:")
        logger.info(f"   Total samples: {len(X)}")
        logger.info(f"   Unique words: {len(unique_words)}")
        logger.info(f"   Sequence length: {X.shape[1]}")
        logger.info(f"   Features per frame: {X.shape[2]}")
        
        logger.info("📈 Word distribution:")
        for word, count in sorted(word_counts.items()):
            logger.info(f"   {word}: {count} sample(s)")
        
        # FIXED: Prepare data (similar to friend's train_test_split approach)
        logger.info("🔧 Step 2: Preparing training data...")
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            X, y, augment=not args.no_augment
        )
        
        # FIXED: Train model (similar to friend's model.fit approach)
        logger.info("🤖 Step 3: Training model...")
        results = trainer.train_model(X_train, X_test, y_train, y_test)
        
        # Save everything
        logger.info("💾 Step 4: Saving model and artifacts...")
        trainer.save_model_artifacts(results)
        
        # Plot training history
        trainer.plot_training_history()
        
        # FIXED: Final summary (like friend's accuracy output)
        logger.info("🎊 Training Summary:")
        logger.info("=" * 50)
        logger.info(f"✅ Final test accuracy: {results['test_accuracy']:.1%}")
        logger.info(f"📊 Total classes: {results['num_classes']}")
        logger.info(f"📚 Training samples: {results['training_samples']}")
        logger.info(f"🧪 Test samples: {results['test_samples']}")
        logger.info(f"⏱️ Training epochs: {results.get('epochs_trained', 'N/A')}")
        logger.info(f"🎯 Classes: {', '.join(results['class_names'])}")
        
        if results['test_accuracy'] > 0.85:
            logger.info("🎉 Excellent! Model achieved high accuracy")
        elif results['test_accuracy'] > 0.70:
            logger.info("👍 Good accuracy. Model should work well")
        elif results['test_accuracy'] > 0.50:
            logger.info("⚠️ Moderate accuracy. Consider more data or tuning")
        else:
            logger.info("❌ Low accuracy. Need more data or different approach")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error("💡 Check your data and configuration")
        return 1

if __name__ == "__main__":
    exit(main())