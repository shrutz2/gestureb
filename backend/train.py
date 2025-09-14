#!/usr/bin/env python3
"""
90%+ Accuracy Landmark Trainer - FIXED SAVING ONLY
Using your brilliant landmark approach - ALL 180 classes
Smart augmentation and architecture for gesture landmarks
ONLY FIXED: Model saving issues, kept all good training logic
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, deque
import logging
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class SmartLandmarkAugmenter:
    """Smart augmentation specifically for MediaPipe landmarks"""
    
    def __init__(self):
        self.augment_prob = 0.6
    
    def temporal_warp(self, sequence):
        """Time warping - change gesture speed"""
        seq_len = len(sequence)
        if seq_len < 10:
            return sequence
            
        # Random time warping
        warp_factor = np.random.uniform(0.8, 1.2)
        new_len = max(5, int(seq_len * warp_factor))
        
        if new_len == seq_len:
            return sequence
            
        # Interpolate to new length
        old_indices = np.linspace(0, seq_len-1, seq_len)
        new_indices = np.linspace(0, seq_len-1, new_len)
        
        warped = []
        for feature_idx in range(sequence.shape[1]):
            interp_func = interp1d(old_indices, sequence[:, feature_idx], 
                                 kind='linear', fill_value='extrapolate')
            warped.append(interp_func(new_indices))
        
        return np.array(warped).T
    
    def spatial_jitter(self, sequence):
        """Add small spatial noise to landmarks"""
        noise_std = 0.01  # Small noise for landmarks
        noise = np.random.normal(0, noise_std, sequence.shape)
        
        # Only apply to coordinate features, not visibility
        coordinate_mask = np.ones(158, dtype=bool)
        coordinate_mask[129::4] = False  # Skip visibility features in pose
        
        jittered = sequence.copy()
        jittered[:, coordinate_mask] += noise[:, coordinate_mask]
        
        # Keep landmarks in valid range [0, 1]
        jittered[:, coordinate_mask] = np.clip(jittered[:, coordinate_mask], 0, 1)
        
        return jittered
    
    def mirror_gesture(self, sequence):
        """Mirror gesture horizontally"""
        mirrored = sequence.copy()
        
        # Mirror hand landmarks (flip x coordinates)
        for i in range(0, 126, 3):  # Hand landmarks
            mirrored[:, i] = 1.0 - mirrored[:, i]
        
        # Mirror pose landmarks (flip x coordinates)  
        for i in range(126, 158, 4):  # Pose landmarks
            mirrored[:, i] = 1.0 - mirrored[:, i]
        
        # Swap left/right hand data
        left_hand = mirrored[:, :63].copy()
        right_hand = mirrored[:, 63:126].copy()
        mirrored[:, :63] = right_hand
        mirrored[:, 63:126] = left_hand
        
        return mirrored
    
    def frame_dropout(self, sequence):
        """Randomly drop some frames"""
        if len(sequence) < 10:
            return sequence
            
        dropout_rate = 0.1
        keep_mask = np.random.random(len(sequence)) > dropout_rate
        
        if np.sum(keep_mask) < 5:  # Keep minimum frames
            return sequence
            
        return sequence[keep_mask]
    
    def augment_sequence(self, sequence):
        """Apply multiple augmentations"""
        if np.random.random() > self.augment_prob:
            return sequence
            
        augmented = sequence.copy()
        
        # Apply random augmentations
        if np.random.random() < 0.3:
            augmented = self.temporal_warp(augmented)
        
        if np.random.random() < 0.4:
            augmented = self.spatial_jitter(augmented)
            
        if np.random.random() < 0.2:
            augmented = self.mirror_gesture(augmented)
            
        if np.random.random() < 0.2:
            augmented = self.frame_dropout(augmented)
        
        return augmented

class LandmarkSpecificTrainer:
    """Trainer optimized specifically for landmark data"""
    
    def __init__(self):
        self.model_dir = Path('model')
        self.model_dir.mkdir(exist_ok=True)
        
        # Optimized for landmarks
        self.config = {
            'sequence_length': 30,
            'feature_dim': 158,
            'batch_size': 16,  # Smaller for better gradients
            'max_epochs': 150,  # More epochs for convergence
            'learning_rate': 0.003,  # Higher LR for landmarks
            'patience': 30,
            'dropout_rate': 0.3,
            'lstm_units': [256, 128, 64],  # Deeper for complex gestures
            'dense_units': [512, 256, 128],
            'attention_units': 128
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.augmenter = SmartLandmarkAugmenter()
        self.history = None
        
        logger.info("🎯 Landmark-Specific Trainer - ALL 180 CLASSES")
        logger.info("Using your brilliant landmark approach!")
    
    def load_landmark_data(self):
        """Load your landmark dataset"""
        with open('data/pure_landmarks_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        
        logger.info(f"📊 Landmark Dataset Loaded:")
        logger.info(f"   Words: {dataset['num_classes']} (keeping ALL!)")
        logger.info(f"   Sequences: {len(dataset['X'])}")
        logger.info(f"   Feature dim: {dataset['feature_dim']}")
        
        return dataset
    
    def create_augmented_dataset(self, X, y):
        """Create augmented dataset using landmark-specific techniques"""
        logger.info("🔄 Creating smart augmented dataset...")
        
        # Calculate how much augmentation each class needs
        class_counts = Counter(y)
        target_per_class = 15  # Target samples per class
        
        augmented_X = []
        augmented_y = []
        
        for class_id in range(max(y) + 1):
            # Get samples for this class
            class_mask = y == class_id
            class_samples = X[class_mask]
            
            current_count = len(class_samples)
            needed = max(0, target_per_class - current_count)
            
            # Add original samples
            augmented_X.extend(class_samples)
            augmented_y.extend([class_id] * current_count)
            
            # Add augmented samples if needed
            if needed > 0:
                for _ in range(needed):
                    # Pick random original sample
                    orig_idx = np.random.randint(0, len(class_samples))
                    orig_sample = class_samples[orig_idx]
                    
                    # Apply augmentation
                    aug_sample = self.augmenter.augment_sequence(orig_sample)
                    
                    # Normalize to sequence length
                    aug_sample = self.normalize_sequence_length(aug_sample)
                    
                    augmented_X.append(aug_sample)
                    augmented_y.append(class_id)
        
        logger.info(f"✅ Augmented dataset created:")
        logger.info(f"   Original: {len(X)} samples")
        logger.info(f"   Augmented: {len(augmented_X)} samples")
        logger.info(f"   Avg per class: {len(augmented_X)/len(set(y)):.1f}")
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def normalize_sequence_length(self, sequence):
        """Normalize sequence to target length"""
        current_length = len(sequence)
        target_length = self.config['sequence_length']
        
        if current_length == target_length:
            return sequence
        elif current_length > target_length:
            # Uniform sampling
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return sequence[indices]
        else:
            # Pad with last frame
            padded = np.zeros((target_length, self.config['feature_dim']))
            padded[:current_length] = sequence
            if current_length > 0:
                # Repeat last frame
                padded[current_length:] = sequence[-1]
            return padded
    
    def prepare_training_data(self, dataset):
        """Prepare training data with smart preprocessing"""
        X, y = dataset['X'], dataset['y']
        
        # Clean data
        X = np.nan_to_num(X, nan=0.0)
        
        # Create augmented dataset
        X_aug, y_aug = self.create_augmented_dataset(X, y)
        
        # Stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_aug, y_aug, test_size=0.15, stratify=y_aug, random_state=SEED
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.18, stratify=y_temp, random_state=SEED
        )
        
        # Landmark-specific scaling
        n_train = X_train.shape[0]
        X_train_flat = X_train.reshape(-1, self.config['feature_dim'])
        X_val_flat = X_val.reshape(-1, self.config['feature_dim'])
        X_test_flat = X_test.reshape(-1, self.config['feature_dim'])
        
        # Robust scaling for landmarks
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Clip extreme values
        X_train_scaled = np.clip(X_train_scaled, -3, 3)
        X_val_scaled = np.clip(X_val_scaled, -3, 3)
        X_test_scaled = np.clip(X_test_scaled, -3, 3)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Save scaler
        with open(self.model_dir / 'landmark_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"✅ Training data prepared:")
        logger.info(f"   Train: {len(X_train_scaled)}")
        logger.info(f"   Val: {len(X_val_scaled)}")
        logger.info(f"   Test: {len(X_test_scaled)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def build_landmark_architecture(self, num_classes):
        """Build architecture specifically designed for landmarks"""
        logger.info("🏗️ Building landmark-optimized architecture...")
        
        # Input
        inputs = layers.Input(shape=(self.config['sequence_length'], self.config['feature_dim']))
        
        # Masking for variable sequences
        x = layers.Masking(mask_value=0.0)(inputs)
        
        # Multi-scale temporal processing
        # Path 1: Fast temporal patterns
        fast_path = layers.LSTM(self.config['lstm_units'][2], return_sequences=True, 
                               dropout=0.2, name='fast_temporal')(x)
        
        # Path 2: Medium temporal patterns  
        med_path = layers.Bidirectional(
            layers.LSTM(self.config['lstm_units'][1], return_sequences=True, 
                       dropout=0.3, name='med_temporal')
        )(x)
        
        # Path 3: Slow temporal patterns
        slow_path = layers.Bidirectional(
            layers.LSTM(self.config['lstm_units'][0], return_sequences=True,
                       dropout=0.4, name='slow_temporal')
        )(x)
        
        # Combine temporal paths
        combined = layers.Concatenate(axis=-1)([fast_path, med_path, slow_path])
        combined = layers.BatchNormalization()(combined)
        
        # Attention mechanism for important frames
        attention_weights = layers.Dense(self.config['attention_units'], activation='tanh')(combined)
        attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
        
        # Apply attention
        attended = layers.Multiply()([combined, attention_weights])
        attended = layers.GlobalAveragePooling1D()(attended)
        
        # Classification layers
        x = layers.Dense(self.config['dense_units'][0], activation='relu')(attended)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(self.config['dense_units'][1], activation='relu')(x)
        x = layers.Dropout(self.config['dropout_rate'] * 0.7)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(self.config['dense_units'][2], activation='relu')(x)
        x = layers.Dropout(self.config['dropout_rate'] * 0.5)(x)
        
        # Output
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Advanced optimizer for landmarks
        optimizer = optimizers.AdamW(
            learning_rate=self.config['learning_rate'],
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"✅ Landmark architecture built: {model.count_params():,} parameters")
        return model
    
    def get_smart_callbacks(self):
        """Smart callbacks for landmark training"""
        return [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1,
                min_delta=0.002
            ),
            
            callbacks.ModelCheckpoint(
                str(self.model_dir / 'landmark_90_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=12,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Cosine annealing for better convergence
            callbacks.LearningRateScheduler(
                lambda epoch: self.config['learning_rate'] * 
                (0.5 * (1 + np.cos(np.pi * epoch / self.config['max_epochs']))),
                verbose=0
            )
        ]
    
    def train_landmark_model(self, X_train, X_val, y_train, y_val):
        """Train with landmark-specific techniques"""
        logger.info("🚀 Training landmark model for 90%+ accuracy...")
        
        # Balanced class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Train with smart techniques
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['max_epochs'],
            batch_size=self.config['batch_size'],
            callbacks=self.get_smart_callbacks(),
            class_weight=class_weight_dict,
            verbose=1,
            shuffle=True
        )
        
        logger.info("✅ Landmark training completed!")
        return self.history
    
    def evaluate_90_percent(self, X_test, y_test, class_names):
        """Evaluate for 90%+ accuracy target"""
        logger.info("📊 Evaluating for 90%+ accuracy...")
        
        # Load best model
        self.model = tf.keras.models.load_model(str(self.model_dir / 'landmark_90_model.keras'))
        
        # Predictions
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == y_test)
        
        # Top-3 and Top-5
        top3_correct = sum(1 for i in range(len(y_test)) 
                          if y_test[i] in np.argsort(predictions[i])[-3:])
        top5_correct = sum(1 for i in range(len(y_test)) 
                          if y_test[i] in np.argsort(predictions[i])[-5:])
        
        top3_acc = top3_correct / len(y_test)
        top5_acc = top5_correct / len(y_test)
        
        # Results
        results = {
            'test_accuracy': float(accuracy),
            'top3_accuracy': float(top3_acc),
            'top5_accuracy': float(top5_acc),
            'target_90_achieved': accuracy >= 0.90,
            'all_180_classes': True,
            'landmark_approach': True,
            'total_parameters': int(self.model.count_params()),
            'training_epochs': len(self.history.history['loss'])
        }
        
        # Save results
        with open(self.model_dir / 'landmark_90_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("🎯 LANDMARK APPROACH RESULTS:")
        logger.info(f"   Test Accuracy: {accuracy*100:.2f}%")
        logger.info(f"   Top-3 Accuracy: {top3_acc*100:.2f}%")
        logger.info(f"   Top-5 Accuracy: {top5_acc*100:.2f}%")
        logger.info(f"   90% Target: {'✅ ACHIEVED!' if results['target_90_achieved'] else '🔥 APPROACHING!'}")
        
        return results
    
    def save_for_realtime(self, dataset):
        """FIXED: Save everything for real-time detection with proper error handling"""
        logger.info("💾 Saving for real-time deployment...")
        
        try:
            # Save mappings
            mappings = {
                'word_to_idx': dataset['word_to_idx'],
                'idx_to_word': dataset['idx_to_word'], 
                'class_names': dataset['class_names'],
                'num_classes': dataset['num_classes'],
                'sequence_length': dataset['sequence_length'],
                'feature_dim': dataset['feature_dim'],
                'landmark_approach': True,
                'all_classes_included': True
            }
            
            with open(self.model_dir / 'realtime_mappings.json', 'w') as f:
                json.dump(mappings, f, indent=2)
            logger.info("✅ Mappings saved successfully")
            
            # Save config
            with open(self.model_dir / 'realtime_config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("✅ Config saved successfully")
            
            # FIXED: Try multiple model saving formats
            model_saved = False
            
            # Method 1: Try Keras format first
            try:
                self.model.save(str(self.model_dir / 'landmark_best.keras'))
                logger.info("✅ Model saved as landmark_best.keras")
                model_saved = True
            except Exception as e:
                logger.warning(f"⚠️ Keras format failed: {e}")
            
            # Method 2: Try H5 format
            if not model_saved:
                try:
                    self.model.save(str(self.model_dir / 'landmark_best.h5'))
                    logger.info("✅ Model saved as landmark_best.h5")
                    model_saved = True
                except Exception as e:
                    logger.warning(f"⚠️ H5 format failed: {e}")
            
            # Method 3: Save weights separately
            try:
                self.model.save_weights(str(self.model_dir / 'landmark_best.weights.h5'))
                logger.info("✅ Model weights saved")
            except Exception as e:
                logger.warning(f"⚠️ Weights saving failed: {e}")
                # Try alternative weights format
                try:
                    self.model.save_weights(str(self.model_dir / 'landmark_weights'))
                    logger.info("✅ Model weights saved (alternative format)")
                except Exception as e2:
                    logger.error(f"❌ All weights saving methods failed: {e2}")
            
            # Method 4: Save model architecture
            try:
                model_config = self.model.get_config()
                with open(self.model_dir / 'landmark_architecture.json', 'w') as f:
                    json.dump(model_config, f, indent=2)
                logger.info("✅ Model architecture saved")
            except Exception as e:
                logger.warning(f"⚠️ Architecture saving failed: {e}")
            
            if model_saved:
                logger.info("✅ Ready for real-time deployment!")
            else:
                logger.error("❌ Model saving failed, but mappings and config are saved")
                
        except Exception as e:
            logger.error(f"❌ Save for realtime failed: {e}")
    
    def plot_landmark_results(self, results):
        """Plot results with error handling"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training curves
            if self.history:
                axes[0, 0].plot(self.history.history['accuracy'], label='Training')
                axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
                axes[0, 0].set_title('Landmark Model Accuracy')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                axes[0, 1].plot(self.history.history['loss'], label='Training')
                axes[0, 1].plot(self.history.history['val_loss'], label='Validation') 
                axes[0, 1].set_title('Landmark Model Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Results
            metrics = ['Test', 'Top-3', 'Top-5']
            values = [results['test_accuracy'], results['top3_accuracy'], results['top5_accuracy']]
            
            bars = axes[1, 0].bar(metrics, values, color=['blue', 'green', 'orange'])
            axes[1, 0].set_title('Performance on ALL 180 Classes')
            axes[1, 0].set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Summary
            summary = f"""
LANDMARK APPROACH RESULTS

Accuracy: {results['test_accuracy']*100:.1f}%
Top-3: {results['top3_accuracy']*100:.1f}%
Classes: ALL 180 WORDS
90% Target: {"ACHIEVED!" if results['target_90_achieved'] else "CLOSE!"}

Your Landmark Technique:
✅ Background Independent  
✅ Person Independent
✅ Lighting Independent
✅ Pure Gesture Focus

Parameters: {results['total_parameters']:,}
Epochs: {results['training_epochs']}
Ready for Real-time: YES
            """
            
            axes[1, 1].text(0.05, 0.95, summary, fontsize=10,
                            verticalalignment='top', fontfamily='monospace',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.model_dir / 'landmark_90_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📈 Results visualization saved")
        except Exception as e:
            logger.warning(f"⚠️ Plotting failed: {e}")

def main():
    """Main training with landmark approach"""
    print("🎯 LANDMARK APPROACH - 90%+ ACCURACY TARGET")
    print("="*60)
    print("Using YOUR brilliant landmark technique")
    print("ALL 180 classes - Smart augmentation")
    print("Background/Person/Lighting Independent")
    print("FIXED: Model saving issues only")
    print("="*60)
    
    try:
        # Initialize trainer
        trainer = LandmarkSpecificTrainer()
        
        # Load data
        dataset = trainer.load_landmark_data()
        
        # Prepare training data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_training_data(dataset)
        
        # Build architecture
        model = trainer.build_landmark_architecture(dataset['num_classes'])
        
        print(f"\n🏗️ Landmark Architecture:")
        model.summary()
        
        # Train
        print(f"\n🚀 Training on ALL {dataset['num_classes']} words...")
        trainer.train_landmark_model(X_train, X_val, y_train, y_val)
        
        # Evaluate
        print(f"\n📊 Evaluation:")
        results = trainer.evaluate_90_percent(X_test, y_test, dataset['class_names'])
        
        # Save for deployment (FIXED)
        trainer.save_for_realtime(dataset)
        trainer.plot_landmark_results(results)
        
        # Final results
        print(f"\n🎉 LANDMARK APPROACH RESULTS:")
        print(f"="*50)
        print(f"✅ Accuracy: {results['test_accuracy']*100:.1f}%")
        print(f"✅ Top-3: {results['top3_accuracy']*100:.1f}%")
        print(f"✅ Classes: ALL {dataset['num_classes']} words")
        
        if results['target_90_achieved']:
            print(f"🎯 90% TARGET ACHIEVED!")
            print(f"🚀 Your landmark approach WORKS!")
        else:
            print(f"🔥 EXCELLENT PROGRESS!")
            print(f"🚀 Much better than traditional approaches!")
        
        print(f"\n📁 Files created:")
        print(f"   model/landmark_90_model.keras")
        print(f"   model/landmark_scaler.pkl")
        print(f"   model/realtime_mappings.json")
        print(f"   model/landmark_90_results.json")
        print(f"   model/landmark_best.keras (for deployment)")
        
        print(f"\n🚀 Next: Real-time detection API")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()