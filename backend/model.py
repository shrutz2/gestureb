#!/usr/bin/env python3
"""
ENHANCED Sign Language Recognition Model
Fixes low accuracy issues with advanced architecture and training strategies
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import warnings
from collections import Counter
import random

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class EnhancedSignLanguageTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Enhanced configuration for better performance
        self.config = {
            'sequence_length': 15,
            'feature_dim': 158,
            'conv1d_filters': [64, 128, 256],
            'lstm_units': [256, 128],
            'attention_units': 128,
            'dense_units': [512, 256],
            'dropout_rate': 0.4,
            'spatial_dropout': 0.2,
            'recurrent_dropout': 0.2,
            'l1_reg': 0.00001,
            'l2_reg': 0.0001,
            'learning_rate': 0.0005,
            'batch_size': 16,  # Smaller batch for better gradient updates
            'max_epochs': 200,
            'patience': 30,
            'warmup_epochs': 10,
            'augmentation_prob': 0.3
        }
        
        logger.info("🚀 Enhanced Sign Language Trainer initialized")
    
    def load_dataset(self):
        """Load and analyze dataset"""
        dataset_path = self.models_dir / 'robust_dataset.pkl'
        
        if not dataset_path.exists():
            raise FileNotFoundError("❌ Dataset not found. Run preprocessing first.")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Analyze class distribution
        train_counts = Counter(dataset['y_train'])
        logger.info(f"📊 Dataset Analysis:")
        logger.info(f"   📝 Total classes: {dataset['num_classes']}")
        logger.info(f"   🎯 Training samples: {len(dataset['X_train'])}")
        logger.info(f"   ✅ Validation samples: {len(dataset['X_val'])}")
        logger.info(f"   🧪 Test samples: {len(dataset['X_test'])}")
        logger.info(f"   📏 Feature dimension: {dataset['feature_dim']}")
        logger.info(f"   ⏱️ Sequence length: {dataset['sequence_length']}")
        logger.info(f"   ⚖️ Min/Max samples per class: {min(train_counts.values())}/{max(train_counts.values())}")
        
        return dataset
    
    def augment_sequence(self, sequence, prob=0.3):
        """Advanced data augmentation for sign language sequences"""
        if np.random.random() > prob:
            return sequence
        
        augmented = sequence.copy()
        
        # 1. Time warping - slight speed variations
        if np.random.random() < 0.3:
            # Randomly remove 1-2 frames
            indices_to_remove = np.random.choice(len(augmented), size=min(2, len(augmented)//3), replace=False)
            augmented = np.delete(augmented, indices_to_remove, axis=0)
        
        # 2. Gaussian noise - simulate camera noise
        if np.random.random() < 0.4:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = augmented + noise
        
        # 3. Spatial scaling - simulate distance variations
        if np.random.random() < 0.3:
            scale_factor = np.random.uniform(0.95, 1.05)
            # Scale only spatial coordinates (x, y) not z or visibility
            for i in range(augmented.shape[0]):
                # Hand landmarks (x,y,z pattern)
                for j in range(0, 126, 3):
                    augmented[i, j] *= scale_factor      # x
                    augmented[i, j+1] *= scale_factor    # y
                # Pose landmarks (x,y,z,v pattern)
                for j in range(126, 158, 4):
                    augmented[i, j] *= scale_factor      # x
                    augmented[i, j+1] *= scale_factor    # y
        
        # 4. Temporal shifting - simulate timing variations
        if np.random.random() < 0.2:
            shift = np.random.randint(-2, 3)
            if shift > 0:
                augmented = np.concatenate([augmented[shift:], np.repeat(augmented[-1:], shift, axis=0)])
            elif shift < 0:
                augmented = np.concatenate([np.repeat(augmented[:1], -shift, axis=0), augmented[:shift]])
        
        return augmented
    
    def create_data_generator(self, X, y, batch_size, augment=False):
        """Create data generator with augmentation"""
        def generator():
            indices = np.arange(len(X))
            while True:
                np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    batch_indices = indices[start:start + batch_size]
                    batch_X = []
                    batch_y = []
                    
                    for idx in batch_indices:
                        sequence = X[idx]
                        if augment:
                            sequence = self.augment_sequence(sequence, self.config['augmentation_prob'])
                        
                        # Ensure correct sequence length
                        if len(sequence) != self.config['sequence_length']:
                            sequence = self.normalize_sequence_length(sequence)
                        
                        batch_X.append(sequence)
                        batch_y.append(y[idx])
                    
                    yield np.array(batch_X), np.array(batch_y)
        
        return generator
    
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
            # Repeat last frame
            padded = np.zeros((target_length, self.config['feature_dim']))
            padded[:current_length] = sequence
            if current_length > 0:
                padded[current_length:] = sequence[-1]
            return padded
    
    def normalize_features(self, X_train, X_val, X_test):
        """Enhanced feature normalization"""
        logger.info("🔧 Normalizing features with enhanced preprocessing...")
        
        # Reshape for scaling
        train_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, self.config['feature_dim'])
        X_val_flat = X_val.reshape(-1, self.config['feature_dim'])
        X_test_flat = X_test.reshape(-1, self.config['feature_dim'])
        
        # Robust scaling with outlier handling
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Clip extreme values
        X_train_scaled = np.clip(X_train_scaled, -3, 3)
        X_val_scaled = np.clip(X_val_scaled, -3, 3)
        X_test_scaled = np.clip(X_test_scaled, -3, 3)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(train_shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        logger.info("✅ Enhanced feature normalization completed")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def build_enhanced_model(self, num_classes):
        """Build enhanced model with CNN + LSTM + Attention"""
        logger.info(f"🏗️ Building enhanced model for {num_classes} classes...")
        
        # Input
        inputs = layers.Input(
            shape=(self.config['sequence_length'], self.config['feature_dim']),
            name='sequence_input'
        )
        
        # Masking for variable length sequences
        x = layers.Masking(mask_value=0.0)(inputs)
        
        # 1D Convolutional layers for local pattern extraction
        x = layers.Conv1D(
            filters=self.config['conv1d_filters'][0],
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(self.config['spatial_dropout'])(x)
        
        x = layers.Conv1D(
            filters=self.config['conv1d_filters'][1],
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(self.config['spatial_dropout'])(x)
        
        # Bidirectional LSTM layers for temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(
                units=self.config['lstm_units'][0],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['recurrent_dropout'],
                kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
            )
        )(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Bidirectional(
            layers.LSTM(
                units=self.config['lstm_units'][1],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['recurrent_dropout'],
                kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
            )
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.Dense(
            self.config['attention_units'],
            activation='tanh',
            kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
        )(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Reshape((-1, 1))(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense classification layers
        x = layers.Dense(
            units=self.config['dense_units'][0],
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
        )(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.Dense(
            units=self.config['dense_units'][1],
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.config['l1_reg'], l2=self.config['l2_reg'])
        )(x)
        x = layers.Dropout(self.config['dropout_rate'] * 0.5)(x)
        
        # Output layer
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='EnhancedSignLanguageModel')
        
        # Enhanced optimizer with learning rate scheduling
        optimizer = optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        logger.info(f"✅ Enhanced model built: {model.count_params():,} parameters")
        return model
    
    def get_enhanced_callbacks(self):
        """Enhanced callbacks for optimal training"""
        return [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            callbacks.ModelCheckpoint(
                str(self.models_dir / 'best_enhanced_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            ),
            
            # Cosine annealing for better convergence
            callbacks.LearningRateScheduler(
                lambda epoch: self.config['learning_rate'] * 
                (np.cos(np.pi * epoch / self.config['max_epochs']) + 1) / 2
                if epoch > self.config['warmup_epochs'] else self.config['learning_rate']
            ),
            
            callbacks.CSVLogger(
                str(self.models_dir / 'enhanced_training_log.csv'),
                append=False
            )
        ]
    
    def train_enhanced_model(self, X_train, y_train, X_val, y_val):
        """Enhanced training with data augmentation and class balancing"""
        logger.info("🚀 Starting enhanced training...")
        
        # Compute balanced class weights
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        logger.info(f"⚖️ Applied balanced class weights for {len(class_weight_dict)} classes")
        
        # Create data generators
        train_gen = self.create_data_generator(X_train, y_train, self.config['batch_size'], augment=True)
        val_gen = self.create_data_generator(X_val, y_val, self.config['batch_size'], augment=False)
        
        steps_per_epoch = len(X_train) // self.config['batch_size']
        validation_steps = len(X_val) // self.config['batch_size']
        
        # Get callbacks
        callbacks_list = self.get_enhanced_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_gen(),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen(),
            validation_steps=validation_steps,
            epochs=self.config['max_epochs'],
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        logger.info("✅ Enhanced training completed")
        return self.history
    
    def evaluate_enhanced_model(self, X_test, y_test, class_names):
        """Comprehensive evaluation with detailed metrics"""
        logger.info("📊 Evaluating enhanced model...")
        
        # Load best model
        self.model = tf.keras.models.load_model(str(self.models_dir / 'best_enhanced_model.keras'))
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted_classes)
        top3_accuracy = self._calculate_topk_accuracy(predictions, y_test, k=3)
        top5_accuracy = self._calculate_topk_accuracy(predictions, y_test, k=5)
        
        logger.info(f"🎯 Enhanced Model Results:")
        logger.info(f"   ⚡ Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   🥉 Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
        logger.info(f"   🏅 Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
        
        # Performance status
        if accuracy >= 0.7:
            status = "🎉 EXCELLENT - Ready for deployment!"
        elif accuracy >= 0.5:
            status = "✅ GOOD - Suitable for most use cases"
        elif top3_accuracy >= 0.7:
            status = "📈 PROMISING - Good top-3 performance"
        else:
            status = "⚠️ NEEDS IMPROVEMENT - Consider more data/training"
        
        logger.info(f"   📊 Status: {status}")
        
        # Detailed results
        results = {
            'model_info': {
                'total_parameters': self.model.count_params(),
                'num_classes': len(class_names),
                'architecture': 'Enhanced CNN-LSTM-Attention',
                'enhancements': ['Data Augmentation', 'Attention Mechanism', 'Bidirectional LSTM', 'Class Balancing']
            },
            'performance': {
                'top1_accuracy': float(accuracy),
                'top3_accuracy': float(top3_accuracy),
                'top5_accuracy': float(top5_accuracy),
                'test_samples': len(X_test),
                'status': status
            },
            'training_config': self.config,
            'class_names': class_names[:100]  # Save first 100 for space
        }
        
        # Save results
        with open(self.models_dir / 'enhanced_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("💾 Enhanced evaluation results saved")
        return results
    
    def _calculate_topk_accuracy(self, predictions, y_true, k):
        """Calculate top-k accuracy"""
        top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] in top_k_predictions[i]:
                correct += 1
        return correct / len(y_true)
    
    def save_enhanced_artifacts(self):
        """Save all enhanced artifacts"""
        logger.info("💾 Saving enhanced artifacts...")
        
        # Save final model
        self.model.save(str(self.models_dir / 'enhanced_sign_language_model.keras'))
        
        # Save scaler
        with open(self.models_dir / 'enhanced_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training history
        if self.history:
            with open(self.models_dir / 'enhanced_training_history.pkl', 'wb') as f:
                pickle.dump(self.history.history, f)
        
        # Save configuration
        with open(self.models_dir / 'enhanced_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("✅ Enhanced artifacts saved for deployment")
    
    def plot_enhanced_results(self):
        """Plot enhanced training results"""
        if not self.history:
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2, color='blue')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2, color='red')
        axes[0, 0].set_title('Enhanced Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2, color='blue')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2, color='red')
        axes[0, 1].set_title('Enhanced Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-k accuracy plot
        if 'top_k_categorical_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], 
                           label='Training Top-K', linewidth=2, color='green')
            axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], 
                           label='Validation Top-K', linewidth=2, color='orange')
            axes[1, 0].set_title('Top-K Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-K Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary
        final_acc = self.history.history['val_accuracy'][-1]
        best_acc = max(self.history.history['val_accuracy'])
        
        summary_text = f"""
        🚀 ENHANCED MODEL TRAINING SUMMARY
        
        📊 Final Validation Accuracy: {final_acc:.4f}
        🎯 Best Validation Accuracy: {best_acc:.4f}
        📈 Total Epochs: {len(self.history.history['accuracy'])}
        🔧 Model Parameters: {self.model.count_params():,}
        
        🏗️ Architecture: CNN + BiLSTM + Attention
        📝 Classes: {len(self.model.layers[-1].get_weights()[0][0])} words
        🎯 Enhancements:
           • Data Augmentation
           • Attention Mechanism  
           • Bidirectional Processing
           • Class Balancing
           • Advanced Regularization
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, 
                        verticalalignment='top', fontfamily='monospace',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'enhanced_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📈 Enhanced training plots saved")


def main():
    """Main enhanced training pipeline"""
    print("🚀" + "="*79)
    print("🎯 ENHANCED SIGN LANGUAGE MODEL TRAINING")
    print("🚀" + "="*79)
    print("🔧 Features: CNN + BiLSTM + Attention + Data Augmentation")
    print("⚡ Target: Significantly improved accuracy on 666+ words")
    print("📊 Strategy: Advanced architecture + training techniques")
    print("🚀" + "="*79)
    
    try:
        # Initialize enhanced trainer
        trainer = EnhancedSignLanguageTrainer()
        
        # Load dataset
        dataset = trainer.load_dataset()
        
        # Extract data
        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_val = dataset['y_val']
        y_test = dataset['y_test']
        class_names = dataset['class_names']
        
        # Enhanced feature normalization
        X_train_norm, X_val_norm, X_test_norm = trainer.normalize_features(X_train, X_val, X_test)
        
        # Build enhanced model
        model = trainer.build_enhanced_model(dataset['num_classes'])
        
        print(f"\n🏗️ Enhanced Model Architecture:")
        print("-" * 60)
        model.summary(line_length=80)
        
        # Train enhanced model
        print(f"\n🚀 Starting enhanced training on {dataset['num_classes']} words...")
        print("-" * 60)
        trainer.train_enhanced_model(X_train_norm, y_train, X_val_norm, y_val)
        
        # Evaluate enhanced model
        print(f"\n📊 Evaluating enhanced model performance...")
        print("-" * 60)
        results = trainer.evaluate_enhanced_model(X_test_norm, y_test, class_names)
        
        # Save all enhanced artifacts
        trainer.save_enhanced_artifacts()
        trainer.plot_enhanced_results()
        
        # Final summary
        print("\n🎉" + "="*79)
        print("✅ ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉" + "="*79)
        print(f"🎯 Enhanced Results:")
        print(f"   ⚡ Top-1 Accuracy: {results['performance']['top1_accuracy']*100:.2f}%")
        print(f"   🥉 Top-3 Accuracy: {results['performance']['top3_accuracy']*100:.2f}%")
        print(f"   🏅 Top-5 Accuracy: {results['performance']['top5_accuracy']*100:.2f}%")
        print(f"   📊 Status: {results['performance']['status']}")
        print(f"   🔧 Total Parameters: {results['model_info']['total_parameters']:,}")
        
        print(f"\n📦 Enhanced Artifacts:")
        print(f"   🤖 Model: models/enhanced_sign_language_model.keras")
        print(f"   🔧 Scaler: models/enhanced_scaler.pkl")
        print(f"   ⚙️ Config: models/enhanced_config.json")
        print(f"   📊 Results: models/enhanced_evaluation_results.json")
        print(f"   📈 Plots: models/enhanced_training_results.png")
        
        print(f"\n🚀 Next Steps:")
        if results['performance']['top1_accuracy'] >= 0.5:
            print(f"   ✅ Model ready for deployment!")
            print(f"   🖥️ Test with: python app.py")
            print(f"   🌐 Deploy with: docker-compose up")
        else:
            print(f"   📈 Model shows improvement - consider:")
            print(f"   📊 More training data per class")
            print(f"   🔄 Longer training with current architecture")
            print(f"   🎯 Focus on top-performing classes for demo")
        
        print("🎉" + "="*79)
        
    except Exception as e:
        logger.error(f"❌ Enhanced training failed: {e}")
        raise


if __name__ == "__main__":
    main()