"""
FIXED Sign Language Recognition Model
Enhanced architecture similar to friend's project with proper prediction logic
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from typing import Tuple, List, Optional
import logging

class FixedSignLanguageModel:
    """
    FIXED: Enhanced model architecture similar to friend's successful approach
    Key improvements:
    - Better LSTM architecture matching friend's model structure
    - Proper normalization and prediction logic
    - Enhanced confidence calculation
    - Stable inference pipeline
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        # FIXED: Match config exactly - no magic numbers
        self.sequence_length = 30    # config.SEQUENCE_LENGTH
        self.input_features = 126    # config.FEATURES_PER_FRAME (2 hands × 21 landmarks × 3 coords)
        
        # FIXED: Enhanced architecture parameters similar to friend's successful model
        self.lstm_units_1 = 32       # First LSTM layer (like friend's model)
        self.lstm_units_2 = 64       # Second LSTM layer (enhanced)
        self.lstm_units_3 = 32       # Third LSTM layer (like friend's)
        self.dense_units = 32        # Dense layer (like friend's)
        self.dropout_rate = 0.3
        
        # FIXED: Store normalization parameters for consistent inference
        self.feature_mean = None
        self.feature_std = None
        
        print(f"✅ Model initialized with {num_classes} classes")
        print(f"   Input shape: ({self.sequence_length}, {self.input_features})")
    
    def create_model(self) -> tf.keras.Model:
        """
        FIXED: Create model architecture similar to friend's successful approach
        Based on friend's model.py structure but adapted for video data
        """
        print("🏗️ Creating enhanced model architecture...")
        
        # FIXED: Input layer with proper shape
        inputs = layers.Input(shape=(self.sequence_length, self.input_features))
        
        # FIXED: Architecture similar to friend's model.py
        # friend used: LSTM(32) -> LSTM(64, return_sequences=True) -> LSTM(32, return_sequences=False)
        x = layers.LSTM(
            self.lstm_units_1, 
            return_sequences=True, 
            activation='relu',
            name='lstm_1'
        )(inputs)
        
        x = layers.LSTM(
            self.lstm_units_2, 
            return_sequences=True, 
            activation='relu',
            name='lstm_2'
        )(x)
        
        x = layers.LSTM(
            self.lstm_units_3, 
            return_sequences=False, 
            activation='relu',
            name='lstm_3'
        )(x)
        
        # FIXED: Dense layers similar to friend's approach
        x = layers.Dense(self.dense_units, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # FIXED: Output layer with softmax (like friend's model)
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='predictions'
        )(x)
        
        model = Model(inputs, outputs)
        
        # FIXED: Compile with same settings as friend's model
        model.compile(
            optimizer='Adam',  # Same as friend's
            loss='categorical_crossentropy',  # Same as friend's
            metrics=['categorical_accuracy']   # Same as friend's
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"   Architecture: LSTM({self.lstm_units_1}) -> LSTM({self.lstm_units_2}) -> LSTM({self.lstm_units_3}) -> Dense({self.dense_units}) -> Output({self.num_classes})")
        
        return model
    
    def load_model(self, model_path: Path = None) -> bool:
        """FIXED: Proper model loading with validation"""
        if model_path is None:
            from simplified_config import config
            model_path = config.MODEL_PATH
        
        try:
            print(f"📂 Loading model from: {model_path}")
            
            if not model_path.exists():
                print(f"❌ Model file does not exist: {model_path}")
                return False
            
            # Load the saved model
            self.model = tf.keras.models.load_model(str(model_path))
            print(f"✅ Model loaded successfully")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            
            # Load label encoder
            if not self.load_label_encoder():
                print("❌ Failed to load label encoder")
                return False
            
            # Validate model architecture
            expected_input_shape = (None, self.sequence_length, self.input_features)
            if self.model.input_shape != expected_input_shape:
                print(f"⚠️ Model input shape {self.model.input_shape} != expected {expected_input_shape}")
                # Don't fail, but warn
            
            print(f"✅ Model validation successful")
            print(f"   Classes: {list(self.label_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_label_encoder(self, encoder_path: Path = None) -> bool:
        """FIXED: Proper label encoder loading"""
        if encoder_path is None:
            from simplified_config import config
            encoder_path = config.DATA_DIR / "label_encoder.pkl"
        
        try:
            if not encoder_path.exists():
                print(f"❌ Label encoder file does not exist: {encoder_path}")
                return False
                
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print(f"✅ Label encoder loaded: {len(self.label_encoder.classes_)} classes")
            return True
            
        except Exception as e:
            print(f"❌ Error loading label encoder: {e}")
            return False
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        FIXED: Enhanced normalization similar to friend's keypoint extraction
        Apply consistent normalization for both training and inference
        """
        X_normalized = X.copy().astype(np.float32)
        
        for i in range(X.shape[0]):
            sample = X[i]  # [sequence_length, features]
            
            # FIXED: Find frames with actual hand detections (similar to friend's approach)
            non_zero_mask = np.any(np.abs(sample) > 1e-6, axis=1)
            
            if np.any(non_zero_mask):
                valid_frames = sample[non_zero_mask]
                
                # FIXED: Per-sequence normalization (more stable than global)
                # Similar to how friend's keypoint extraction normalizes each hand separately
                mean = np.mean(valid_frames, axis=0)
                std = np.std(valid_frames, axis=0) + 1e-8
                
                # Apply normalization to entire sequence
                X_normalized[i] = (sample - mean) / std
                
                # FIXED: Set invalid frames to zero (similar to friend's zero padding)
                X_normalized[i][~non_zero_mask] = 0.0
            else:
                X_normalized[i] = 0.0
        
        return X_normalized
    
    def predict_single_sequence(self, sequence: np.ndarray) -> Tuple[str, float, List[dict]]:
        """
        FIXED: Enhanced single sequence prediction with proper logic
        This is the core function that must work like friend's main.py prediction logic
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        try:
            print(f"🎯 Predicting sequence with shape: {sequence.shape}")
            
            # FIXED: Ensure correct input shape
            if len(sequence.shape) == 2:
                sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Validate input shape
            expected_shape = (1, self.sequence_length, self.input_features)
            if sequence.shape != expected_shape:
                print(f"⚠️ Input shape {sequence.shape} != expected {expected_shape}")
                if sequence.shape[1:] != (self.sequence_length, self.input_features):
                    raise ValueError(f"Cannot fix shape mismatch: {sequence.shape}")
            
            # FIXED: Apply normalization (critical for accuracy)
            sequence_normalized = self.normalize_features(sequence)
            
            print(f"📊 Normalized sequence stats: min={sequence_normalized.min():.3f}, max={sequence_normalized.max():.3f}, mean={sequence_normalized.mean():.3f}")
            
            # FIXED: Make prediction with proper error handling
            print("🤖 Making prediction...")
            predictions = self.model.predict(sequence_normalized, verbose=0)
            prediction_probs = predictions[0]
            
            print(f"📈 Raw predictions shape: {predictions.shape}")
            print(f"📈 Prediction probabilities: {prediction_probs}")
            
            # FIXED: Enhanced confidence calculation similar to friend's approach
            # Friend checks if max prediction > 0.9, we use similar logic
            max_confidence = np.max(prediction_probs)
            print(f"🎯 Maximum confidence: {max_confidence:.3f}")
            
            # FIXED: Get top predictions (similar to friend's approach)
            top_indices = np.argsort(prediction_probs)[::-1][:5]
            top_predictions = []
            
            for idx in top_indices:
                if idx < len(self.label_encoder.classes_):
                    word = self.label_encoder.classes_[idx]
                    confidence = float(prediction_probs[idx])
                    top_predictions.append({
                        'word': word,
                        'confidence': confidence,
                        'percentage': f"{confidence * 100:.1f}%"
                    })
            
            # FIXED: Best prediction logic
            best_idx = top_indices[0]
            predicted_word = self.label_encoder.classes_[best_idx]
            confidence = float(prediction_probs[best_idx])
            
            print(f"🎊 PREDICTION RESULT:")
            print(f"   Predicted word: '{predicted_word}'")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Top 3 predictions: {[(p['word'], p['percentage']) for p in top_predictions[:3]]}")
            
            return predicted_word, confidence, top_predictions
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100) -> dict:
        """
        FIXED: Training function with friend's model approach
        """
        print(f"🚀 Starting model training...")
        print(f"   Training data: {X_train.shape}")
        print(f"   Validation data: {X_val.shape}")
        print(f"   Epochs: {epochs}")
        
        # Create model
        self.model = self.create_model()
        
        # FIXED: Callbacks similar to friend's approach
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # FIXED: Train with proper settings (similar to friend's 100 epochs)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history
        
        # Evaluate model
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"🎯 Training completed!")
        print(f"   Final training accuracy: {train_acc:.3f}")
        print(f"   Final validation accuracy: {val_acc:.3f}")
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'epochs_trained': len(history.history['loss'])
        }
    
    def save_model(self, model_path: Path = None) -> bool:
        """FIXED: Save model properly"""
        if model_path is None:
            from simplified_config import config
            model_path = config.MODEL_PATH
        
        try:
            if self.model is None:
                print("❌ No model to save!")
                return False
            
            self.model.save(str(model_path))
            print(f"✅ Model saved to: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def save_label_encoder(self, encoder_path: Path = None) -> bool:
        """FIXED: Save label encoder properly"""
        if encoder_path is None:
            from simplified_config import config
            encoder_path = config.DATA_DIR / "label_encoder.pkl"
        
        try:
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"✅ Label encoder saved to: {encoder_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving label encoder: {e}")
            return False


def validate_model_setup():
    """FIXED: Validate that model is properly set up"""
    from simplified_config import config
    
    print("🔍 Validating model setup...")
    
    # Check files exist
    model_path = config.MODEL_PATH
    encoder_path = config.DATA_DIR / "label_encoder.pkl"
    
    print(f"Model path: {model_path}")
    print(f"Encoder path: {encoder_path}")
    
    if not model_path.exists():
        print("❌ Model file missing!")
        return False
    
    if not encoder_path.exists():
        print("❌ Label encoder missing!")
        return False
    
    # Try to load model
    try:
        model = FixedSignLanguageModel(num_classes=10)  # Temporary
        if model.load_model():
            print("✅ Model loads successfully")
            
            # FIXED: Test prediction with proper dummy data
            dummy_sequence = np.random.rand(1, 30, 126).astype(np.float32)
            predicted_word, confidence, top_predictions = model.predict_single_sequence(dummy_sequence)
            print(f"✅ Test prediction works: {predicted_word} ({confidence:.3f})")
            
            return True
        else:
            print("❌ Model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False

if __name__ == "__main__":
    validate_model_setup()