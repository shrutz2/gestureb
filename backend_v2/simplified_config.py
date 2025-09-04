"""
FIXED Simplified Configuration for Windows
Removed emojis that cause Unicode issues on Windows terminal
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """FIXED: Configuration without emoji characters for Windows compatibility"""
    
    # === PROJECT STRUCTURE ===
    BASE_DIR: Path = Path(__file__).parent
    VIDEOS_DIR: Path = BASE_DIR / "videos"
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # === ENHANCED MEDIAPIPE SETTINGS ===
    MIN_DETECTION_CONFIDENCE: float = 0.8
    MIN_TRACKING_CONFIDENCE: float = 0.7
    MAX_NUM_HANDS: int = 2
    
    # === MODEL ARCHITECTURE ===
    SEQUENCE_LENGTH: int = 30
    FEATURES_PER_FRAME: int = 126
    
    LSTM_UNITS_1: int = 32
    LSTM_UNITS_2: int = 64
    LSTM_UNITS_3: int = 32
    DENSE_UNITS: int = 32
    DROPOUT_RATE: float = 0.3
    
    # === TRAINING SETTINGS ===
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.15
    AUGMENTATION_FACTOR: int = 5
    
    # === INFERENCE SETTINGS ===
    CONFIDENCE_THRESHOLD: float = 0.7
    HIGH_CONFIDENCE_THRESHOLD: float = 0.9
    SMOOTHING_WINDOW: int = 5
    PREDICTION_BUFFER_SIZE: int = 7
    
    # === HAND DETECTION ===
    HAND_CONFIDENCE_THRESHOLD: float = 0.8
    MIN_VALID_FRAMES: int = 15
    MIN_PREDICTION_FRAMES: int = 8
    STABILITY_WINDOW: int = 5
    
    # === DATA PROCESSING ===
    FRAME_SAMPLING_RATE: int = 15
    MAX_SEQUENCE_LENGTH: int = 45
    MIN_SEQUENCE_LENGTH: int = 15
    
    # === API SETTINGS ===
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 5000
    CORS_ORIGINS: list = None
    
    # === QUALITY CONTROL ===
    ENABLE_QUALITY_FILTERING: bool = True
    LANDMARK_VARIANCE_THRESHOLD: float = 1e-4
    CONFIDENCE_STABILITY_THRESHOLD: float = 0.1
    
    def __post_init__(self):
        """Initialize derived values and create directories"""
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
        
        # Create directories
        for dir_path in [self.VIDEOS_DIR, self.MODELS_DIR, self.DATA_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        issues = []
        
        if self.CONFIDENCE_THRESHOLD >= self.HIGH_CONFIDENCE_THRESHOLD:
            issues.append("CONFIDENCE_THRESHOLD should be < HIGH_CONFIDENCE_THRESHOLD")
        
        if self.MIN_VALID_FRAMES < 10:
            issues.append("MIN_VALID_FRAMES should be >= 10 for reliable training")
        
        if self.SEQUENCE_LENGTH < 20:
            issues.append("SEQUENCE_LENGTH should be >= 20 for gesture recognition")
        
        if issues:
            for issue in issues:
                print(f"Config issue: {issue}")
            return False
        
        return True
    
    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / "sign_model.h5"
    
    @property
    def LABEL_ENCODER_PATH(self) -> Path:
        return self.DATA_DIR / "label_encoder.pkl"
    
    @property
    def TRAINING_DATA_PATH(self) -> Path:
        return self.DATA_DIR / "training_data.npz"
    
    @property
    def CLASS_MAPPING_PATH(self) -> Path:
        return self.DATA_DIR / "class_mapping.json"
    
    @property
    def TRAINING_RESULTS_PATH(self) -> Path:
        return self.DATA_DIR / "training_results.json"
    
    def get_model_config(self) -> dict:
        """Get model configuration dictionary"""
        return {
            'sequence_length': self.SEQUENCE_LENGTH,
            'features_per_frame': self.FEATURES_PER_FRAME,
            'lstm_units': [self.LSTM_UNITS_1, self.LSTM_UNITS_2, self.LSTM_UNITS_3],
            'dense_units': self.DENSE_UNITS,
            'dropout_rate': self.DROPOUT_RATE,
            'learning_rate': self.LEARNING_RATE
        }
    
    def get_processing_config(self) -> dict:
        """Get data processing configuration"""
        return {
            'min_detection_confidence': self.MIN_DETECTION_CONFIDENCE,
            'min_tracking_confidence': self.MIN_TRACKING_CONFIDENCE,
            'hand_confidence_threshold': self.HAND_CONFIDENCE_THRESHOLD,
            'min_valid_frames': self.MIN_VALID_FRAMES,
            'frame_sampling_rate': self.FRAME_SAMPLING_RATE,
            'sequence_length': self.SEQUENCE_LENGTH
        }

# Global config instance
config = Config()

# FIXED: Windows-compatible logging setup
def setup_windows_logging():
    """Setup Windows-compatible logging without emoji characters"""
    
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure logging for Windows with proper encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOGS_DIR / 'enhanced_app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('SignLanguageApp')
    
    # Windows-safe validation
    if config.validate_config():
        logger.info("Configuration validated successfully")
    else:
        logger.warning("Configuration has issues - check settings")
    
    return logger

def apply_environment_overrides():
    """Apply environment variable overrides"""
    if os.getenv('API_PORT'):
        config.API_PORT = int(os.getenv('API_PORT'))
        print(f"API_PORT override: {config.API_PORT}")
    
    if os.getenv('CONFIDENCE_THRESHOLD'):
        config.CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD'))
        print(f"CONFIDENCE_THRESHOLD override: {config.CONFIDENCE_THRESHOLD}")
    
    if os.getenv('SEQUENCE_LENGTH'):
        config.SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH'))
        print(f"SEQUENCE_LENGTH override: {config.SEQUENCE_LENGTH}")
    
    if os.getenv('MIN_DETECTION_CONFIDENCE'):
        config.MIN_DETECTION_CONFIDENCE = float(os.getenv('MIN_DETECTION_CONFIDENCE'))
        print(f"MIN_DETECTION_CONFIDENCE override: {config.MIN_DETECTION_CONFIDENCE}")

# Setup logging and apply overrides
logger = setup_windows_logging()
apply_environment_overrides()

# Windows-safe startup log without emojis
logger.info("Enhanced Sign Language Recognition Configuration")
logger.info("=" * 50)
logger.info(f"Base directory: {config.BASE_DIR}")
logger.info(f"Videos directory: {config.VIDEOS_DIR}")
logger.info(f"Models directory: {config.MODELS_DIR}")
logger.info(f"Data directory: {config.DATA_DIR}")
logger.info(f"Logs directory: {config.LOGS_DIR}")
logger.info("")
logger.info("Model Configuration:")
logger.info(f"   Sequence length: {config.SEQUENCE_LENGTH}")
logger.info(f"   Features per frame: {config.FEATURES_PER_FRAME}")
logger.info(f"   LSTM architecture: {config.LSTM_UNITS_1} -> {config.LSTM_UNITS_2} -> {config.LSTM_UNITS_3}")
logger.info(f"   Dense units: {config.DENSE_UNITS}")
logger.info(f"   Dropout rate: {config.DROPOUT_RATE}")
logger.info("")
logger.info("Detection Configuration:")
logger.info(f"   Min detection confidence: {config.MIN_DETECTION_CONFIDENCE}")
logger.info(f"   Min tracking confidence: {config.MIN_TRACKING_CONFIDENCE}")
logger.info(f"   Hand confidence threshold: {config.HAND_CONFIDENCE_THRESHOLD}")
logger.info(f"   Prediction confidence threshold: {config.CONFIDENCE_THRESHOLD}")
logger.info(f"   High confidence threshold: {config.HIGH_CONFIDENCE_THRESHOLD}")
logger.info("")
logger.info("Training Configuration:")
logger.info(f"   Epochs: {config.EPOCHS}")
logger.info(f"   Batch size: {config.BATCH_SIZE}")
logger.info(f"   Learning rate: {config.LEARNING_RATE}")
logger.info(f"   Validation split: {config.VALIDATION_SPLIT}")
logger.info(f"   Augmentation factor: {config.AUGMENTATION_FACTOR}")
logger.info("")
logger.info("API Configuration:")
logger.info(f"   Host: {config.API_HOST}")
logger.info(f"   Port: {config.API_PORT}")
logger.info(f"   CORS origins: {config.CORS_ORIGINS}")
logger.info("=" * 50)
logger.info("Windows-compatible configuration loaded successfully")