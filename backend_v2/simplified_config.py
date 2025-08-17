"""
Simplified Configuration for Sign Language Recognition
Optimized for 90%+ accuracy with limited data per class
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Simple, production-ready configuration"""
    
    # === PROJECT STRUCTURE ===
    BASE_DIR: Path = Path(__file__).parent
    VIDEOS_DIR: Path = BASE_DIR / "videos"
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # === MEDIAPIPE SETTINGS ===
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MAX_NUM_HANDS: int = 2
    
    # === MODEL ARCHITECTURE (Simplified) ===
    SEQUENCE_LENGTH: int = 30  # 1 second at 30fps
    FEATURES_PER_FRAME: int = 126  # 2 hands × 21 landmarks × 3 coords
    LSTM_UNITS: int = 64  # Smaller model for limited data
    DENSE_UNITS: int = 128
    DROPOUT_RATE: float = 0.3
    
    # === TRAINING SETTINGS ===
    BATCH_SIZE: int = 16
    EPOCHS: int = 100  # Reduced from 200
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    AUGMENTATION_FACTOR: int = 10  # Realistic augmentation
    
    # === REAL-TIME INFERENCE ===
    CONFIDENCE_THRESHOLD: float = 0.75
    SMOOTHING_WINDOW: int = 5  # For temporal smoothing
    
    # === API SETTINGS ===
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 5000
    CORS_ORIGINS: list = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Initialize derived values"""
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
        
        # Create directories
        for dir_path in [self.VIDEOS_DIR, self.MODELS_DIR, self.DATA_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / "sign_model.h5"
    
    @property
    def LABEL_ENCODER_PATH(self) -> Path:
        return self.MODELS_DIR / "label_encoder.pkl"
    
    @property
    def TRAINING_DATA_PATH(self) -> Path:
        return self.DATA_DIR / "training_data.npz"

# Global config instance
config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment overrides
if os.getenv('API_PORT'):
    config.API_PORT = int(os.getenv('API_PORT'))
if os.getenv('CONFIDENCE_THRESHOLD'):
    config.CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD'))