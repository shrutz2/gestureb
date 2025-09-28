import numpy as np
import tensorflow as tf
import os
import random
import git
from datetime import datetime
import json
from pathlib import Path

# --- Constants for Reproducibility ---
SEQUENCE_LENGTH = 30
FEATURE_DIM = 126  # 2 hands * 21 points * 3 coords (x,y,z)

def set_all_seeds(seed=42):
    """Set seeds for reproducibility across numpy, random, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def get_git_sha():
    """Get the current Git commit SHA for model versioning."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except git.InvalidGitRepositoryError:
        return 'no-git-repo'

def save_metadata(path, metadata, filename="metadata.json"):
    """Saves metadata to a JSON file."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / filename, 'w') as f:
        json.dump(metadata, f, indent=4)

def get_model_artifact_name(model_name, extension=".keras"):
    """Generates a versioned artifact name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    git_sha = get_git_sha()
    return f"{model_name}_v{timestamp}_{git_sha}{extension}"

def normalize_landmarks_sequence(sequence):
    """
    Normalizes coordinates relative to the wrist/palm base (0th landmark).
    This creates scale and translation invariance.
    
    The input is a flattened array [x1, y1, z1, x2, y2, z2, ...]
    """
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 3) # Reshape to (N_landmarks, 3)
        
    # Assuming the first point (index 0) is the wrist/palm base
    # Hand 1: points 0-20. Hand 2: points 21-41
    
    # Check if hand 1 is present (e.g., if wrist x is non-zero)
    if np.any(sequence[0] != 0):
        # Subtract wrist coordinates from all points in the first hand
        wrist_coords = sequence[0]
        sequence[:21] = sequence[:21] - wrist_coords
    
    # Check if hand 2 is present (e.g., if wrist x is non-zero)
    if sequence.shape[0] > 21 and np.any(sequence[21] != 0):
        # Subtract wrist coordinates from all points in the second hand
        wrist_coords_2 = sequence[21]
        sequence[21:] = sequence[21:] - wrist_coords_2
        
    # Flatten back to a feature vector
    return sequence.flatten()