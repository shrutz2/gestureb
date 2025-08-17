#!/usr/bin/env python3
"""
Quick Setup Script for Sign Language Recognition System
Handles initialization, video copying, and basic validation
"""
import os
import sys
import shutil
from pathlib import Path
import argparse
import subprocess

def print_banner():
    """Print setup banner"""
    banner = """
    🤟 Sign Language Recognition System v2.0 Setup
    ================================================
    Production-ready real-time ASL recognition
    """
    print(banner)

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")

def create_directories():
    """Create required directories"""
    directories = [
        "videos",
        "models", 
        "logs",
        "data",
        "cache",
        "models/plots"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def copy_videos_from_old_backend():
    """Copy videos from old backend if available"""
    old_backend_videos = Path("../backend/videos")
    new_videos_dir = Path("videos")
    
    if old_backend_videos.exists():
        print(f"📹 Found old backend videos in {old_backend_videos}")
        
        video_files = list(old_backend_videos.glob("*.mp4"))
        if video_files:
            print(f"Copying {len(video_files)} video files...")
            
            for video_file in video_files:
                dest_file = new_videos_dir / video_file.name
                if not dest_file.exists():
                    shutil.copy2(video_file, dest_file)
                    print(f"  ✅ Copied: {video_file.name}")
                else:
                    print(f"  ⏭️ Exists: {video_file.name}")
        else:
            print("⚠️ No video files found in old backend")
    else:
        print("ℹ️ No old backend found - you'll need to add videos manually")

def copy_label_map():
    """Copy label map from old backend"""
    old_label_map = Path("../backend/label_map.pbtxt")
    new_label_map = Path("label_map.pbtxt")
    
    if old_label_map.exists() and not new_label_map.exists():
        shutil.copy2(old_label_map, new_label_map)
        print("✅ Copied label_map.pbtxt from old backend")

def install_requirements():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running manually: pip install -r requirements.txt")

def create_env_file():
    """Create environment configuration file"""
    env_content = """# Sign Language Recognition System Configuration
# Uncomment and modify as needed

# Performance Settings
USE_GPU=false
MIXED_PRECISION=true
BATCH_INFERENCE=true

# Model Parameters
SEQUENCE_LENGTH=45
CONFIDENCE_THRESHOLD=0.75
PREDICTION_BUFFER_SIZE=7

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
MAX_WORKERS=4

# Training Settings
AUGMENTATION_MULTIPLIER=50
EPOCHS=200
BATCH_SIZE=16

# Logging
LOG_LEVEL=INFO
DEBUG=false
"""
    
    env_file = Path(".env.example")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env.example file")

def validate_setup():
    """Validate the setup"""
    print("\n🔍 Validating setup...")
    
    # Check for videos
    video_dir = Path("videos")
    video_files = list(video_dir.glob("*.mp4"))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video files")
        # Show first few video names
        for video in video_files[:5]:
            print(f"  📹 {video.stem}")
        if len(video_files) > 5:
            print(f"  ... and {len(video_files) - 5} more")
    else:
        print("⚠️ No video files found in videos/ directory")
        print("   Add your sign language videos (format: word.mp4)")
    
    # Check key imports
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import mediapipe as mp
        print(f"✅ MediaPipe: {mp.__version__}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")

def print_next_steps():
    """Print next steps for user"""
    next_steps = """
    🎉 Setup Complete! Next Steps:
    ===============================
    
    1. 📹 Add Training Videos (if not already done):
       - Copy sign language videos to backend_v2/videos/
       - Format: word.mp4 (e.g., hello.mp4, thanks.mp4)
    
    2. 🚀 Train Your Model:
       python training/advanced_trainer.py
    
    3. 🌐 Start the API Server:
       python api/app.py
    
    4. 🔍 Test the System:
       Open http://localhost:5000/health in your browser
    
    📚 Documentation: README.md
    🆘 Troubleshooting: README.md#troubleshooting
    
    Happy coding! 🤟
    """
    print(next_steps)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Sign Language Recognition System")
    parser.add_argument("--skip-install", action="store_true", 
                       help="Skip pip install (for development)")
    parser.add_argument("--no-copy", action="store_true",
                       help="Don't copy from old backend")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Copy from old backend
    if not args.no_copy:
        copy_videos_from_old_backend()
        copy_label_map()
    
    # Step 4: Install dependencies
    if not args.skip_install:
        install_requirements()
    
    # Step 5: Create configuration
    create_env_file()
    
    # Step 6: Validate setup
    validate_setup()
    
    # Step 7: Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
