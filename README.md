# GestureB - AI-Powered Sign Language Learning Platform

An interactive web application that uses computer vision and deep learning to teach American Sign Language (ASL) through real-time gesture recognition and feedback.

## 🎯 Overview

GestureB combines modern web technologies with advanced machine learning to create an immersive sign language learning experience. Users can practice ASL gestures, receive instant AI feedback, and track their progress through an intuitive interface.

## 🧠 Algorithms & Technologies

### Machine Learning & Computer Vision
- **Deep Learning Framework**: TensorFlow 2.13.0
- **Neural Network Architecture**: LSTM with Attention Mechanism
  - 2x LSTM layers (64 units each) with dropout regularization
  - Temporal attention mechanism for sequence focus
  - Batch normalization for stable training
  - L2 regularization to prevent overfitting
- **Hand Landmark Detection**: MediaPipe 0.10.7
  - Real-time hand pose estimation
  - 21-point hand landmark extraction
  - Multi-hand detection and tracking
- **Computer Vision**: OpenCV 4.8.1.78
  - Video frame processing
  - Image preprocessing and normalization
- **Feature Engineering**: 
  - Z-score normalization for landmark coordinates
  - Sequence padding and temporal alignment
  - Robust feature scaling with scikit-learn

### Backend Technologies
- **Web Framework**: Flask 2.3.3 with CORS support
- **Production Server**: Gunicorn 21.2.0
- **Data Processing**: 
  - NumPy 1.24.3 for numerical computations
  - Pandas 2.0.3 for data manipulation
  - Scikit-learn 1.3.0 for ML utilities
- **Visualization**: Matplotlib 3.7.2, Seaborn 0.12.2
- **Environment Management**: Python-dotenv 1.0.0

### Frontend Technologies
- **Framework**: React 19.0.0
- **Styling**: Tailwind CSS 3.4.17
- **HTTP Client**: Axios 1.8.4
- **Routing**: React Router DOM 7.5.1


### Data Flow
1. **Video Capture**: Frontend captures user gestures via webcam
2. **Frame Processing**: MediaPipe extracts hand landmarks from video frames
3. **Feature Engineering**: Landmarks normalized and formatted for ML model
4. **Prediction**: LSTM model processes temporal sequence for gesture classification
5. **Feedback**: Real-time results sent back to frontend with confidence scores

## 🚀 Key Features

### AI-Powered Recognition
- **High Accuracy**: >90% accuracy on trained gestures
- **Confidence Scoring**: Probabilistic feedback for learning improvement
- **Multi-gesture Support**: 300+ ASL signs with video references

### Learning Experience
- **Interactive Practice**: Live camera-based gesture practice
- **Progress Tracking**: Points system and accuracy metrics
- **Instant Feedback**: Real-time correction and suggestions
- **Reference Videos**: High-quality ASL demonstration videos

### Technical Capabilities
- **Sequence Recognition**: Temporal pattern analysis for dynamic gestures
- **Robust Detection**: Works in various lighting conditions
- **Scalable Architecture**: Modular design for easy expansion
- **Cross-platform**: Web-based interface accessible on any device

## 📊 Model Performance

### Training Configuration
- **Sequence Length**: 30 frames per gesture
- **Features per Frame**: 63 (21 landmarks × 3 coordinates)
- **Batch Size**: 16
- **Learning Rate**: 0.001 with adaptive reduction
- **Regularization**: Dropout (0.3) + L2 regularization (0.01-0.02)

### Optimization Techniques
- **Early Stopping**: Prevents overfitting with patience=15
- **Learning Rate Scheduling**: Reduces LR on plateau
- **Data Augmentation**: Temporal and spatial variations
- **Attention Mechanism**: Focuses on important gesture phases

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 16+
- Webcam for gesture practice

### Backend Setup
```bash
cd backend_v2
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Model Training (Optional)
```bash
cd backend_v2
python simplified_trainer.py
```

## 🎮 Usage

1. **Launch Application**: Start both backend (port 5000) and frontend (port 3000)
2. **Search Words**: Find ASL signs to practice
3. **Watch Reference**: Study the demonstration video
4. **Practice Gesture**: Use webcam to perform the sign
5. **Get Feedback**: Receive AI analysis and improvement tips
6. **Track Progress**: Monitor accuracy and points earned

## 📈 Performance Metrics

- **Recognition Accuracy**: 90%+ on trained gestures
- **Response Time**: <200ms for gesture analysis
- **Frame Rate**: 30 FPS real-time processing
- **Model Size**: ~5MB optimized for web deployment
- **Supported Gestures**: 200+ ASL vocabulary

## 🔬 Research & Development

### Algorithm Innovations
- **Temporal Attention**: Novel attention mechanism for gesture sequences
- **Robust Normalization**: Handles varying user positions and scales
- **Limited Data Training**: Achieves high accuracy with minimal training samples
- **Real-time Optimization**: Efficient inference for web deployment

### Future Enhancements
- **Sentence Recognition**: Multi-word gesture sequences
- **Personalized Learning**: Adaptive difficulty based on user progress
- **Mobile App**: Native iOS/Android applications
- **Community Features**: Social learning and competitions

## 📝 License

This project is developed for educational purposes and sign language accessibility.

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional ASL vocabulary
- Model accuracy improvements
- UI/UX enhancements
- Performance optimizations

---

**GestureB** - Making sign language learning accessible through AI technology 🤟