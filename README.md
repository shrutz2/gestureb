# GestureB - AI-Powered Sign Language Learning Platform

An interactive web application that uses computer vision and deep learning to teach American Sign Language (ASL) through real-time gesture recognition and feedback.

## 🎯 Overview

GestureB combines modern web technologies with advanced machine learning to create an immersive sign language learning experience. Users can practice ASL gestures, receive instant AI feedback, and track their progress through an intuitive interface.

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