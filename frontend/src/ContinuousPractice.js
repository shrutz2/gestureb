// ContinuousPractice.js - No time limit, automatic detection
import React, { useState, useEffect, useRef } from 'react';

const BACKEND_URL = 'http://localhost:5000';
const API = `${BACKEND_URL}/api`;

const ContinuousPractice = ({ word, onBack }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [handsDetected, setHandsDetected] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const [detectionCount, setDetectionCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  useEffect(() => {
    startWebcam();
    return () => cleanup();
  }, []);

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraReady(true);
      }
    } catch (error) {
      console.error('Webcam error:', error);
      alert('Camera access needed');
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
  };

  const startContinuousDetection = () => {
    if (!cameraReady) return;

    setIsDetecting(true);
    setFinalResult(null);
    setDetectionCount(0);
    setCorrectCount(0);

    // Check every 500ms for hands and predictions
    detectionIntervalRef.current = setInterval(async () => {
      const frame = captureFrame();
      if (!frame) return;

      try {
        const response = await fetch(`${API}/predict-continuous`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            frame: frame,
            target_word: word
          })
        });

        if (response.ok) {
          const result = await response.json();
          
          setHandsDetected(result.hands_detected);
          
          if (result.hands_detected && result.prediction) {
            setCurrentPrediction(result.prediction);
            setDetectionCount(prev => prev + 1);
            
            if (result.prediction.is_correct) {
              setCorrectCount(prev => prev + 1);
            }
            
            // Auto-complete after 5 correct detections
            if (result.prediction.is_correct) {
              const newCorrectCount = correctCount + 1;
              if (newCorrectCount >= 3) {
                stopDetection();
                setFinalResult({
                  is_correct: true,
                  predicted_word: result.prediction.word,
                  confidence: result.prediction.confidence,
                  message: `🎉 Perfect! Detected "${word}" correctly ${newCorrectCount} times!`,
                  points: 20 + newCorrectCount * 5
                });
              }
            }
          } else {
            setCurrentPrediction(null);
          }
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
    }, 500); // Check every 500ms
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
  };

  const handleTryAgain = () => {
    setFinalResult(null);
    setCurrentPrediction(null);
    setDetectionCount(0);
    setCorrectCount(0);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <button onClick={onBack} style={{ padding: '10px 20px', background: '#6b7280', color: 'white', border: 'none', borderRadius: '5px' }}>
          ← Back
        </button>
        <h1>Practice: "{word}"</h1>
        <div></div>
      </div>

      <div style={{ display: 'flex', gap: '20px' }}>
        {/* Left side - Video */}
        <div style={{ flex: 1 }}>
          <h3>Reference Video</h3>
          <div style={{ background: '#f3f4f6', padding: '20px', borderRadius: '10px', textAlign: 'center' }}>
            <video 
              width="100%" 
              height="300" 
              controls
              style={{ borderRadius: '10px' }}
            >
              <source src={`${BACKEND_URL}/videos/${word}.mp4`} type="video/mp4" />
              Video not available
            </video>
          </div>
        </div>

        {/* Right side - Camera */}
        <div style={{ flex: 1 }}>
          <h3>Your Practice</h3>
          <div style={{ position: 'relative', background: '#000', borderRadius: '10px', overflow: 'hidden' }}>
            <video 
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ width: '100%', height: '300px', objectFit: 'cover' }}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            
            {/* Status overlay */}
            <div style={{ 
              position: 'absolute', 
              top: '10px', 
              left: '10px', 
              background: 'rgba(0,0,0,0.7)', 
              color: 'white', 
              padding: '10px', 
              borderRadius: '5px',
              fontSize: '14px'
            }}>
              <div>Hands: {handsDetected ? '✅ Detected' : '❌ Not detected'}</div>
              <div>Detections: {detectionCount}</div>
              <div>Correct: {correctCount}</div>
            </div>

            {/* Current prediction */}
            {currentPrediction && (
              <div style={{ 
                position: 'absolute', 
                bottom: '10px', 
                left: '10px', 
                right: '10px',
                background: currentPrediction.is_correct ? 'rgba(34, 197, 94, 0.9)' : 'rgba(239, 68, 68, 0.9)',
                color: 'white', 
                padding: '10px', 
                borderRadius: '5px',
                textAlign: 'center'
              }}>
                <div style={{ fontWeight: 'bold' }}>
                  {currentPrediction.is_correct ? '✅' : '❌'} {currentPrediction.word}
                </div>
                <div style={{ fontSize: '12px' }}>
                  Confidence: {Math.round(currentPrediction.confidence * 100)}%
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ marginTop: '20px', textAlign: 'center' }}>
            {!isDetecting && !finalResult && (
              <button 
                onClick={startContinuousDetection}
                disabled={!cameraReady}
                style={{ 
                  padding: '15px 30px', 
                  background: '#3b82f6', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '10px',
                  fontSize: '16px',
                  cursor: 'pointer'
                }}
              >
                Start Practicing "{word}"
              </button>
            )}

            {isDetecting && (
              <button 
                onClick={stopDetection}
                style={{ 
                  padding: '15px 30px', 
                  background: '#ef4444', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '10px',
                  fontSize: '16px',
                  cursor: 'pointer'
                }}
              >
                Stop Detection
              </button>
            )}

            {finalResult && (
              <div style={{ 
                background: finalResult.is_correct ? '#dcfce7' : '#fef2f2',
                border: `2px solid ${finalResult.is_correct ? '#22c55e' : '#ef4444'}`,
                borderRadius: '10px',
                padding: '20px',
                marginTop: '20px'
              }}>
                <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px' }}>
                  {finalResult.message}
                </div>
                <div style={{ marginBottom: '15px' }}>
                  <div>Predicted: {finalResult.predicted_word}</div>
                  <div>Confidence: {Math.round(finalResult.confidence * 100)}%</div>
                  <div>Points: {finalResult.points}</div>
                </div>
                <button 
                  onClick={handleTryAgain}
                  style={{ 
                    padding: '10px 20px', 
                    background: '#3b82f6', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '5px',
                    marginRight: '10px'
                  }}
                >
                  Try Again
                </button>
                <button 
                  onClick={onBack}
                  style={{ 
                    padding: '10px 20px', 
                    background: '#6b7280', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '5px'
                  }}
                >
                  Choose Another Word
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div style={{ marginTop: '30px', background: '#f9fafb', padding: '20px', borderRadius: '10px' }}>
        <h3>Instructions:</h3>
        <ul style={{ margin: '10px 0', paddingLeft: '20px' }}>
          <li>📹 Watch the reference video on the left</li>
          <li>✋ Position your hands clearly in the camera view</li>
          <li>🎯 Perform the sign language gesture for "{word}"</li>
          <li>⏱️ No time limit - practice until you get it right!</li>
          <li>✅ Get 3 correct detections to complete</li>
        </ul>
      </div>
    </div>
  );
};

export default ContinuousPractice;