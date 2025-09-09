// SIMPLE VERSION - Exactly like friend's main.py logic
import React, { useState, useEffect, useRef } from 'react';

const SimplePracticePage = ({ word, onBack }) => {
  // Friend's exact variables from main.py
  const [sentence, setSentence] = useState([]);
  const [keypoints, setKeypoints] = useState([]);
  const [lastPrediction, setLastPrediction] = useState('');
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    startCamera();
    return cleanup;
  }, []);

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        startProcessing();
      }
    } catch (error) {
      console.error('Camera error:', error);
    }
  };

  // Friend's exact processing loop
  const startProcessing = () => {
    intervalRef.current = setInterval(async () => {
      if (!videoRef.current) return;

      // Capture frame like friend's cv2.read()
      const frame = captureFrame();
      if (!frame) return;

      // Mock hand detection and landmarks (replace with real detection later)
      const landmarks = generateMockLandmarks();
      drawLandmarks(landmarks);
      
      // Add to keypoints like friend's main.py
      setKeypoints(prev => {
        const newKeypoints = [...prev, landmarks];
        
        // Friend's exact logic: if 10 frames collected
        if (newKeypoints.length === 10) {
          makePrediction(newKeypoints);
          return []; // Clear keypoints
        }
        
        return newKeypoints;
      });

    }, 100); // Friend's processing interval
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = 640;
    canvas.height = 480;
    ctx.drawImage(video, 0, 0, 640, 480);
    
    return canvas.toDataURL('image/jpeg', 0.8);
  };

  const generateMockLandmarks = () => {
    // Simple mock landmarks for testing
    const landmarks = [];
    for (let i = 0; i < 21; i++) {
      landmarks.push({
        x: 0.3 + Math.random() * 0.4,
        y: 0.3 + Math.random() * 0.4,
        z: 0
      });
    }
    return landmarks;
  };

  const drawLandmarks = (landmarks) => {
    const canvas = document.getElementById('overlay-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw red dots like friend's project
    landmarks.forEach(landmark => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      
      ctx.fillStyle = '#FF4444';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  };

  // Friend's exact prediction logic
  const makePrediction = async (keypointsData) => {
    try {
      // Mock prediction for now
      const words = ['hello', 'thanks', 'please', 'sorry', 'yes', 'no'];
      const prediction = words[Math.floor(Math.random() * words.length)];
      const confidence = Math.random() * 0.5 + 0.5;

      console.log('Prediction:', prediction, 'Confidence:', confidence);

      // Friend's exact logic: if confidence > 0.9
      if (confidence > 0.9) {
        // Check if different from last prediction
        if (lastPrediction !== prediction) {
          // Add to sentence
          setSentence(prev => {
            const newSentence = [...prev, prediction];
            
            // Limit to 7 words like friend's
            if (newSentence.length > 7) {
              return newSentence.slice(-7);
            }
            return newSentence;
          });
          
          setLastPrediction(prediction);
        }
      }
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  // Reset like friend's spacebar
  const resetSentence = () => {
    setSentence([]);
    setKeypoints([]);
    setLastPrediction('');
  };

  // Keyboard controls like friend's
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.code === 'Space') {
        event.preventDefault();
        resetSentence();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Format sentence like friend's
  const displaySentence = () => {
    if (sentence.length === 0) return '';
    
    let formatted = [...sentence];
    if (formatted.length > 0) {
      formatted[0] = formatted[0].charAt(0).toUpperCase() + formatted[0].slice(1);
    }
    
    return formatted.join(' ');
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: '#000', 
      color: 'white', 
      position: 'relative' 
    }}>
      {/* Back button */}
      <button 
        onClick={onBack}
        style={{ 
          position: 'absolute', 
          top: '20px', 
          left: '20px', 
          zIndex: 10,
          padding: '10px 20px',
          background: 'rgba(255,255,255,0.2)',
          color: 'white',
          border: '1px solid white',
          borderRadius: '8px',
          cursor: 'pointer'
        }}
      >
        Back
      </button>

      {/* Main video area */}
      <div style={{ 
        position: 'relative', 
        width: '100vw', 
        height: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        {/* Video */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scaleX(-1)'
          }}
        />
        
        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        
        {/* Overlay for landmarks */}
        <canvas
          id="overlay-canvas"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            transform: 'scaleX(-1)'
          }}
        />
        
        {/* Sentence display like friend's */}
        {displaySentence() && (
          <div style={{
            position: 'absolute',
            bottom: '50px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0,0,0,0.7)',
            padding: '20px 40px',
            borderRadius: '10px',
            fontSize: '32px',
            fontWeight: 'bold',
            color: 'white',
            textAlign: 'center'
          }}>
            {displaySentence()}
          </div>
        )}
        
        {/* Instructions */}
        <div style={{
          position: 'absolute',
          top: '80px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(0,0,0,0.5)',
          padding: '10px 20px',
          borderRadius: '8px',
          fontSize: '14px',
          textAlign: 'center'
        }}>
          Practicing "{word}" • Press SPACE to reset
        </div>

        {/* Debug info */}
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          background: 'rgba(0,0,0,0.5)',
          padding: '10px',
          borderRadius: '8px',
          fontSize: '12px'
        }}>
          Keypoints: {keypoints.length}/10<br/>
          Last: {lastPrediction}<br/>
          Words: {sentence.length}
        </div>
      </div>
    </div>
  );
};

export default SimplePracticePage;