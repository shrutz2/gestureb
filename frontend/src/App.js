// App.js - COMPLETE WORKING VERSION
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const BACKEND_URL = 'http://localhost:5000';
const API = `${BACKEND_URL}/api`;

// Landing Page Component
const LandingPage = ({ onGetStarted }) => {
  const [showContent, setShowContent] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);

  useEffect(() => {
    const contentTimer = setTimeout(() => setShowContent(true), 1000);
    return () => clearTimeout(contentTimer);
  }, []);

  const handleLogoClick = () => {
    const logo = document.querySelector('.logo-icon');
    if (logo) {
      logo.style.animation = 'logoSpin 500ms cubic-bezier(1, 0, 0, 1)';
      setTimeout(() => logo.style.animation = '', 500);
    }
  };

  return (
    <div className="landing-page">
      <div className="content">
        <div className="logo" onClick={handleLogoClick}>
          <div className="logo-icon"></div>
          <div className="logo-shadow"></div>
        </div>

        {showContent && (
          <>
            <div className="heading">GestureB</div>
            <div className="subheading">Learn Sign Language with AI</div>
            <div className="description">
              <div className="description-text">
                Master sign language through interactive video lessons, real-time AI feedback,
                and personalized practice sessions.
              </div>
            </div>

            <div className="action-buttons">
              <button className="action-primary" onClick={onGetStarted}>Get Started</button>
            </div>
          </>
        )}

        {showWelcome && (
          <div className="welcome-sticker" onClick={() => setShowWelcome(false)}>
            <div className="sticker-bubble">
              Welcome to GestureB! 👋
            </div>
            <img src="/result.png" alt="Welcome" className="sticker-img" />
          </div>
        )}
      </div>
    </div>
  );
};

// Search Page Component
const SearchPage = ({ onSearch }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  const popularWords = ['hello', 'love', 'help', 'please', 'sorry', 'good', 'bad', 'yes', 'no'];

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    setIsSearching(true);
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() })
      });
      const data = await res.json();
      if (data.found) {
        onSearch(query.trim());
      } else {
        alert(data.message || 'Word not found. Try: ' + (data.results?.join(', ') || 'different words'));
      }
    } catch (e) {
      console.error('Search error:', e);
      alert('Search failed. Check if backend is running on http://localhost:5000');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="search-page">
      <div className="search-content">
        <div className="search-header">
          <div className="brand-logo">
            <span className="logo-mini">G</span>
            <span className="brand-name">GestureB</span>
          </div>
        </div>

        <div className="search-main">
          <h1 className="search-title">What would you like to learn?</h1>
          <p className="search-subtitle">Search for any sign language word</p>
          
          <form onSubmit={(e) => { e.preventDefault(); handleSearch(searchTerm); }} className="search-form">
            <div className="search-input-container">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Type a word..."
                className="search-input"
                disabled={isSearching}
              />
              <button type="submit" className="search-submit" disabled={isSearching || !searchTerm.trim()}>
                {isSearching ? 'Searching...' : 'Search'}
              </button>
            </div>
          </form>

          <div className="popular-section">
            <h3>Popular Words</h3>
            <div className="word-chips">
              {popularWords.map(w => (
                <button key={w} onClick={() => handleSearch(w)} className="word-chip" disabled={isSearching}>{w}</button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// PracticePage Component - ENHANCED VERSION with longer recording
const PracticePage = ({ word, onBack }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [points, setPoints] = useState(0);
  const [totalPoints, setTotalPoints] = useState(0);
  const [countdown, setCountdown] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [sessionStats, setSessionStats] = useState({ attempts: 0, correct: 0 });
  const [recordingProgress, setRecordingProgress] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recordingIntervalRef = useRef(null);

  // ENHANCED RECORDING CONFIGURATION
  const RECORDING_CONFIG = {
    COUNTDOWN_TIME: 3,        // 3 second countdown
    RECORDING_TIME: 4,        // 4 seconds recording (longer!)
    FRAME_INTERVAL: 120,      // 120ms between frames (slower)
    MIN_FRAMES: 25,           // More frames needed
    MAX_FRAMES: 35,           // More frames captured
    SEQUENCE_MODE: true,      // Use sequence-based recognition
    SEQUENCE_LENGTH: 16,      // Number of frames for sequence model
    CONTINUOUS_RECORDING: true // Enable continuous recording for sequences
  };

  useEffect(() => {
    startWebcam();
    return () => cleanup();
  }, []);

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user',
          frameRate: { ideal: 30 }  // Higher frame rate for smoother capture
        } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraReady(true);
        console.log('✅ Camera ready with enhanced settings');
      }
    } catch (error) {
      console.error('❌ Camera error:', error);
      alert('Camera access needed for sign practice. Please allow camera permission and refresh.');
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      canvas.width = 640;
      canvas.height = 480;
      
      // Enhanced frame capture with better quality
      ctx.drawImage(video, 0, 0, 640, 480);
      
      // Use higher quality JPEG
      return canvas.toDataURL('image/jpeg', 0.9);
    } catch (error) {
      console.error('Frame capture failed:', error);
      return null;
    }
  };

  const startRecording = async () => {
    if (!cameraReady) {
      alert('Camera not ready. Please wait...');
      return;
    }

    setIsRecording(true);
    setFeedback(null);
    setRecordingProgress(0);
    setCountdown(RECORDING_CONFIG.COUNTDOWN_TIME);
    setSessionStats(prev => ({ ...prev, attempts: prev.attempts + 1 }));

    console.log(` Starting ${RECORDING_CONFIG.SEQUENCE_MODE ? 'sequence' : 'static'} recording for: ${word}`);

    // Countdown phase
    const countInterval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countInterval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    // Start actual recording after countdown
    setTimeout(() => {
      const frames = [];
      let frameCount = 0;
      const totalFrames = Math.floor(RECORDING_CONFIG.RECORDING_TIME * 1000 / RECORDING_CONFIG.FRAME_INTERVAL);

      console.log(` Recording ${totalFrames} frames over ${RECORDING_CONFIG.RECORDING_TIME} seconds`);

      // For sequence mode with continuous recording, we use a sliding window approach
      if (RECORDING_CONFIG.SEQUENCE_MODE && RECORDING_CONFIG.CONTINUOUS_RECORDING) {
        console.log(` Using continuous sequence recording mode`);
        
        recordingIntervalRef.current = setInterval(() => {
          const frame = captureFrame();
          if (frame) {
            frames.push(frame);
            frameCount++;
            
            // Update progress
            const progress = (frameCount / totalFrames) * 100;
            setRecordingProgress(Math.min(progress, 100));
            
            // For continuous mode, we analyze once we have enough frames for a sequence
            // but we keep recording to capture the full gesture
            if (frames.length >= RECORDING_CONFIG.SEQUENCE_LENGTH && 
                (frameCount % Math.floor(RECORDING_CONFIG.SEQUENCE_LENGTH / 2) === 0)) {
              // Get the most recent frames for the sequence
              const sequenceFrames = frames.slice(-RECORDING_CONFIG.SEQUENCE_LENGTH);
              console.log(` Analyzing sequence of ${sequenceFrames.length} frames (continuous mode)`);
              
              // Analyze without stopping recording
              analyzeGesture(sequenceFrames, false);
            }
            
            // Stop recording after reaching total frames
            if (frameCount >= totalFrames || frameCount >= RECORDING_CONFIG.MAX_FRAMES) {
              clearInterval(recordingIntervalRef.current);
              setIsRecording(false);
              setRecordingProgress(100);
              
              console.log(` Completed recording with ${frames.length} total frames`);
              
              // Final analysis with the most recent sequence
              const finalSequence = frames.slice(-RECORDING_CONFIG.SEQUENCE_LENGTH);
              analyzeGesture(finalSequence, true);
            }
          }
        }, RECORDING_CONFIG.FRAME_INTERVAL);
      } else {
        // Original static frame recording mode
        recordingIntervalRef.current = setInterval(() => {
          const frame = captureFrame();
          if (frame) {
            frames.push(frame);
            frameCount++;
            
            // Update progress
            const progress = (frameCount / totalFrames) * 100;
            setRecordingProgress(progress);
            
            if (frameCount >= totalFrames || frameCount >= RECORDING_CONFIG.MAX_FRAMES) {
              clearInterval(recordingIntervalRef.current);
              setIsRecording(false);
              setRecordingProgress(100);
              
              console.log(` Captured ${frames.length} frames for analysis`);
              analyzeGesture(frames, true);
            }
          }
        }, RECORDING_CONFIG.FRAME_INTERVAL);
      }
    }, RECORDING_CONFIG.COUNTDOWN_TIME * 1000);
  };

  const analyzeGesture = async (frames, isFinalAnalysis = true) => {
    console.log(` Analyzing ${frames.length} frames for: ${word} (${isFinalAnalysis ? 'final' : 'continuous'} analysis)`);
    
    // Only set analyzing state for final analysis to avoid UI flicker during continuous mode
    if (isFinalAnalysis) {
      setIsAnalyzing(true);
    }
    
    try {
      // Enhanced validation
      const validFrames = frames.filter(f => f && f.length > 1000); // Higher quality threshold
      
      // For continuous mode, we might have fewer frames during initial analysis
      const minFramesRequired = RECORDING_CONFIG.SEQUENCE_MODE ? 
        Math.min(RECORDING_CONFIG.SEQUENCE_LENGTH, RECORDING_CONFIG.MIN_FRAMES) : 
        RECORDING_CONFIG.MIN_FRAMES;
      
      if (validFrames.length < minFramesRequired) {
        if (isFinalAnalysis) {
          setFeedback({
            is_correct: false,
            message: `⚠️ Need more clear frames (got ${validFrames.length}, need ${minFramesRequired}). Try with better lighting!`,
            confidence: 0,
            points: 0,
            predicted_word: '',
            improvement_tips: [
              "Ensure bright, even lighting",
              "Keep hands clearly visible",
              "Avoid shadows on your hands",
              "Use a plain background"
            ]
          });
        }
        if (isFinalAnalysis) setIsAnalyzing(false);
        return;
      }
      
      console.log(`Sending ${validFrames.length} high-quality frames to backend`);
      
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          frames: validFrames,
          target_word: word,
          visualize_detection: true,
          use_sequence_model: RECORDING_CONFIG.SEQUENCE_MODE
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(' API Error:', response.status, errorText);
        
        if (isFinalAnalysis) {
          setFeedback({
            is_correct: false,
            message: `Server error (${response.status}). Please try again!`,
            confidence: 0,
            points: 0,
            predicted_word: ''
          });
        }
        return;
      }
      
      const result = await response.json();
      console.log(` ${RECORDING_CONFIG.SEQUENCE_MODE ? 'Sequence' : 'Static'} prediction result:`, result);
      
      // Only update UI for final analysis or if we got a correct result during continuous mode
      if (isFinalAnalysis || (result.is_correct && result.confidence > 0.6)) {
        setFeedback(result);
        
        // Update points and stats
        if (result.is_correct) {
          setPoints(prev => prev + (result.points || 10));
          setTotalPoints(prev => prev + (result.points || 10));
          
          // Only update stats for final analysis to avoid double-counting
          if (isFinalAnalysis) {
            setSessionStats(prev => ({ ...prev, correct: prev.correct + 1 }));
          } else {
            // For continuous mode with correct result, stop recording
            if (recordingIntervalRef.current) {
              clearInterval(recordingIntervalRef.current);
              setIsRecording(false);
              setRecordingProgress(100);
              console.log('🎯 Early success! Stopping recording due to correct prediction');
            }
          }
        }
      }
      
    } catch (error) {
      console.error('❌ Network error:', error);
      if (isFinalAnalysis) {
        setFeedback({
          is_correct: false,
          message: 'Network error. Please check if backend is running on http://localhost:5000',
          confidence: 0,
          points: 0,
          predicted_word: ''
        });
      }
    } finally {
      if (isFinalAnalysis) {
        setIsAnalyzing(false);
      }
    }
  };

  const resetPractice = () => {
    setFeedback(null);
    setRecordingProgress(0);
  };

  const getSuccessRate = () => {
    if (sessionStats.attempts === 0) return 0;
    return Math.round((sessionStats.correct / sessionStats.attempts) * 100);
  };

  return (
    <div className="practice-page">
      <div className="practice-header">
        <button onClick={onBack} className="back-button">
          ← Back
        </button>
        <div className="practice-title">
          <h1>Practice: "{word}"</h1>
          <div className="points-display">
            <span>Session: {points} pts</span>
            <span>Total: {totalPoints} pts</span>
            <span>Success: {getSuccessRate()}%</span>
          </div>
        </div>
      </div>

      <div className="practice-container">
        <div className="video-panel">
          <div className="panel-header">
            <h2>Reference Video</h2>
            <p>Watch carefully and copy the gesture</p>
          </div>
          <video 
            controls
            className="reference-video"
            src={`${BACKEND_URL}/videos/${word.toLowerCase()}.mp4`}
            onError={() => setVideoError(true)}
            onLoadedData={() => setVideoError(false)}
          >
            Your browser doesn't support video.
          </video>
          {videoError && (
            <div style={{padding: '20px', textAlign: 'center', color: '#666'}}>
              <p> Video not available for "{word}"</p>
              <p>Practice based on your knowledge of the sign!</p>
            </div>
          )}
        </div>

        <div className="camera-panel">
          <div className="panel-header">
            <h2>Your Practice</h2>
            <p>Sign clearly for {RECORDING_CONFIG.RECORDING_TIME} seconds</p>
          </div>
          <div className="camera-wrapper">
            <video ref={videoRef} autoPlay muted playsInline className="practice-camera" />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            
            {countdown > 0 && (
              <div className="countdown-overlay">
                <div className="countdown-circle">
                  <span className="countdown-number">{countdown}</span>
                </div>
                <div className="countdown-text">Get Ready!</div>
              </div>
            )}

            {isRecording && countdown === 0 && (
              <div className="recording-overlay">
                <div className="recording-indicator">
                  <div className="rec-dot"></div>
                  <span>Recording... ({RECORDING_CONFIG.RECORDING_TIME}s)</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${recordingProgress}%` }}
                  ></div>
                </div>
                <div className="recording-tip">
                  Hold the sign steady and clear!
                </div>
              </div>
            )}

            <div className="camera-controls">
              {!isRecording && !isAnalyzing && (
                <button onClick={startRecording} className="record-button enhanced" disabled={!cameraReady}>
                  {feedback ? 'Practice Again' : `Start ${RECORDING_CONFIG.RECORDING_TIME}s Practice`}
                </button>
              )}
              
              {isAnalyzing && (
                <div style={{textAlign: 'center', padding: '20px'}}>
                  <div className="spinner enhanced"></div>
                  <p>Analyzing your sign...</p>
                  <p className="analyze-tip">Using enhanced AI recognition</p>
                </div>
              )}
            </div>

            {feedback && (
              <div className={`feedback-panel ${feedback.is_correct ? 'success' : 'retry'} enhanced`}>
                <div className="feedback-icon">
                  {feedback.is_correct ? '🎉' : ''}
                </div>
                <div className="feedback-message">{feedback.message}</div>
                
                {feedback.visualization_url && (
                  <div className="hand-detection-visualization">
                    <h4>Hand Detection</h4>
                    <img 
                      src={`${BACKEND_URL}${feedback.visualization_url}`} 
                      alt="Hand detection visualization" 
                      className="detection-image"
                    />
                    <p className="detection-note">
                      {feedback.hands_detected_count > 0 ? ' Hand detected' : ' No hand detected'}
                    </p>
                  </div>
                )}
                
                <div className="feedback-stats">
                  <p><strong>Confidence:</strong> {Math.round(feedback.confidence * 100)}%</p>
                  {feedback.predicted_word && feedback.predicted_word !== word && (
                    <div className="confusion-alert">
                      <p><strong>AI Detected:</strong> {feedback.predicted_word}</p>
                      <p><strong>You were practicing:</strong> {word}</p>
                    </div>
                  )}
                  {feedback.is_correct && feedback.points > 0 && (
                    <div className="points-earned">+{feedback.points} Points! </div>
                  )}
                  {feedback.top_predictions && feedback.top_predictions.length > 1 && (
                    <div className="top-predictions">
                      <p><strong>Top predictions:</strong></p>
                      <ul>
                        {feedback.top_predictions.slice(0, 3).map((pred, i) => (
                          <li key={i}>{pred.word} ({pred.percentage})</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <button onClick={resetPractice} className="try-again-button">
                  {feedback.is_correct ? ' Practice More' : 'Try Again'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="tips-section">
        <h3> Quick Tips for Better Recognition</h3>
        <div className="tips-grid">
          <div className="tip-item"><span></span><span>Keep camera steady</span></div>
          <div className="tip-item"><span></span><span>Ensure bright, even lighting</span></div>
          <div className="tip-item"><span></span><span>Center your hands in the frame</span></div>
          <div className="tip-item"><span></span><span>Move slowly and deliberately</span></div>
          <div className="tip-item"><span></span><span>Watch the reference video first</span></div>
          <div className="tip-item"><span></span><span>Use a plain background</span></div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [currentPage, setCurrentPage] = useState('landing');
  const [currentWord, setCurrentWord] = useState('');

  const handleGetStarted = () => setCurrentPage('search');
  const handleSearch = (word) => {
    setCurrentWord(word);
    setCurrentPage('practice');
  };
  const handleBackToSearch = () => {
    setCurrentPage('search');
    setCurrentWord('');
  };

  return (
    <div className="App">
      {currentPage === 'landing' && <LandingPage onGetStarted={handleGetStarted} />}
      {currentPage === 'search' && <SearchPage onSearch={handleSearch} />}
      {currentPage === 'practice' && <PracticePage word={currentWord} onBack={handleBackToSearch} />}
    </div>
  );
}

export default App;
