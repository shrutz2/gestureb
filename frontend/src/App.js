// SIMPLE App.js - Backend Does All Landmark Work
// No MediaPipe CDN issues, backend handles everything

import React, { useState, useEffect, useRef, createContext, useContext } from 'react';
import './App.css';

const BACKEND_URL = 'http://localhost:5000';
const API = `${BACKEND_URL}/api`;

// Auth Context (keeping same)
const AuthContext = createContext();
const useAuth = () => useContext(AuthContext);

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(false);

  const login = async (email, password) => {
    try {
      const response = await fetch(`${API}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await response.json();
      if (response.ok) {
        setToken(data.token);
        setUser(data.user);
        return { success: true };
      }
      return { success: false, error: data.error };
    } catch (error) {
      return { success: false, error: 'Network error' };
    }
  };

  const register = async (username, email, password) => {
    try {
      const response = await fetch(`${API}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
      });
      const data = await response.json();
      if (response.ok) {
        setToken(data.token);
        setUser(data.user);
        return { success: true };
      }
      return { success: false, error: data.error };
    } catch (error) {
      return { success: false, error: 'Network error' };
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
  };

  const refreshUserStats = async () => {};

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout, refreshUserStats, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
};

// Auth Page (keeping same)
const AuthPage = ({ onSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      let result;
      if (isLogin) {
        result = await login(formData.email, formData.password);
      } else {
        result = await register(formData.username, formData.email, formData.password);
      }

      if (result.success) {
        onSuccess();
      } else {
        setError(result.error);
      }
    } catch (error) {
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>🤖</div>
          <h1>{isLogin ? 'Welcome Back' : 'Join GestureB'}</h1>
          <p>{isLogin ? 'Sign in to continue' : 'Start learning today'}</p>
        </div>

        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <div style={{ marginBottom: '1rem' }}>
              <input type="text" name="username" value={formData.username} onChange={handleChange}
                placeholder="Username" style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '8px', fontSize: '16px' }}
                required minLength={3} />
            </div>
          )}

          <div style={{ marginBottom: '1rem' }}>
            <input type="email" name="email" value={formData.email} onChange={handleChange}
              placeholder="Email" style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '8px', fontSize: '16px' }}
              required />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <input type="password" name="password" value={formData.password} onChange={handleChange}
              placeholder="Password" style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '8px', fontSize: '16px' }}
              required minLength={6} />
          </div>

          {error && (
            <div style={{ color: '#dc3545', marginBottom: '1rem', padding: '8px', background: '#f8d7da', borderRadius: '4px' }}>
              {error}
            </div>
          )}

          <button type="submit" disabled={loading}
            style={{ width: '100%', padding: '12px', background: '#18119e', color: 'white', border: 'none', borderRadius: '8px', fontSize: '16px', fontWeight: '600', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1 }}>
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: '1rem' }}>
          <span>{isLogin ? "Don't have an account?" : "Already have an account?"}</span>
          <button type="button" onClick={() => setIsLogin(!isLogin)}
            style={{ background: 'none', border: 'none', color: '#18119e', cursor: 'pointer', textDecoration: 'underline', marginLeft: '8px' }}>
            {isLogin ? 'Sign Up' : 'Sign In'}
          </button>
        </div>
      </div>
    </div>
  );
};

// Landing Page (keeping same)
const LandingPage = ({ onGetStarted }) => {
  const [showContent, setShowContent] = useState(false);
  const { user, logout } = useAuth();

  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 1000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="landing-page">
      {user && (
        <div style={{ position: 'fixed', top: '1rem', right: '1rem', background: 'white', padding: '1rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', display: 'flex', alignItems: 'center', gap: '1rem', zIndex: 1000 }}>
          <div>
            <div style={{ fontWeight: '600' }}>{user.username}</div>
            <div style={{ fontSize: '0.875rem', color: '#666' }}>Level {user.stats?.level || 1} • {user.stats?.total_points || 0} pts</div>
          </div>
          <button onClick={logout} style={{ background: '#dc3545', color: 'white', border: 'none', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer' }}>Logout</button>
        </div>
      )}

      <div className="content">
        <div className="logo">
          <div className="logo-icon"></div>
          <div className="logo-shadow"></div>
        </div>

        {showContent && (
          <>
            <div className="heading">GestureB</div>
            <div className="subheading">Learn Sign Language with AI</div>
            <div className="description">
              <div className="description-text">Master sign language through interactive lessons and real-time AI feedback.</div>
            </div>
            <div className="action-buttons">
              <button className="action-primary" onClick={onGetStarted}>{user ? 'Continue Learning' : 'Get Started'}</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// FIXED Search Page - Updated to match backend response
const SearchPage = ({ onSearch, onProfile }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const { user, token, refreshUserStats } = useAuth();

  const popularWords = ['hello', 'name', 'love', 'help', 'please', 'sorry', 'good', 'bad', 'yes', 'no'];

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    setIsSearching(true);
    
    console.log(`🔍 Searching for: "${query}"`);
    
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json', 
          'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({ query: query.trim() })
      });
      
      const data = await res.json();
      console.log('🎯 Search response:', data);
      
      // FIXED: Check for 'success' instead of 'found' to match backend response
      if (data.success && data.words && data.words.length > 0) {
        console.log(`✅ Word found: ${query}`);
        onSearch(query.trim());
      } else {
        console.log('❌ Word not found or no words returned');
        alert(data.message || `Word "${query}" not found. Try: ${popularWords.join(', ')}`);
      }
    } catch (e) {
      console.error('❌ Search error:', e);
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
              <input type="text" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Type a word..." className="search-input" disabled={isSearching} />
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

          {/* Add debug info */}
          <div style={{ 
            marginTop: '2rem', 
            padding: '1rem', 
            background: '#f0f8ff', 
            borderRadius: '8px',
            fontSize: '0.875rem',
            color: '#666'
          }}>
            <h4 style={{ color: '#18119e', marginBottom: '0.5rem' }}>Debug Info:</h4>
            <p>Backend URL: {BACKEND_URL}</p>
            <p>Search endpoint: {API}/search</p>
            <p>Current search term: "{searchTerm}"</p>
            <p>Status: {isSearching ? 'Searching...' : 'Ready'}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// SIMPLE PRACTICE PAGE - Backend Does Everything
const PracticePage = ({ word, onBack }) => {
  // Simple state management
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [countdown, setCountdown] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [sessionStats, setSessionStats] = useState({ attempts: 0, correct: 0 });
  const [modelStatus, setModelStatus] = useState('Loading...');
  const [pronunciationFeedback, setPronunciationFeedback] = useState(null);
  const [showPronunciationResult, setShowPronunciationResult] = useState(false);
  const [debugLandmarkCollection, setDebugLandmarkCollection] = useState(null);
  const { user, token, refreshUserStats } = useAuth();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const speechSynthesisRef = useRef(null);

  // Constants
  const RECORDING_TIME = 4;
  const FRAME_RATE = 10;
  const MAX_FRAMES = RECORDING_TIME * FRAME_RATE;

  useEffect(() => {
    startWebcam();
    return () => cleanup();
  }, []);

  useEffect(() => {
    if (cameraReady) {
      checkModelStatus();
      const statusInterval = setInterval(checkModelStatus, 3000);
      return () => clearInterval(statusInterval);
    }
  }, [cameraReady]);

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: 'user' } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setCameraReady(true);
          console.log('✅ Camera ready - Backend will handle landmarks');
        };
      }
    } catch (error) {
      console.error('❌ Camera error:', error);
      alert('Camera access needed. Please allow camera permission.');
    }
  };

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${API}/status`);
      const data = await response.json();
      
      if (data.model_loaded) {
        setModelStatus(`✅ Ready (${data.total_classes} words)`);
      } else {
        setModelStatus('❌ Loading...');
      }
    } catch (error) {
      setModelStatus('❌ Backend offline');
    }
  };

  const startSimpleRecording = async () => {
    console.log('🎬 Starting simple recording - Backend handles landmarks...');
    
    setIsRecording(true);
    setFeedback(null);
    setCountdown(3);
    setSessionStats(prev => ({ ...prev, attempts: prev.attempts + 1 }));

    // Countdown
    const countInterval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countInterval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    // Start frame collection after countdown
    setTimeout(() => {
      collectFramesSimple();
    }, 3000);
  };

  const collectFramesSimple = () => {
    console.log('📸 Collecting frames - Backend will extract landmarks...');
    
    const collectedFrames = [];
    let frameCount = 0;
    
    const frameInterval = setInterval(() => {
      if (frameCount < MAX_FRAMES && videoRef.current && canvasRef.current) {
        // Capture frame
        const canvas = canvasRef.current;
        const video = videoRef.current;
        const ctx = canvas.getContext('2d');
        
        canvas.width = 640;
        canvas.height = 480;
        ctx.drawImage(video, 0, 0, 640, 480);
        
        const frameData = canvas.toDataURL('image/jpeg', 0.9);
        collectedFrames.push(frameData);
        frameCount++;
        
        console.log(`Frame ${frameCount}/${MAX_FRAMES} collected`);
      } else {
        clearInterval(frameInterval);
        setIsRecording(false);
        
        console.log(`✅ Collected ${collectedFrames.length} frames - sending to backend`);
        processWithBackend(collectedFrames);
      }
    }, 1000 / FRAME_RATE);
  };

  const processWithBackend = async (frames) => {
    console.log(`🧠 Backend processing: ${frames.length} frames`);
    setIsAnalyzing(true);
    
    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          target_word: word,
          frames: frames,
          approach: 'backend_landmark_extraction'
        })
      });
      
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('🎯 Backend result:', result);
      
      setFeedback(result);
      
      if (result.is_correct) {
        setSessionStats(prev => ({ ...prev, correct: prev.correct + 1 }));
        if (refreshUserStats) refreshUserStats();
      }
      
    } catch (error) {
      console.error('❌ Backend processing error:', error);
      setFeedback({
        is_correct: false,
        message: `Processing failed: ${error.message}. Check backend connection.`,
        confidence: 0
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // NEW: Debug landmark extraction function
  const debugLandmarks = async (frames) => {
    console.log(`🔍 DEBUG: Analyzing ${frames.length} frames for landmark quality...`);
    
    try {
      const response = await fetch(`${API}/debug_landmarks`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ frames: frames })
      });
      
      if (!response.ok) {
        throw new Error(`Debug error: ${response.status}`);
      }
      
      const debugResult = await response.json();
      console.log('🔍 LANDMARK DEBUG RESULT:', debugResult);
      
      // Show debug info in a more readable format
      const debugInfo = debugResult.debug_info;
      console.log(`
🔍 LANDMARK EXTRACTION ANALYSIS:
👋 Hand Detection: ${debugInfo.hand_detection_stats.frames_with_hands}/${debugInfo.total_frames} frames
🎯 Average Confidence: ${debugInfo.hand_detection_stats.average_confidence.toFixed(3)}
📊 Coordinate Ranges: 
   X: [${debugInfo.coordinate_ranges.x_min.toFixed(3)}, ${debugInfo.coordinate_ranges.x_max.toFixed(3)}]
   Y: [${debugInfo.coordinate_ranges.y_min.toFixed(3)}, ${debugInfo.coordinate_ranges.y_max.toFixed(3)}]
   Z: [${debugInfo.coordinate_ranges.z_min.toFixed(3)}, ${debugInfo.coordinate_ranges.z_max.toFixed(3)}]
📈 Landmark Quality:
   Non-zero frames: ${debugInfo.landmark_quality.non_zero_frames}/${debugInfo.total_frames}
   Average magnitude: ${debugInfo.landmark_quality.average_landmark_magnitude.toFixed(3)}
      `);
      
      alert(`🔍 LANDMARK DEBUG COMPLETE!\nCheck console for detailed analysis.\n\nQuick Summary:\n✅ Hands detected: ${debugInfo.hand_detection_stats.frames_with_hands}/${debugInfo.total_frames} frames\n🎯 Avg confidence: ${(debugInfo.hand_detection_stats.average_confidence * 100).toFixed(1)}%\n📊 Non-zero landmarks: ${debugInfo.landmark_quality.non_zero_frames} frames`);
      
    } catch (error) {
      console.error('❌ Debug error:', error);
      alert(`Debug failed: ${error.message}`);
    }
  };

  // Dictation function to speak the word
  const speakWord = (text) => {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();
      
      // Create a new speech utterance
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9; // Slightly slower rate for clarity
      utterance.pitch = 1;
      utterance.volume = 1;
      
      // Store reference to control speech
      speechSynthesisRef.current = utterance;
      
      // Speak the word
      window.speechSynthesis.speak(utterance);
      
      console.log(`🔊 Speaking: "${text}"`);
      return true;
    } else {
      console.error('Speech synthesis not supported in this browser');
      return false;
    }
  };
  
  // Pronunciation check function
  const checkPronunciation = () => {
    if (!word) return;
    
    setShowPronunciationResult(false);
    setPronunciationFeedback(null);
    
    // First speak the word for reference
    speakWord(word);
    
    // Setup speech recognition
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      
      // Start listening after a short delay to allow the spoken word to finish
      setTimeout(() => {
        try {
          recognition.start();
          console.log('🎤 Listening for pronunciation...');
        } catch (e) {
          console.error('Speech recognition error:', e);
        }
      }, 1500);
      
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript.toLowerCase().trim();
        const confidence = event.results[0][0].confidence;
        
        console.log(`🎤 Heard: "${transcript}" (${Math.round(confidence * 100)}% confidence)`);
        
        // Compare with target word
        const isCorrect = transcript === word.toLowerCase();
        const similarityScore = calculateSimilarity(transcript, word.toLowerCase());
        
        // Prepare feedback
        const feedback = {
          heard: transcript,
          target: word.toLowerCase(),
          isCorrect: isCorrect,
          confidence: confidence,
          similarityScore: similarityScore,
          feedbackMessage: getFeedbackMessage(isCorrect, similarityScore)
        };
        
        setPronunciationFeedback(feedback);
        setShowPronunciationResult(true);
      };
      
      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setPronunciationFeedback({
          error: true,
          message: `Could not recognize speech: ${event.error}`
        });
        setShowPronunciationResult(true);
      };
      
      recognition.onend = () => {
        console.log('Speech recognition ended');
        if (!pronunciationFeedback) {
          setPronunciationFeedback({
            error: true,
            message: 'No speech detected. Please try again.'
          });
          setShowPronunciationResult(true);
        }
      };
    } else {
      console.error('Speech recognition not supported in this browser');
      setPronunciationFeedback({
        error: true,
        message: 'Speech recognition not supported in this browser'
      });
      setShowPronunciationResult(true);
    }
  };
  
  // Calculate similarity between two strings (for pronunciation feedback)
  const calculateSimilarity = (str1, str2) => {
    if (str1 === str2) return 1.0;
    
    // Simple Levenshtein distance implementation
    const len1 = str1.length;
    const len2 = str2.length;
    
    const matrix = Array(len1 + 1).fill().map(() => Array(len2 + 1).fill(0));
    
    for (let i = 0; i <= len1; i++) matrix[i][0] = i;
    for (let j = 0; j <= len2; j++) matrix[0][j] = j;
    
    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[i][j] = Math.min(
          matrix[i - 1][j] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j - 1] + cost
        );
      }
    }
    
    // Convert distance to similarity score (0-1)
    const maxLen = Math.max(len1, len2);
    if (maxLen === 0) return 1.0; // Both strings empty
    
    const distance = matrix[len1][len2];
    return 1 - (distance / maxLen);
  };
  
  // Get feedback message based on pronunciation results
  const getFeedbackMessage = (isCorrect, similarityScore) => {
    if (isCorrect) {
      return 'Perfect pronunciation! 👏';
    } else if (similarityScore > 0.8) {
      return 'Very close! Try again with clearer pronunciation.';
    } else if (similarityScore > 0.6) {
      return 'Getting there! Focus on the correct sounds.';
    } else if (similarityScore > 0.4) {
      return 'Keep practicing! Your pronunciation needs improvement.';
    } else {
      return 'Try again. Listen carefully to the word pronunciation.';
    }
  };

  const resetPractice = () => {
    setFeedback(null);
    setPronunciationFeedback(null);
    setShowPronunciationResult(false);
    setSessionStats({ attempts: 0, correct: 0 });
  };

  const getSuccessRate = () => {
    if (sessionStats.attempts === 0) return 0;
    return Math.round((sessionStats.correct / sessionStats.attempts) * 100);
  };

  // Simple Status Display
  const SimpleStatusDisplay = () => (
    <div style={{
      position: 'fixed',
      top: '10px',
      right: '10px',
      background: 'rgba(0,0,0,0.85)',
      color: 'white',
      padding: '12px',
      borderRadius: '8px',
      fontSize: '11px',
      fontFamily: 'monospace',
      zIndex: 1000,
      minWidth: '250px',
      border: '2px solid #00ff88'
    }}>
      <div style={{fontWeight: 'bold', color: '#00ff88', marginBottom: '8px'}}>
        🎯 Backend Landmark Processing
      </div>
      
      <div style={{marginBottom: '4px'}}>
        <strong>Camera:</strong> {cameraReady ? '✅ Ready' : '❌ Loading'}
      </div>
      
      <div style={{marginBottom: '4px'}}>
        <strong>AI Model:</strong> {modelStatus}
      </div>
      
      <div style={{marginBottom: '4px'}}>
        <strong>Backend:</strong> Handles all landmarks
      </div>
      
      <div style={{marginBottom: '4px'}}>
        <strong>Success Rate:</strong> {getSuccessRate()}% ({sessionStats.correct}/{sessionStats.attempts})
      </div>
      
      <div style={{fontSize: '9px', color: '#888', marginTop: '6px'}}>
        extract_landmark.py on server
      </div>
    </div>
  );

  return (
    <div className="practice-page">
      <SimpleStatusDisplay />
      
      <div className="practice-header">
        <button onClick={onBack} className="back-button">← Back</button>
        <div className="practice-title">
          <h1>Practice: "{word}"</h1>
          <div className="points-display">
            Level {user?.stats?.level || 1} • Total {user?.stats?.total_points || 0} pts
          </div>
        </div>
      </div>

      <div className="practice-container">
        <div className="video-panel">
          <div className="panel-header">
            <h2>Reference Video</h2>
            <p>Watch and copy the gesture</p>
          </div>
          <video controls className="reference-video"
            src={`${BACKEND_URL}/videos/${word.toLowerCase()}.mp4`}
            onError={(e) => e.target.style.display = 'none'}>
          </video>
          
          {/* Dictation and Pronunciation Section */}
          <div className="dictation-section" style={{
            marginTop: '20px',
            padding: '15px',
            background: '#f0f8ff',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3 style={{ marginTop: '0', color: '#18119e' }}>🔊 Hear & Practice Pronunciation</h3>
            
            <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
              <button 
                onClick={() => speakWord(word)}
                style={{
                  flex: '1',
                  padding: '10px',
                  background: '#18119e',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                <span>🔊</span> Listen to "{word}"
              </button>
              
              <button 
                onClick={checkPronunciation}
                style={{
                  flex: '1',
                  padding: '10px',
                  background: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                <span>🎤</span> Practice Speaking
              </button>
            </div>
            
            {/* Pronunciation Feedback */}
            {showPronunciationResult && pronunciationFeedback && (
              <div style={{
                padding: '15px',
                background: pronunciationFeedback.error ? '#fff3f3' : 
                             (pronunciationFeedback.isCorrect ? '#f0fff0' : '#fff8e6'),
                borderRadius: '6px',
                marginTop: '10px',
                border: `1px solid ${pronunciationFeedback.error ? '#ffcccc' : 
                                    (pronunciationFeedback.isCorrect ? '#ccffcc' : '#ffe0b2')}`
              }}>
                {pronunciationFeedback.error ? (
                  <div>
                    <p style={{ margin: '0 0 10px 0', fontWeight: 'bold', color: '#d32f2f' }}>
                      ❌ {pronunciationFeedback.message}
                    </p>
                    <button 
                      onClick={checkPronunciation}
                      style={{
                        padding: '8px 12px',
                        background: '#f44336',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Try Again
                    </button>
                  </div>
                ) : (
                  <div>
                    <p style={{ 
                      margin: '0 0 5px 0', 
                      fontWeight: 'bold', 
                      color: pronunciationFeedback.isCorrect ? '#2e7d32' : '#ed6c02'
                    }}>
                      {pronunciationFeedback.isCorrect ? '✅' : '⚠️'} {pronunciationFeedback.feedbackMessage}
                    </p>
                    
                    <div style={{ margin: '10px 0' }}>
                      <p style={{ margin: '0 0 5px 0' }}>
                        <strong>You said:</strong> "{pronunciationFeedback.heard}"
                      </p>
                      <p style={{ margin: '0 0 5px 0' }}>
                        <strong>Target word:</strong> "{pronunciationFeedback.target}"
                      </p>
                      <p style={{ margin: '0' }}>
                        <strong>Similarity:</strong> {Math.round(pronunciationFeedback.similarityScore * 100)}%
                      </p>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                      <button 
                        onClick={() => speakWord(word)}
                        style={{
                          padding: '8px 12px',
                          background: '#18119e',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          flex: '1'
                        }}
                      >
                        Listen Again
                      </button>
                      
                      <button 
                        onClick={checkPronunciation}
                        style={{
                          padding: '8px 12px',
                          background: '#4CAF50',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          flex: '1'
                        }}
                      >
                        Try Again
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="camera-panel">
          <div className="panel-header">
            <h2>Your Practice</h2>
            <p>Backend extracts landmarks from frames</p>
          </div>
          
          <div className="camera-wrapper" style={{ position: 'relative' }}>
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

            <div className="camera-controls">
              {!isRecording && !isAnalyzing && (
                <>
                  <button 
                    onClick={startSimpleRecording} 
                    className="record-button"
                    disabled={!cameraReady}
                    style={{
                      background: cameraReady ? '#4CAF50' : '#ccc',
                      opacity: cameraReady ? 1 : 0.6,
                      marginBottom: '10px'
                    }}
                  >
                    Start {RECORDING_TIME}s Recording
                    {!cameraReady && <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>Camera loading...</div>}
                  </button>
                  
                  {/* NEW: Debug button */}
                  <button 
                    onClick={debugLandmarkCollection}
                    disabled={!cameraReady}
                    style={{
                      background: cameraReady ? '#FF9800' : '#ccc',
                      color: 'white',
                      border: 'none',
                      padding: '10px 20px',
                      borderRadius: '8px',
                      cursor: cameraReady ? 'pointer' : 'not-allowed',
                      fontSize: '0.9rem',
                      opacity: cameraReady ? 1 : 0.6
                    }}
                  >
                    🔍 Debug Landmarks
                  </button>
                </>
              )}
              
              {isAnalyzing && (
                <div style={{textAlign: 'center', padding: '20px'}}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    border: '4px solid #f3f3f3',
                    borderTop: '4px solid #18119e',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    margin: '0 auto 1rem'
                  }}></div>
                  <p>Backend processing landmarks...</p>
                  <p style={{ fontSize: '0.875rem', color: '#666' }}>extract_landmark.py method</p>
                </div>
              )}
            </div>

            {feedback && (
              <div className={`feedback-panel ${feedback.is_correct ? 'success' : 'retry'}`}>
                <div className="feedback-icon">{feedback.is_correct ? '🎉' : '🤔'}</div>
                <div className="feedback-message">{feedback.message}</div>
                
                <div className="feedback-stats">
                  <p><strong>AI Confidence:</strong> {Math.round(feedback.confidence * 100)}%</p>
                  <p><strong>Processing:</strong> Backend landmarks</p>
                  
                  {feedback.predicted_word && (
                    <p><strong>AI Detected:</strong> "{feedback.predicted_word}"</p>
                  )}
                  
                  {feedback.debug_info && (
                    <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem' }}>
                      <p>Landmark frames: {feedback.debug_info.landmark_frames}</p>
                      <p>Method: {feedback.debug_info.method}</p>
                    </div>
                  )}
                  
                  {feedback.is_correct && feedback.points > 0 && (
                    <div className="points-earned">+{feedback.points} Points! 🎯</div>
                  )}
                  
                  {/* Detailed Results Section */}
                  <div style={{ 
                    marginTop: '15px', 
                    padding: '10px', 
                    background: 'rgba(255,255,255,0.8)', 
                    borderRadius: '6px',
                    border: '1px solid #ddd'
                  }}>
                    <h4 style={{ margin: '0 0 10px 0', color: '#18119e' }}>Detailed Results</h4>
                    
                    {feedback.top_predictions && (
                      <div>
                        <p style={{ fontWeight: 'bold', margin: '5px 0' }}>Top Predictions:</p>
                        <ul style={{ margin: '0 0 10px 0', paddingLeft: '20px' }}>
                          {feedback.top_predictions.map((pred, idx) => (
                            <li key={idx} style={{ 
                              margin: '3px 0',
                              color: pred.word.toLowerCase() === word.toLowerCase() ? '#2e7d32' : 'inherit',
                              fontWeight: pred.word.toLowerCase() === word.toLowerCase() ? 'bold' : 'normal'
                            }}>
                              {pred.word} ({Math.round(pred.confidence * 100)}%)
                              {pred.word.toLowerCase() === word.toLowerCase() && ' ✓'}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}>
                      <button 
                        onClick={() => speakWord(word)}
                        style={{
                          padding: '8px 12px',
                          background: '#18119e',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px'
                        }}
                      >
                        <span>🔊</span> Hear Word
                      </button>
                      
                      <button 
                        onClick={checkPronunciation}
                        style={{
                          padding: '8px 12px',
                          background: '#4CAF50',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px'
                        }}
                      >
                        <span>🎤</span> Practice Speaking
                      </button>
                    </div>
                  </div>
                </div>
                
                <button onClick={resetPractice} className="retry-button">
                  {feedback.is_correct ? '🎯 Practice More' : '🔄 Try Again'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="tips-section">
        <h3>🎯 Simple Recognition Tips</h3>
        <div className="tips-grid">
          <div className="tip-item">
            <span>👋</span>
            <span>Keep hands visible throughout gesture</span>
          </div>
          <div className="tip-item">
            <span>💡</span>
            <span>Good lighting helps recognition</span>
          </div>
          <div className="tip-item">
            <span>📺</span>
            <span>Plain background recommended</span>
          </div>
          <div className="tip-item">
            <span>🎯</span>
            <span>Backend extracts all landmarks</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// User Profile Component (keeping same)
const UserProfile = ({ onBack }) => {
  const { user, logout } = useAuth();
  
  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa', padding: '2rem' }}>
      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        <button onClick={onBack} className="back-button">← Back</button>
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <h1>User Profile</h1>
          <p>Username: {user?.username}</p>
          <p>Email: {user?.email}</p>
          <button onClick={logout} style={{ background: '#dc3545', color: 'white', border: 'none', padding: '12px 24px', borderRadius: '6px', cursor: 'pointer' }}>
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [currentPage, setCurrentPage] = useState('landing');
  const [currentWord, setCurrentWord] = useState('');

  return (
    <AuthProvider>
      <div className="App">
        <AuthWrapper 
          currentPage={currentPage}
          currentWord={currentWord}
          setCurrentPage={setCurrentPage}
          setCurrentWord={setCurrentWord}
        />
      </div>
    </AuthProvider>
  );
}

// Auth Wrapper Component
const AuthWrapper = ({ currentPage, currentWord, setCurrentPage, setCurrentWord }) => {
  const { user, loading, isAuthenticated } = useAuth();

  const handleGetStarted = () => {
    if (isAuthenticated) {
      setCurrentPage('search');
    } else {
      setCurrentPage('auth');
    }
  };

  const handleAuthSuccess = () => setCurrentPage('search');
  const handleSearch = (word) => { 
    console.log(`🎯 Navigating to practice page for word: "${word}"`);
    setCurrentWord(word); 
    setCurrentPage('practice'); 
  };
  const handleBackToSearch = () => { setCurrentPage('search'); setCurrentWord(''); };
  const handleProfile = () => setCurrentPage('profile');
  const handleBackFromProfile = () => setCurrentPage('search');

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: '#f8f9fa' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="logo">
            <div className="logo-icon" style={{ animation: 'spin 2s linear infinite' }}></div>
          </div>
          <h2>GestureB</h2>
          <p>Loading simple landmark system...</p>
        </div>
      </div>
    );
  }

  switch (currentPage) {
    case 'auth': return <AuthPage onSuccess={handleAuthSuccess} />;
    case 'search': return isAuthenticated ? <SearchPage onSearch={handleSearch} onProfile={handleProfile} /> : <LandingPage onGetStarted={handleGetStarted} />;
    case 'practice': return isAuthenticated ? <PracticePage word={currentWord} onBack={handleBackToSearch} /> : <LandingPage onGetStarted={handleGetStarted} />;
    case 'profile': return isAuthenticated ? <UserProfile onBack={handleBackFromProfile} /> : <LandingPage onGetStarted={handleGetStarted} />;
    case 'landing':
    default: return <LandingPage onGetStarted={handleGetStarted} />;
  }
};

export default App;