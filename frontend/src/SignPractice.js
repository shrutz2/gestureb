// PracticePage.jsx  (FULL FILE)
import React, { useState, useEffect, useRef } from 'react';
import './App.css';               // your existing styles

const RECORDING_SEC   = 4;
const FRAME_INTERVAL  = 150;      // ms  (≈ 6.6 fps)
const BUFFER_LIMIT    = 10;       // frames
const COUNTDOWN_SEC   = 3;

export default function PracticePage({ word, onBack }) {
  const { user, token, refreshUserStats } = useAuth();   // from your AuthContext
  const videoRef        = useRef(null);
  const canvasRef       = useRef(null);
  const overlayRef      = useRef(null);
  const streamRef       = useRef(null);

  /* ----- state ----- */
  const [cameraReady, setCameraReady]     = useState(false);
  const [handsDetected, setHandsDetected] = useState(false);
  const [detectionConf, setDetectionConf] = useState(0);
  const [mediaPipeHands, setMediaPipeHands]=useState(null);

  const [countdown, setCountdown]         = useState(0);
  const [isRecording, setIsRecording]     = useState(false);
  const [isAnalysing, setIsAnalysing]     = useState(false);
  const [feedback, setFeedback]           = useState(null);

  const [realtimeFrames, setRealtimeFrames] = useState([]);   // base64 frames
  const [realLandmarks, setRealLandmarks]   = useState([]);   // raw 21-pt arrays
  const [lastPred, setLastPred]             = useState('');

  const liveIntervalRef = useRef(null);   // only for live overlay
  const recDataRef        = useRef({ frames: [], landmarks: [] }); // snapshot buffer

  /* ==========  LIFE-CYCLE  ========== */
  useEffect(() => { startCam(); return cleanup; }, []);
  useEffect(() => {               // keep live feed running forever
    if (cameraReady) startLiveSnapshot();
    return () => clearInterval(liveIntervalRef.current);
  }, [cameraReady]);

  /* ----------  MEDIA PIPE INIT  ---------- */
  const startCam = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 30 } }
    });
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => setCameraReady(true);
    }
  };

  const cleanup = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    clearInterval(liveIntervalRef.current);
    if (mediaPipeHands) mediaPipeHands.close();
  };

  /* ----------  LIVE OVERLAY (never stopped)  ---------- */
  const startLiveSnapshot = () => {
    if (liveIntervalRef.current) return;
    liveIntervalRef.current = setInterval(async () => {
      if (!videoRef.current || !overlayRef.current) return;
      const frame = captureFrame();
      if (frame) {
        setRealtimeFrames(prev => [...prev, frame].slice(-BUFFER_LIMIT));
        if (!mediaPipeHands) mockDetect(frame);   // fallback
      }
    }, FRAME_INTERVAL);
  };

  /* ----------  REAL MEDIAPIPE  ---------- */
  useEffect(() => {
    if (!cameraReady) return;
    const script1 = document.createElement('script');
    script1.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/hands.min.js';
    script1.async = true;
    script1.onload = () => {
      const hands = new window.Hands({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`
      });
      hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.7, minTrackingConfidence: 0.7 });
      hands.onResults(onHandsResults);
      setMediaPipeHands(hands);

      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => await hands.send({ image: videoRef.current }),
        width: 640, height: 480
      });
      camera.start();
    };
    document.body.appendChild(script1);
  }, [cameraReady]);

  const onHandsResults = (res) => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext('2d');
    overlayRef.current.width  = 640;
    overlayRef.current.height = 480;
    ctx.clearRect(0, 0, 640, 480);

    if (res.multiHandLandmarks && res.multiHandLandmarks.length) {
      setHandsDetected(true);
      setDetectionConf(res.multiHandedness[0].score);

      const all = [];                       // flatten 21 pts
      res.multiHandLandmarks.forEach(hand => {
        hand.forEach(p => all.push({ x: p.x, y: p.y, z: p.z }));
        // draw red dots
        hand.forEach(p => {
          ctx.fillStyle = '#FF4444'; ctx.beginPath();
          ctx.arc(p.x * 640, p.y * 480, 6, 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
        });
      });
      setRealLandmarks(all);
    } else {
      setHandsDetected(false); setDetectionConf(0); setRealLandmarks([]);
    }
  };

  const mockDetect = (frame) => { /* your old mock code if MP fails */ };

  /* ----------  4-SEC RECORDING (only snapshot)  ---------- */
  const startRecording = () => {
    if (!handsDetected) return alert('Show your hands first');
    setCountdown(COUNTDOWN_SEC);
    setFeedback(null);
    setIsRecording(true);

    // clear snapshot buffer
    recDataRef.current = { frames: [...realtimeFrames], landmarks: [...realLandmarks] };

    // countdown
    const cd = setInterval(() => {
      setCountdown(c => {
        if (c === 1) { clearInterval(cd); return 0; }
        return c - 1;
      });
    }, 1000);

    // after countdown simply call analyse
    setTimeout(() => {
      setIsRecording(false);
      analyseGesture();
    }, COUNTDOWN_SEC * 1000);
  };

  /* ----------  ANALYSE (no timers touched)  ---------- */
  const analyseGesture = async () => {
    const { frames, landmarks } = recDataRef.current;
    if (frames.length < 5) {
      setFeedback({ is_correct: false, message: 'Need more frames – keep hands visible' });
      return;
    }
    setIsAnalysing(true);
    try {
      const res = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          frames, landmarks, target_word: word,
          real_hand_detection: true, enhanced_processing: true
        })
      });
      const data = await res.json();
      setFeedback(data);
      if (data.is_correct) refreshUserStats();
    } catch (e) { setFeedback({ is_correct: false, message: 'Network error' }); }
    setIsAnalysing(false);
  };

  /* ----------  UTILS  ---------- */
  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    const ctx = canvasRef.current.getContext('2d');
    canvasRef.current.width = 640; canvasRef.current.height = 480;
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    return canvasRef.current.toDataURL('image/jpeg', 0.9);
  };

  /* ----------  RENDER  ---------- */
  return (
    <div className="practice-page">
      {/* header */}
      <div className="practice-header">
        <button onClick={onBack} className="back-button">← Back</button>
        <div className="practice-title">
          <h1>Practice "{word}"</h1>
          <div className="points-display">
            Level {user?.stats?.level || 1} • Total {user?.stats?.total_points || 0} pts
          </div>
        </div>
      </div>

      <div className="practice-container">
        {/* reference video panel */}
        <div className="video-panel">
          <h2>Reference Video</h2>
          <video controls className="reference-video"
                 src={`http://localhost:5000/videos/${word}.mp4`}
                 onError={e => e.target.style.display = 'none'} />
        </div>

        {/* camera panel */}
        <div className="camera-panel">
          <h2>Your Practice</h2>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '1rem', fontSize: '0.9rem' }}>
            <span style={{ background: handsDetected ? '#4caf50' : '#f44336', color: 'white', padding: '4px 12px', borderRadius: 12 }}>
              {handsDetected ? '✅ Hands' : '❌ No Hands'}
            </span>
            <span>Conf: {Math.round(detectionConf * 100)}%</span>
            <span>Method: {mediaPipeHands ? 'MediaPipe' : 'Mock'}</span>
          </div>

          <div style={{ position: 'relative' }}>
            <video ref={videoRef} autoPlay muted playsInline className="practice-camera" />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <canvas ref={overlayRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', transform: 'scaleX(-1)' }} />

            {countdown > 0 && (
              <div className="countdown-overlay">
                <div className="countdown-circle"><span className="countdown-number">{countdown}</span></div>
                <div className="countdown-text">Get Ready!</div>
              </div>
            )}

            <div className="camera-controls">
              {!isRecording && !isAnalysing && (
                <button onClick={startRecording} className="record-button" disabled={!handsDetected}
                        style={{ background: handsDetected ? '#4caf50' : '#ccc' }}>
                  Start {RECORDING_SEC}s Recording
                </button>
              )}
              {isAnalysing && (
                <div style={{ textAlign: 'center' }}>
                  <div className="spinner" />
                  <p>Analysing…</p>
                </div>
              )}
            </div>

            {feedback && (
              <div className={`feedback-panel ${feedback.is_correct ? 'success' : 'retry'}`}>
                <div className="feedback-icon">{feedback.is_correct ? '🎉' : '🤔'}</div>
                <div className="feedback-message">{feedback.message}</div>
                {feedback.is_correct && <div className="points-earned">+{feedback.points} pts</div>}
                <button onClick={() => setFeedback(null)} className="retry-button">Practice Again</button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="tips-section">
        <h3>Recognition Tips</h3>
        <div className="tips-grid">
          <div className="tip-item"><span>👋</span><span>Keep hands clearly visible</span></div>
          <div className="tip-item"><span>💡</span><span>Bright, even lighting</span></div>
          <div className="tip-item"><span>📺</span><span>Plain background</span></div>
          <div className="tip-item"><span>🐌</span><span>Move slowly</span></div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* tiny helper to consume AuthContext (keep wherever you already have it) */
const useAuth = () => {
  const ctx = React.useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
};