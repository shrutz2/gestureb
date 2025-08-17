import React, { useState, useEffect, useRef } from 'react';
import './App.css'; // Using existing styles

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';
const API = `${BACKEND_URL}/api`;

const SentencePractice = ({ sentence, onBack }) => {
  const [words, setWords] = useState([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [completedWords, setCompletedWords] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [sentenceProgress, setSentenceProgress] = useState(0);
  const [sessionStats, setSessionStats] = useState({ 
    totalWords: 0, 
    completedWords: 0, 
    correctAttempts: 0,
    totalAttempts: 0,
    points: 0
  });

  useEffect(() => {
    if (sentence) {
      initializeSentence();
    }
  }, [sentence]);

  const initializeSentence = async () => {
    setIsLoading(true);
    try {
      // Split sentence into words and clean them
      const wordList = sentence
        .toLowerCase()
        .replace(/[^\w\s]/g, '') // Remove punctuation
        .split(' ')
        .filter(word => word.length > 0)
        .map(word => word.trim());

      // Check which words are available in the model
      const availableWords = [];
      for (const word of wordList) {
        try {
          const response = await fetch(`${API}/word-info/${word}`);
          if (response.ok) {
            const data = await response.json();
            if (data.available) {
              availableWords.push({
                word: word,
                index: availableWords.length,
                completed: false,
                attempts: 0,
                correct: false
              });
            }
          }
        } catch (error) {
          console.log(`Word "${word}" not available in model`);
        }
      }

      setWords(availableWords);
      setSessionStats(prev => ({
        ...prev,
        totalWords: availableWords.length
      }));
      
      if (availableWords.length === 0) {
        alert('No words from this sentence are available in the model. Please try a different sentence.');
        onBack();
      }
    } catch (error) {
      console.error('Error initializing sentence:', error);
      alert('Failed to process sentence. Please try again.');
      onBack();
    } finally {
      setIsLoading(false);
    }
  };

  const handleWordComplete = (wordData) => {
    const updatedWords = [...words];
    updatedWords[currentWordIndex] = {
      ...updatedWords[currentWordIndex],
      completed: true,
      correct: wordData.is_correct,
      attempts: updatedWords[currentWordIndex].attempts + 1,
      confidence: wordData.confidence
    };
    
    setWords(updatedWords);
    setCompletedWords(prev => [...prev, updatedWords[currentWordIndex]]);
    
    // Update session stats
    setSessionStats(prev => ({
      ...prev,
      completedWords: prev.completedWords + 1,
      correctAttempts: wordData.is_correct ? prev.correctAttempts + 1 : prev.correctAttempts,
      totalAttempts: prev.totalAttempts + 1,
      points: prev.points + (wordData.points || 0)
    }));

    // Update progress
    const newProgress = ((currentWordIndex + 1) / words.length) * 100;
    setSentenceProgress(newProgress);

    // Move to next word or complete sentence
    if (currentWordIndex < words.length - 1) {
      setTimeout(() => {
        setCurrentWordIndex(currentWordIndex + 1);
      }, 2000); // 2 second delay before next word
    } else {
      // Sentence completed
      setTimeout(() => {
        alert(`🎉 Sentence completed! You got ${sessionStats.correctAttempts + (wordData.is_correct ? 1 : 0)}/${words.length} words correct!`);
      }, 1000);
    }
  };

  const resetSentence = () => {
    setCurrentWordIndex(0);
    setCompletedWords([]);
    setSentenceProgress(0);
    setWords(words.map(word => ({
      ...word,
      completed: false,
      correct: false,
      attempts: 0
    })));
    setSessionStats({
      totalWords: words.length,
      completedWords: 0,
      correctAttempts: 0,
      totalAttempts: 0,
      points: 0
    });
  };

  const skipCurrentWord = () => {
    if (currentWordIndex < words.length - 1) {
      setCurrentWordIndex(currentWordIndex + 1);
      setSentenceProgress(((currentWordIndex + 1) / words.length) * 100);
    }
  };

  const goToPreviousWord = () => {
    if (currentWordIndex > 0) {
      setCurrentWordIndex(currentWordIndex - 1);
      setSentenceProgress((currentWordIndex / words.length) * 100);
    }
  };

  if (isLoading) {
    return (
      <div className="sentence-practice loading">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Processing sentence...</p>
        </div>
      </div>
    );
  }

  if (words.length === 0) {
    return (
      <div className="sentence-practice error">
        <h2>No Available Words</h2>
        <p>None of the words in "{sentence}" are available in our model.</p>
        <button onClick={onBack} className="back-button">Try Another Sentence</button>
      </div>
    );
  }

  const currentWord = words[currentWordIndex];
  const isLastWord = currentWordIndex === words.length - 1;
  const allWordsCompleted = words.every(word => word.completed);

  return (
    <div className="sentence-practice">
      <div className="sentence-header">
        <button onClick={onBack} className="back-button">
          ← Back to Search
        </button>
        <div className="sentence-title">
          <h1>Sentence Practice</h1>
          <p className="sentence-text">"{sentence}"</p>
        </div>
      </div>

      <div className="sentence-progress-section">
        <div className="progress-header">
          <h3>Progress: {sessionStats.completedWords}/{sessionStats.totalWords} words</h3>
          <div className="stats-summary">
            <span>Success Rate: {sessionStats.totalAttempts > 0 ? Math.round((sessionStats.correctAttempts / sessionStats.totalAttempts) * 100) : 0}%</span>
            <span>Points: {sessionStats.points}</span>
          </div>
        </div>
        
        <div className="progress-bar-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${sentenceProgress}%` }}
            ></div>
          </div>
        </div>

        <div className="words-overview">
          {words.map((word, index) => (
            <div 
              key={index} 
              className={`word-status ${
                index === currentWordIndex ? 'current' : 
                word.completed ? (word.correct ? 'completed-correct' : 'completed-incorrect') : 
                'pending'
              }`}
              onClick={() => index <= currentWordIndex && setCurrentWordIndex(index)}
            >
              <span className="word-text">{word.word}</span>
              <span className="word-indicator">
                {word.completed ? (word.correct ? '✅' : '❌') : 
                 index === currentWordIndex ? '👉' : '⏳'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {!allWordsCompleted && currentWord && (
        <div className="current-word-practice">
          <div className="word-focus">
            <h2>Practice Word {currentWordIndex + 1}: "{currentWord.word}"</h2>
            <p>Focus on this word and sign it clearly</p>
          </div>

          {/* Embed the existing PracticePage component logic here */}
          <WordPracticeEmbed 
            word={currentWord.word}
            onComplete={handleWordComplete}
            wordNumber={currentWordIndex + 1}
            totalWords={words.length}
          />

          <div className="word-navigation">
            <button 
              onClick={goToPreviousWord}
              disabled={currentWordIndex === 0}
              className="nav-button"
            >
              ← Previous Word
            </button>
            
            <button 
              onClick={skipCurrentWord}
              disabled={isLastWord}
              className="nav-button skip"
            >
              Skip Word →
            </button>
          </div>
        </div>
      )}

      {allWordsCompleted && (
        <div className="sentence-completed">
          <div className="completion-celebration">
            <h2>🎉 Sentence Completed!</h2>
            <p>Great job practicing "{sentence}"</p>
            
            <div className="final-stats">
              <div className="stat-item">
                <span className="stat-number">{sessionStats.correctAttempts}</span>
                <span className="stat-label">Correct Words</span>
              </div>
              <div className="stat-item">
                <span className="stat-number">{sessionStats.totalWords}</span>
                <span className="stat-label">Total Words</span>
              </div>
              <div className="stat-item">
                <span className="stat-number">{Math.round((sessionStats.correctAttempts / sessionStats.totalWords) * 100)}%</span>
                <span className="stat-label">Success Rate</span>
              </div>
              <div className="stat-item">
                <span className="stat-number">{sessionStats.points}</span>
                <span className="stat-label">Points Earned</span>
              </div>
            </div>

            <div className="completion-actions">
              <button onClick={resetSentence} className="action-button primary">
                Practice Again
              </button>
              <button onClick={onBack} className="action-button secondary">
                Try New Sentence
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="sentence-tips">
        <h3>Sentence Practice Tips</h3>
        <div className="tips-grid">
          <div className="tip-item">
            <span className="tip-icon">📝</span>
            <span>Practice each word slowly and clearly</span>
          </div>
          <div className="tip-item">
            <span className="tip-icon">🔄</span>
            <span>Repeat difficult words multiple times</span>
          </div>
          <div className="tip-item">
            <span className="tip-icon">⏸️</span>
            <span>Take breaks between words if needed</span>
          </div>
          <div className="tip-item">
            <span className="tip-icon">🎯</span>
            <span>Focus on accuracy over speed</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Simplified embedded word practice component
const WordPracticeEmbed = ({ word, onComplete, wordNumber, totalWords }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [frames, setFrames] = useState([]);
  const [countdown, setCountdown] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [captureProgress, setCaptureProgress] = useState(0);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recordingIntervalRef = useRef(null);

  useEffect(() => {
    startWebcam();
    return () => {
      cleanup();
    };
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
        video: { width: 640, height: 480 } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraReady(true);
      }
    } catch (error) {
      console.error('Webcam error:', error);
      alert('Camera access needed for practice');
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

  const startRecording = () => {
    if (!cameraReady) return;

    setIsRecording(true);
    setFrames([]);
    setFeedback(null);
    setCountdown(3);
    setCaptureProgress(0);

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

    // Start recording after countdown
    setTimeout(() => {
      const captureCount = 15;
      const interval = 200;
      let frameCount = 0;

      recordingIntervalRef.current = setInterval(() => {
        const frame = captureFrame();
        if (frame) {
          setFrames(prev => [...prev, frame]);
          frameCount++;
          setCaptureProgress((frameCount / captureCount) * 100);
          
          if (frameCount >= captureCount) {
            clearInterval(recordingIntervalRef.current);
            setIsRecording(false);
            analyzeGesture();
          }
        }
      }, interval);
    }, 3000);
  };

  const analyzeGesture = async () => {
    if (frames.length === 0) return;

    setIsAnalyzing(true);

    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'User-Id': 'sentence-practice'
        },
        body: JSON.stringify({
          frames: frames,
          target_word: word
        })
      });

      if (!response.ok) throw new Error('Prediction failed');

      const result = await response.json();
      setFeedback(result);
      
      // Automatically complete word after 3 seconds if correct
      if (result.is_correct) {
        setTimeout(() => {
          onComplete(result);
        }, 3000);
      }
      
    } catch (error) {
      console.error('Analysis error:', error);
      setFeedback({
        message: 'Analysis failed. Please try again!',
        is_correct: false,
        confidence: 0,
        points: 0
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTryAgain = () => {
    setFrames([]);
    setFeedback(null);
    setCaptureProgress(0);
  };

  const handleSkipWord = () => {
    onComplete({
      is_correct: false,
      message: 'Word skipped',
      confidence: 0,
      points: 0
    });
  };

  return (
    <div className="word-practice-embed">
      <div className="embed-header">
        <h3>Word {wordNumber} of {totalWords}: "{word}"</h3>
      </div>

      <div className="embed-camera">
        <video 
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="embed-video"
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        
        {countdown > 0 && (
          <div className="countdown-overlay">
            <div className="countdown-circle">
              <span className="countdown-number">{countdown}</span>
            </div>
          </div>
        )}

        {isRecording && countdown === 0 && (
          <div className="capture-overlay">
            <div className="recording-indicator">
              <span className="rec-dot"></span>
              Recording...
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${captureProgress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      <div className="embed-controls">
        {!isRecording && !isAnalyzing && !feedback && (
          <button 
            onClick={startRecording}
            className="record-button"
            disabled={!cameraReady}
          >
            Practice "{word}"
          </button>
        )}
        
        {isAnalyzing && (
          <div className="analyzing-indicator">
            <div className="spinner"></div>
            <span>Analyzing...</span>
          </div>
        )}

        {feedback && (
          <div className={`embed-feedback ${feedback.is_correct ? 'success' : 'retry'}`}>
            <div className="feedback-content">
              <div className="feedback-icon">
                {feedback.is_correct ? '✅' : '❌'}
              </div>
              <div className="feedback-text">
                <p className="feedback-message">{feedback.message}</p>
                <div className="feedback-details">
                  <span>Confidence: {Math.round(feedback.confidence * 100)}%</span>
                  {feedback.predicted_word && (
                    <span>Detected: {feedback.predicted_word}</span>
                  )}
                </div>
              </div>
            </div>
            
            {feedback.is_correct ? (
              <div className="success-actions">
                <p className="auto-continue">Moving to next word in 3 seconds...</p>
                <button 
                  onClick={() => onComplete(feedback)}
                  className="continue-button"
                >
                  Continue Now
                </button>
              </div>
            ) : (
              <div className="retry-actions">
                <button 
                  onClick={handleTryAgain}
                  className="try-again-button"
                >
                  Try Again
                </button>
                <button 
                  onClick={handleSkipWord}
                  className="skip-button"
                >
                  Skip Word
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SentencePractice;
