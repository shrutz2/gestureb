import os
from datetime import datetime

from flask import (
    request,
    jsonify,
    send_from_directory,
    current_app as app,
    Response,
)
from celery.result import AsyncResult
import cv2

from app import db
from app.models import Video, FrameLandmark  # DB Models
from app.tasks import process_video_landmarks, celery
from app.config import VIDEO_DIR
from app.utils.real_time_recognition import get_recognizer

# =============================================================================
# API ROUTES ONLY - NO HTML RENDERING
# =============================================================================

@app.route('/')
def api_info():
    """API Information endpoint"""
    return jsonify({
        'name': 'Sign Language Recognition API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'auth': '/api/auth/*',
            'recognition': '/api/recognition/*',
            'videos': '/api/videos/*',
            'docs': 'API documentation coming soon'
        }
    })

# =============================================================================
# VIDEO STREAMING API
# =============================================================================

def generate_frames():
    """Generate frames for video streaming"""
    recognizer = get_recognizer()
    
    while True:
        frame = recognizer.get_frame()
        if frame is None:
            # Send a black frame if no frame available
            frame = recognizer.get_frame()  # Will return placeholder
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route for API"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/recognition/predictions')
def current_predictions():
    """Get current predictions as JSON API"""
    recognizer = get_recognizer()
    current, top3 = recognizer.get_current_predictions()
    
    if current is None:
        return jsonify({
            'status': 'waiting',
            'message': 'Collecting frames...',
            'data': None
        })
    
    return jsonify({
        'status': 'success',
        'message': 'Predictions available',
        'data': {
            'current_prediction': current,
            'top3': [
                {'label': label, 'confidence': float(conf)}
                for label, conf in top3
            ]
        }
    })

@app.route('/api/recognition/toggle', methods=['POST'])
def toggle_recognition():
    """Start or stop recognition via API"""
    try:
        data = request.get_json()
        action = data.get('action', 'start') if data else 'start'
        recognizer = get_recognizer()
        
        if action == 'start':
            success = recognizer.start_capture()
            return jsonify({
                'status': 'started' if success else 'error',
                'success': success,
                'message': 'Camera started successfully' if success else 'Failed to start camera'
            })
        else:
            recognizer.stop_capture()
            return jsonify({
                'status': 'stopped',
                'success': True,
                'message': 'Camera stopped successfully'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/recognition/status')
def recognition_status():
    """Get current recognition status via API"""
    recognizer = get_recognizer()
    return jsonify({
        'status': 'success',
        'data': {
            'is_running': recognizer.is_running,
            'timestamp': datetime.utcnow().isoformat()
        }
    })

# =============================================================================
# VIDEO MANAGEMENT API
# =============================================================================

def _save_video_file(label, video_file):
    """
    Saves uploaded video to disk and creates a Video record in the database.
    Returns the Video instance.
    """
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.mp4"
    file_path = os.path.join(VIDEO_DIR, filename)
    
    # Save file
    video_file.save(file_path)
    
    # Create database record
    video = Video(
        label=label,
        filename=filename,
        file_path=file_path,
        created_at=datetime.utcnow()
    )
    db.session.add(video)
    db.session.commit()
    
    return video

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get all videos via API"""
    try:
        videos = Video.query.all()
        video_list = []
        
        for video in videos:
            video_data = {
                'id': video.id,
                'label': video.label,
                'filename': video.filename,
                'created_at': video.created_at.isoformat() if video.created_at else None,
                'url': f'/api/videos/download/{video.id}'
            }
            video_list.append(video_data)
        
        return jsonify({
            'status': 'success',
            'count': len(video_list),
            'data': video_list
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving videos: {str(e)}'
        }), 500

@app.route('/api/videos/upload', methods=['POST'])
def upload_video():
    """Upload video via API"""
    try:
        if 'video' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        label = request.form.get('label', 'unknown')
        
        if video_file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Save video
        video = _save_video_file(label, video_file)
        
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded successfully',
            'data': {
                'id': video.id,
                'label': video.label,
                'filename': video.filename,
                'url': f'/api/videos/download/{video.id}'
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }), 500

@app.route('/api/videos/download/<int:video_id>')
def download_video(video_id):
    """Download video file via API"""
    try:
        video = Video.query.get_or_404(video_id)
        return send_from_directory(
            VIDEO_DIR, 
            video.filename, 
            as_attachment=True
        )
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Download failed: {str(e)}'
        }), 404

@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete video via API"""
    try:
        video = Video.query.get_or_404(video_id)
        
        # Delete file from filesystem
        if os.path.exists(video.file_path):
            os.remove(video.file_path)
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Video deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Delete failed: {str(e)}'
        }), 500

# =============================================================================
# ML PROCESSING API
# =============================================================================

@app.route('/api/ml/process', methods=['POST'])
def process_video_api():
    """Process video for landmarks via API"""
    try:
        if 'video' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        label = request.form.get('label', 'unknown')
        
        # Save video first
        video = _save_video_file(label, video_file)
        
        # Start processing task
        task = process_video_landmarks.delay(video.id)
        
        return jsonify({
            'status': 'processing',
            'message': 'Video processing started',
            'data': {
                'task_id': task.id,
                'video_id': video.id,
                'label': label
            }
        }), 202
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Processing failed: {str(e)}'
        }), 500

@app.route('/api/ml/task/<task_id>')
def get_task_status(task_id):
    """Get ML task status via API"""
    try:
        task = AsyncResult(task_id, app=celery)
        
        if task.state == 'PENDING':
            response = {
                'status': 'pending',
                'message': 'Task is waiting to be processed'
            }
        elif task.state == 'PROGRESS':
            response = {
                'status': 'processing',
                'message': 'Task is being processed',
                'progress': task.info.get('progress', 0)
            }
        elif task.state == 'SUCCESS':
            response = {
                'status': 'completed',
                'message': 'Task completed successfully',
                'result': task.info
            }
        else:
            response = {
                'status': 'failed',
                'message': 'Task failed',
                'error': str(task.info)
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Task status check failed: {str(e)}'
        }), 500

# =============================================================================
# HEALTH CHECK API
# =============================================================================

@app.route('/api/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': 'connected',
            'celery': 'running' if celery.control.inspect().active() else 'stopped',
            'recognition': 'available'
        }
    })

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """API 404 handler"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'code': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """API 500 handler"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'code': 500
    }), 500