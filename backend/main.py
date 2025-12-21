from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os
from datetime import datetime

from config import API_CONFIG, UPLOADS_DIR
from inference.audio_inference import AudioInference
from inference.video_inference import VideoInference

# Initialize FastAPI app
app = FastAPI(
    title="VoiceGuard API",
    description="End-to-End Deepfake Detection System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for test samples
TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"
if TEST_FILES_DIR.exists():
    app.mount("/test_files", StaticFiles(directory=str(TEST_FILES_DIR)), name="test_files")
    print(f"✓ Mounted test_files directory: {TEST_FILES_DIR}")
else:
    print(f"⚠️ test_files directory not found: {TEST_FILES_DIR}")
    print(f"   Looking for: {TEST_FILES_DIR.absolute()}")



# Initialize inference modules
audio_inference = AudioInference()
video_inference = VideoInference()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VoiceGuard API - Deepfake Detection System",
        "version": "1.0.0",
        "endpoints": {
            "audio": "/predict/audio",
            "video": "/predict/video"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "audio_model": "loaded" if audio_inference.model is not None else "not loaded",
        "video_model": "loaded" if video_inference.model is not None else "not loaded"
    }


def validate_file_size(file: UploadFile):
    """Validate file size"""
    # Read file to check size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > API_CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {API_CONFIG['max_file_size'] / (1024*1024):.0f}MB"
        )


def validate_audio_format(filename: str):
    """Validate audio file format"""
    ext = Path(filename).suffix.lower()
    if ext not in API_CONFIG["allowed_audio_formats"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Allowed formats: {', '.join(API_CONFIG['allowed_audio_formats'])}"
        )


def validate_video_format(filename: str):
    """Validate video file format"""
    ext = Path(filename).suffix.lower()
    if ext not in API_CONFIG["allowed_video_formats"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed formats: {', '.join(API_CONFIG['allowed_video_formats'])}"
        )


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict if uploaded audio is real or deepfake
    
    Args:
        file: Audio file (wav, mp3, m4a, flac)
    
    Returns:
        JSON with prediction results
    """
    try:
        # Validate file
        validate_file_size(file)
        validate_audio_format(file.filename)
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"audio_{timestamp}_{file.filename}"
        temp_path = UPLOADS_DIR / temp_filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference
        result = audio_inference.predict(str(temp_path))
        
        # Clean up
        if temp_path.exists():
            os.remove(temp_path)
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return JSONResponse(content={
            "success": True,
            "type": "audio",
            "filename": file.filename,
            "result": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities']
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict if uploaded video is real or deepfake
    
    Args:
        file: Video file (mp4, avi, mov, mkv)
    
    Returns:
        JSON with prediction results
    """
    try:
        # Validate file
        validate_file_size(file)
        validate_video_format(file.filename)
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"video_{timestamp}_{file.filename}"
        temp_path = UPLOADS_DIR / temp_filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference
        result = video_inference.predict(str(temp_path))
        
        # Clean up
        if temp_path.exists():
            os.remove(temp_path)
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return JSONResponse(content={
            "success": True,
            "type": "video",
            "filename": file.filename,
            "result": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities']
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🛡️ VoiceGuard API Server")
    print("=" * 60)
    print(f"Audio Model: {'✓ Loaded' if audio_inference.model else '✗ Not loaded'}")
    print(f"Video Model: {'✓ Loaded' if video_inference.model else '✗ Not loaded'}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        log_level="info"
    )
