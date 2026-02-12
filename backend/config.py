import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
# Use D: drive for large datasets to avoid space issues on C:
DATA_DIR = Path("D:/Deeptrace_Data")
MODELS_DIR = BASE_DIR / "saved_models"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset paths
AUDIO_DATASET_DIR = DATA_DIR / "asvspoof2019"
VIDEO_DATASET_DIR = DATA_DIR / "video_dataset"  # Your new video dataset will go here

# Model paths
AUDIO_MODEL_PATH = MODELS_DIR / "audio_model.pth"
VIDEO_MODEL_PATH = MODELS_DIR / "video_model.pth"

# Audio preprocessing config
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 128,
    "n_mfcc": 40,
    "max_duration": 4,  # seconds
    "hop_length": 512,
    "n_fft": 2048,
}

# Video preprocessing config
VIDEO_CONFIG = {
    "frame_size": (224, 224),
    "fps": 5,  # frames per second to extract
    "max_frames": 30,  # maximum frames per video
    "face_detection_confidence": 0.5,
}

# Model hyperparameters
AUDIO_MODEL_CONFIG = {
    "input_channels": 1,
    "num_classes": 2,  # Real or Fake
    "dropout": 0.5,
}

VIDEO_MODEL_CONFIG = {
    "num_classes": 2,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "dropout": 0.5,
}

# Training config
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
}

# API config
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_audio_formats": [".wav", ".mp3", ".m4a", ".flac"],
    "allowed_video_formats": [".mp4", ".avi", ".mov", ".mkv"],
}

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
