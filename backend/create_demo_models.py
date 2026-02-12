"""
Demo/Mock Models for Deeptrace
This creates pre-initialized models for demonstration purposes
without requiring dataset training.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import AUDIO_MODEL_PATH, VIDEO_MODEL_PATH, MODELS_DIR
from models.audio_model import AudioDeepfakeDetector, save_model as save_audio_model
from models.video_model import VideoDeepfakeDetector, save_model as save_video_model


def create_demo_audio_model():
    """Create and save a demo audio model"""
    print("Creating demo audio model...")
    
    # Create model
    model = AudioDeepfakeDetector()
    
    # Initialize with random weights (already done by default)
    # In a real scenario, these would be trained weights
    
    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    save_audio_model(model, AUDIO_MODEL_PATH)
    
    print(f"✓ Demo audio model saved to {AUDIO_MODEL_PATH}")
    return model


def create_demo_video_model():
    """Create and save a demo video model"""
    print("Creating demo video model...")
    
    # Create model
    model = VideoDeepfakeDetector()
    
    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    save_video_model(model, VIDEO_MODEL_PATH)
    
    print(f"✓ Demo video model saved to {VIDEO_MODEL_PATH}")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Demo Models for Deeptrace")
    print("=" * 60)
    print("\nNote: These are untrained models for demonstration purposes.")
    print("They will make random predictions until properly trained.\n")
    
    # Create both models
    audio_model = create_demo_audio_model()
    print()
    video_model = create_demo_video_model()
    
    print("\n" + "=" * 60)
    print("✅ Demo models created successfully!")
    print("=" * 60)
    print(f"\nAudio model: {AUDIO_MODEL_PATH}")
    print(f"Video model: {VIDEO_MODEL_PATH}")
    print("\nYou can now run the backend server:")
    print("  python main.py")
    print("\nNote: Predictions will be random until models are trained.")
