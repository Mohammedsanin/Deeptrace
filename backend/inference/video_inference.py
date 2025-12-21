import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config import VIDEO_MODEL_PATH, DEVICE
from datasets.video_preprocessing import VideoPreprocessor
from models.video_model import load_model


class VideoInference:
    """Video deepfake detection inference"""
    
    def __init__(self, model_path=VIDEO_MODEL_PATH):
        self.device = DEVICE
        self.preprocessor = VideoPreprocessor()
        
        # Load model
        if Path(model_path).exists():
            self.model = load_model(model_path, self.device)
            print(f"✓ Video model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("Please train the model first using train_video.py")
            self.model = None
    
    def predict(self, video_file_path):
        """
        Predict if video is real or deepfake
        
        Args:
            video_file_path: Path to video file
        
        Returns:
            dict: {
                'prediction': 'Real' or 'Deepfake',
                'confidence': float (0-1),
                'probabilities': {'Real': float, 'Deepfake': float}
            }
        """
        if self.model is None:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'probabilities': {'Real': 0.0, 'Deepfake': 0.0},
                'error': 'Model not loaded'
            }
        
        try:
            # Preprocess video
            video_array = self.preprocessor.preprocess_video(video_file_path)
            
            if video_array is None:
                return {
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probabilities': {'Real': 0.0, 'Deepfake': 0.0},
                    'error': 'Failed to preprocess video'
                }
            
            # Convert to tensor (1, T, C, H, W)
            video_tensor = torch.FloatTensor(video_array).permute(0, 3, 1, 2).unsqueeze(0)
            video_tensor = video_tensor.to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(video_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction
            probs = probabilities.cpu().numpy()[0]
            pred_class = np.argmax(probs)
            confidence = float(probs[pred_class])
            
            prediction = 'Real' if pred_class == 0 else 'Deepfake'
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'Real': float(probs[0]),
                    'Deepfake': float(probs[1])
                }
            }
        
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'probabilities': {'Real': 0.0, 'Deepfake': 0.0},
                'error': str(e)
            }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'preprocessor'):
            del self.preprocessor


if __name__ == "__main__":
    # Test inference
    print("Testing Video Inference...")
    
    inference = VideoInference()
    
    if inference.model is not None:
        print("\n✓ Video inference module ready")
        print(f"Device: {inference.device}")
    else:
        print("\n⚠️ Model not available. Train the model first.")
