import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config import AUDIO_MODEL_PATH, DEVICE
from datasets.audio_preprocessing import AudioPreprocessor
from models.audio_model import load_model


class AudioInference:
    """Audio deepfake detection inference"""
    
    def __init__(self, model_path=AUDIO_MODEL_PATH):
        self.device = DEVICE
        self.preprocessor = AudioPreprocessor()
        
        # Load model
        if Path(model_path).exists():
            self.model = load_model(model_path, self.device)
            print(f"✓ Audio model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("Please train the model first using train_audio.py")
            self.model = None
    
    def predict(self, audio_file_path):
        """
        Predict if audio is real or deepfake
        
        Args:
            audio_file_path: Path to audio file
        
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
            # Preprocess audio
            features = self.preprocessor.preprocess(audio_file_path)
            
            if features is None:
                return {
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probabilities': {'Real': 0.0, 'Deepfake': 0.0},
                    'error': 'Failed to preprocess audio'
                }
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
            features_tensor = features_tensor.to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
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


if __name__ == "__main__":
    # Test inference
    print("Testing Audio Inference...")
    
    inference = AudioInference()
    
    if inference.model is not None:
        print("\n✓ Audio inference module ready")
        print(f"Device: {inference.device}")
    else:
        print("\n⚠️ Model not available. Train the model first.")
