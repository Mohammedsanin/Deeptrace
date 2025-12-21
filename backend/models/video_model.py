import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import VIDEO_MODEL_CONFIG, DEVICE


class VideoDeepfakeDetector(nn.Module):
    """CNN + LSTM based video deepfake detection model"""
    
    def __init__(self, config=VIDEO_MODEL_CONFIG):
        super(VideoDeepfakeDetector, self).__init__()
        
        self.num_classes = config["num_classes"]
        self.lstm_hidden = config["lstm_hidden"]
        self.lstm_layers = config["lstm_layers"]
        self.dropout = config["dropout"]
        
        # CNN backbone - EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Remove the final classification layer
        self.cnn = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Get feature dimension
        self.feature_dim = 1280  # EfficientNet-B0 output channels
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(self.lstm_hidden * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.lstm_hidden * 2, 128)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(128, self.num_classes)
        
    def forward(self, x):
        # x shape: (batch, time, channels, height, width)
        batch_size, time_steps, c, h, w = x.size()
        
        # Process each frame through CNN
        # Reshape to (batch * time, channels, height, width)
        x = x.view(batch_size * time_steps, c, h, w)
        
        # Extract features
        features = self.cnn(x)  # (batch * time, feature_dim, 1, 1)
        features = features.view(batch_size, time_steps, -1)  # (batch, time, feature_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # (batch, time, lstm_hidden * 2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, time, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_hidden * 2)
        
        # Classification
        x = self.fc1(attended)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def save_model(model, path):
    """Save model to disk"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': VIDEO_MODEL_CONFIG
    }, path)
    print(f"Model saved to {path}")


def load_model(path, device=DEVICE):
    """Load model from disk"""
    checkpoint = torch.load(path, map_location=device)
    
    model = VideoDeepfakeDetector(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    # Test model
    print("Testing Video Deepfake Detector...")
    
    model = VideoDeepfakeDetector()
    print(f"\nModel created successfully")
    
    # Test forward pass
    batch_size = 2
    time_steps = 30
    channels = 3
    height = 224
    width = 224
    
    dummy_input = torch.randn(batch_size, time_steps, channels, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print("Running forward pass...")
    
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0]}")
    
    # Test probabilities
    probs = F.softmax(output, dim=1)
    print(f"Probabilities: {probs[0]}")
    
    print("\n✓ Model test successful!")
