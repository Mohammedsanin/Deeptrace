import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import AUDIO_MODEL_CONFIG, DEVICE


class AudioDeepfakeDetector(nn.Module):
    """CNN-based audio deepfake detection model"""
    
    def __init__(self, config=AUDIO_MODEL_CONFIG):
        super(AudioDeepfakeDetector, self).__init__()
        
        self.input_channels = config["input_channels"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(128, self.num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def save_model(model, path):
    """Save model to disk"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': AUDIO_MODEL_CONFIG
    }, path)
    print(f"Model saved to {path}")


def load_model(path, device=DEVICE):
    """Load model from disk"""
    checkpoint = torch.load(path, map_location=device)
    
    model = AudioDeepfakeDetector(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    # Test model
    print("Testing Audio Deepfake Detector...")
    
    model = AudioDeepfakeDetector()
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_steps = 125  # 4 seconds at 16kHz with hop_length=512
    
    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0]}")
    
    # Test probabilities
    probs = F.softmax(output, dim=1)
    print(f"Probabilities: {probs[0]}")
    
    print("\n✓ Model test successful!")
