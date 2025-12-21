import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parent))

from config import (
    AUDIO_DATASET_DIR, AUDIO_MODEL_PATH, TRAINING_CONFIG, DEVICE
)
from datasets.audio_preprocessing import (
    AudioPreprocessor, AudioDeepfakeDataset, prepare_asvspoof_dataset
)
from models.audio_model import AudioDeepfakeDetector, save_model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='binary')
    val_recall = recall_score(all_labels, all_preds, average='binary')
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return val_loss, val_acc, val_precision, val_recall, val_f1


def train_audio_model():
    """Main training function for audio model"""
    print("=" * 60)
    print("Audio Deepfake Detection Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not AUDIO_DATASET_DIR.exists():
        print(f"\n❌ Dataset not found at {AUDIO_DATASET_DIR}")
        print("Please run datasets/download_datasets.py first")
        return
    
    # Prepare dataset
    print("\n📂 Loading dataset...")
    audio_files, labels = prepare_asvspoof_dataset(AUDIO_DATASET_DIR, split='train')
    
    if not audio_files:
        print("❌ No audio files found. Please check dataset structure.")
        return
    
    # Create dataset
    preprocessor = AudioPreprocessor()
    dataset = AudioDeepfakeDataset(audio_files, labels, preprocessor)
    
    # Split into train and validation
    val_size = int(TRAINING_CONFIG["validation_split"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print(f"\n🤖 Initializing model on {DEVICE}...")
    model = AudioDeepfakeDetector().to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🏋️ Starting training for {TRAINING_CONFIG['epochs']} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_model(model, AUDIO_MODEL_PATH)
            print(f"✓ Model saved (best val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"Best model saved to: {AUDIO_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_audio_model()
