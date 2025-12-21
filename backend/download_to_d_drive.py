"""
Download datasets to D: drive
This script sets up environment variables and downloads datasets to D: drive
"""
import os
import sys
from pathlib import Path

# Set Kaggle cache to D: drive
os.environ['KAGGLE_CACHE_DIR'] = 'D:/VoiceGuard_Data/kaggle_cache'
os.environ['HF_HOME'] = 'D:/VoiceGuard_Data/huggingface_cache'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import kagglehub
from datasets import load_dataset
from config import AUDIO_DATASET_DIR, VIDEO_DATASET_DIR, DATA_DIR


def download_audio_dataset():
    """Download ASVspoof 2019 dataset to D: drive"""
    print("=" * 60)
    print("Downloading ASVspoof 2019 Audio Dataset to D: drive...")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR}")
    print(f"Kaggle cache: {os.environ.get('KAGGLE_CACHE_DIR')}")
    
    try:
        # Create data directory
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download dataset using kagglehub
        print("\nStarting download...")
        path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
        print(f"✓ Dataset downloaded to: {path}")
        
        # The dataset is now in the Kaggle cache on D: drive
        print(f"✓ Dataset cached at: {os.environ.get('KAGGLE_CACHE_DIR')}")
        
        return path
    except Exception as e:
        print(f"✗ Error downloading audio dataset: {e}")
        return None


def download_video_dataset():
    """Download Deepfake Videos dataset to D: drive"""
    print("\n" + "=" * 60)
    print("Downloading Deepfake Videos Dataset to D: drive...")
    print("=" * 60)
    print(f"Target directory: {VIDEO_DATASET_DIR}")
    
    try:
        # Download dataset using Hugging Face datasets
        ds = load_dataset("UniDataPro/deepfake-videos-dataset", split="train")
        
        # Save to D: drive
        VIDEO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(ds)}")
        
        # Save dataset to disk
        ds.save_to_disk(str(VIDEO_DATASET_DIR))
        print(f"✓ Dataset saved to: {VIDEO_DATASET_DIR}")
        
        return ds
    except Exception as e:
        print(f"✗ Error downloading video dataset: {e}")
        return None


def verify_datasets():
    """Verify that datasets are downloaded"""
    print("\n" + "=" * 60)
    print("Verifying Datasets on D: drive...")
    print("=" * 60)
    
    # Check Kaggle cache
    kaggle_cache = Path(os.environ.get('KAGGLE_CACHE_DIR', ''))
    if kaggle_cache.exists():
        print(f"\n✓ Kaggle cache exists: {kaggle_cache}")
        # List contents
        try:
            for item in kaggle_cache.iterdir():
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    print(f"  - {item.name}: {size / (1024**3):.2f} GB")
        except:
            pass
    
    audio_exists = AUDIO_DATASET_DIR.exists()
    video_exists = VIDEO_DATASET_DIR.exists()
    
    print(f"\nAudio Dataset: {'✓ Found' if audio_exists else '✗ Not found'} at {AUDIO_DATASET_DIR}")
    print(f"Video Dataset: {'✓ Found' if video_exists else '✗ Not found'} at {VIDEO_DATASET_DIR}")
    
    # Check D: drive space
    try:
        import shutil
        total, used, free = shutil.disk_usage("D:/")
        print(f"\nD: drive space:")
        print(f"  Total: {total / (1024**3):.2f} GB")
        print(f"  Used: {used / (1024**3):.2f} GB")
        print(f"  Free: {free / (1024**3):.2f} GB")
    except:
        pass
    
    return audio_exists or video_exists


if __name__ == "__main__":
    print("\n🎯 VoiceGuard Dataset Downloader (D: Drive)\n")
    print(f"Data will be stored on D: drive to avoid space issues")
    print(f"Data directory: {DATA_DIR}\n")
    
    # Download audio dataset
    audio_path = download_audio_dataset()
    
    # Download video dataset
    video_ds = download_video_dataset()
    
    # Verify
    if verify_datasets():
        print("\n✅ Datasets downloaded successfully to D: drive!")
    else:
        print("\n⚠️  Check errors above.")
