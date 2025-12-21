"""
Complete dataset migration to D: drive
- Clears C: drive Kaggle cache
- Downloads fresh to D: drive
- Extracts to D: drive
"""
import os
import sys
import shutil
from pathlib import Path

# Set environment variables BEFORE importing kagglehub
os.environ['KAGGLE_CACHE_DIR'] = 'D:/VoiceGuard_Data/kaggle_cache'
os.environ['HF_HOME'] = 'D:/VoiceGuard_Data/huggingface_cache'
os.environ['KAGGLE_DATA_DIR'] = 'D:/VoiceGuard_Data/kaggle_data'

sys.path.append(str(Path(__file__).parent.parent))

import kagglehub
from datasets import load_dataset
from config import AUDIO_DATASET_DIR, VIDEO_DATASET_DIR, DATA_DIR


def clean_c_drive_cache():
    """Remove Kaggle cache from C: drive"""
    print("=" * 60)
    print("Cleaning C: Drive Cache...")
    print("=" * 60)
    
    c_cache_paths = [
        Path.home() / ".cache" / "kagglehub",
        Path.home() / ".cache" / "huggingface",
        Path("C:/Users/USER/.cache/kagglehub"),
        Path("C:/Users/USER/.cache/huggingface"),
    ]
    
    total_freed = 0
    
    for cache_path in c_cache_paths:
        if cache_path.exists():
            try:
                # Calculate size before deletion
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                size_gb = size / (1024**3)
                
                print(f"\nFound cache: {cache_path}")
                print(f"Size: {size_gb:.2f} GB")
                
                # Delete
                print(f"Deleting...")
                shutil.rmtree(cache_path)
                print(f"✓ Deleted {cache_path}")
                
                total_freed += size_gb
            except Exception as e:
                print(f"✗ Error deleting {cache_path}: {e}")
    
    print(f"\n✓ Total space freed on C: drive: {total_freed:.2f} GB")
    return total_freed


def setup_d_drive():
    """Create directory structure on D: drive"""
    print("\n" + "=" * 60)
    print("Setting up D: Drive Structure...")
    print("=" * 60)
    
    directories = [
        Path("D:/VoiceGuard_Data"),
        Path("D:/VoiceGuard_Data/kaggle_cache"),
        Path("D:/VoiceGuard_Data/huggingface_cache"),
        Path("D:/VoiceGuard_Data/asvspoof2019"),
        Path("D:/VoiceGuard_Data/deepfake_videos"),
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    # Check D: drive space
    total, used, free = shutil.disk_usage("D:/")
    print(f"\nD: drive space:")
    print(f"  Total: {total / (1024**3):.2f} GB")
    print(f"  Free: {free / (1024**3):.2f} GB")
    
    if free < 30 * (1024**3):  # Less than 30GB free
        print("\n⚠️  Warning: Less than 30GB free on D: drive")
        print("   Dataset requires ~25GB. Proceed with caution.")
    
    return free > 25 * (1024**3)


def download_audio_to_d():
    """Download ASVspoof 2019 to D: drive"""
    print("\n" + "=" * 60)
    print("Downloading ASVspoof 2019 to D: Drive...")
    print("=" * 60)
    print(f"Cache location: {os.environ['KAGGLE_CACHE_DIR']}")
    print(f"Target: {AUDIO_DATASET_DIR}")
    
    try:
        # Download using kagglehub (will use D: drive cache)
        path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
        print(f"\n✓ Dataset downloaded to: {path}")
        print(f"✓ Cache location: {os.environ['KAGGLE_CACHE_DIR']}")
        
        return path
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def download_video_to_d():
    """Download Deepfake Videos to D: drive"""
    print("\n" + "=" * 60)
    print("Downloading Deepfake Videos to D: Drive...")
    print("=" * 60)
    print(f"Cache location: {os.environ['HF_HOME']}")
    print(f"Target: {VIDEO_DATASET_DIR}")
    
    try:
        # Download using Hugging Face
        ds = load_dataset("UniDataPro/deepfake-videos-dataset", split="train")
        
        # Save to D: drive
        VIDEO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(VIDEO_DATASET_DIR))
        
        print(f"\n✓ Dataset downloaded: {len(ds)} samples")
        print(f"✓ Saved to: {VIDEO_DATASET_DIR}")
        
        return ds
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def verify_migration():
    """Verify everything is on D: drive"""
    print("\n" + "=" * 60)
    print("Verifying Migration...")
    print("=" * 60)
    
    # Check D: drive
    d_cache = Path(os.environ['KAGGLE_CACHE_DIR'])
    print(f"\nD: drive Kaggle cache: {d_cache}")
    print(f"  Exists: {d_cache.exists()}")
    
    if d_cache.exists():
        try:
            size = sum(f.stat().st_size for f in d_cache.rglob('*') if f.is_file())
            print(f"  Size: {size / (1024**3):.2f} GB")
        except:
            pass
    
    # Check C: drive (should be clean)
    c_cache = Path.home() / ".cache" / "kagglehub"
    print(f"\nC: drive Kaggle cache: {c_cache}")
    print(f"  Exists: {c_cache.exists()}")
    
    if c_cache.exists():
        print("  ⚠️  Warning: C: drive cache still exists!")
    else:
        print("  ✓ C: drive cache cleaned")
    
    # Check datasets
    print(f"\nDatasets:")
    print(f"  Audio: {AUDIO_DATASET_DIR.exists()} - {AUDIO_DATASET_DIR}")
    print(f"  Video: {VIDEO_DATASET_DIR.exists()} - {VIDEO_DATASET_DIR}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 VoiceGuard Dataset Migration to D: Drive")
    print("=" * 70)
    
    print("\nThis script will:")
    print("1. Clean up C: drive Kaggle cache (~24GB)")
    print("2. Set up D: drive structure")
    print("3. Download datasets to D: drive")
    print("4. Verify migration")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 1: Clean C: drive
    clean_c_drive_cache()
    
    # Step 2: Setup D: drive
    if not setup_d_drive():
        print("\n❌ Insufficient space on D: drive. Aborting.")
        sys.exit(1)
    
    # Step 3: Download audio dataset
    audio_path = download_audio_to_d()
    
    # Step 4: Download video dataset
    video_ds = download_video_to_d()
    
    # Step 5: Verify
    verify_migration()
    
    print("\n" + "=" * 70)
    print("✅ Migration Complete!")
    print("=" * 70)
    print("\nAll datasets are now on D: drive")
    print("C: drive cache has been cleaned")
    print("\nNext steps:")
    print("  1. Train models: python train_audio.py")
    print("  2. Start backend: python main.py")
