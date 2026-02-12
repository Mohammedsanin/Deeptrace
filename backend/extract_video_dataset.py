"""
Extract videos from HuggingFace dataset format to real/fake directory structure
"""
import sys
from pathlib import Path

# Import HuggingFace datasets before local modules to avoid conflicts
try:
    from datasets import load_from_disk as hf_load_from_disk
except ImportError:
    print("Error: HuggingFace datasets library not installed")
    print("Install with: pip install datasets")
    sys.exit(1)

import shutil

# Paths
HF_DATASET_PATH = Path("D:/Deeptrace_Data/deepfake_videos")
OUTPUT_DIR = Path("D:/Deeptrace_Data/deepfake_videos_extracted")

def extract_videos():
    """Extract videos from HuggingFace dataset"""
    print("=" * 60)
    print("Extracting Deepfake Video Dataset")
    print("=" * 60)
    
    # Create output directories
    real_dir = OUTPUT_DIR / "real"
    fake_dir = OUTPUT_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from {HF_DATASET_PATH}...")
    dataset = hf_load_from_disk(str(HF_DATASET_PATH))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Extract videos
    for idx, sample in enumerate(dataset):
        video_path = sample['video']['path']
        label = sample['label']  # Assuming 0=real, 1=fake
        
        # Determine destination
        if label == 0:
            dest_dir = real_dir
            label_name = "real"
        else:
            dest_dir = fake_dir
            label_name = "fake"
        
        # Copy video file
        video_file = Path(video_path)
        if video_file.exists():
            dest_file = dest_dir / f"{label_name}_{idx}{video_file.suffix}"
            shutil.copy2(video_path, dest_file)
            print(f"✓ Copied {video_file.name} -> {dest_file}")
        else:
            print(f"⚠️  Video not found: {video_path}")
    
    print("\n" + "=" * 60)
    print("✅ Extraction Complete!")
    print(f"Real videos: {len(list(real_dir.glob('*')))}")
    print(f"Fake videos: {len(list(fake_dir.glob('*')))}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    extract_videos()
