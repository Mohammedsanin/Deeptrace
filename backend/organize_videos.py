"""
Simple video dataset organizer - copies videos from HuggingFace cache to real/fake structure
"""
from pathlib import Path
import shutil

# Paths
HF_CACHE = Path("D:/VoiceGuard_Data/huggingface_cache/hub/datasets--UniDataPro--deepfake-videos-dataset")
OUTPUT_DIR = Path("D:/VoiceGuard_Data/deepfake_videos_organized")

def organize_videos():
    """Organize videos into real/fake directories"""
    print("=" * 60)
    print("Organizing Deepfake Video Dataset")
    print("=" * 60)
    
    # Create output directories
    real_dir = OUTPUT_DIR / "real"
    fake_dir = OUTPUT_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files in cache
    print(f"\nSearching for videos in {HF_CACHE}...")
    
    # Look for videos in the snapshots directory
    video_files = []
    for ext in ['*.mp4', '*.mov', '*.MOV']:
        video_files.extend(HF_CACHE.rglob(ext))
    
    print(f"Found {len(video_files)} video files")
    
    # Organize by directory name (deepfake/ vs video/)
    real_count = 0
    fake_count = 0
    
    for video_path in video_files:
        # Check parent directory name
        parent_dir = video_path.parent.name
        
        if parent_dir == 'deepfake':
            dest_dir = fake_dir
            label = "fake"
            fake_count += 1
        elif parent_dir == 'video':
            dest_dir = real_dir
            label = "real"
            real_count += 1
        else:
            # Skip if not in expected directory
            print(f"⚠️  Skipping {video_path.name} (unexpected parent: {parent_dir})")
            continue
        
        # Copy video
        dest_file = dest_dir / f"{label}_{video_path.stem}{video_path.suffix}"
        shutil.copy2(video_path, dest_file)
        print(f"✓ Copied {video_path.name} -> {label}/")
    
    print("\n" + "=" * 60)
    print("✅ Organization Complete!")
    print(f"Real videos: {real_count}")
    print(f"Fake videos: {fake_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    organize_videos()
