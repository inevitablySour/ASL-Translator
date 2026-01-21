"""
Download pre-trained ASL gesture recognition model
"""
import urllib.request
import os
from pathlib import Path


def download_file(url: str, destination: str):
    """Download a file from URL to destination"""
    print(f"Downloading from {url}...")
    print(f"Saving to {destination}...")
    
    # Create directory if it doesn't exist
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100)
            print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    urllib.request.urlretrieve(url, destination, reporthook)
    print("\nDownload complete!")


def main():
    """Download pre-trained ASL models"""
    print("=" * 60)
    print("Downloading Pre-trained ASL Recognition Model")
    print("=" * 60)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # MediaPipe Gesture Recognizer model
    # This is a general hand gesture model from MediaPipe
    model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    model_path = models_dir / "gesture_recognizer.task"
    
    if model_path.exists():
        print(f"\nModel already exists at {model_path}")
        overwrite = input("Overwrite? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Skipping download.")
            return
    
    try:
        download_file(model_url, str(model_path))
        print(f"\n✓ Model downloaded successfully to: {model_path}")
        print("\nNote: This is MediaPipe's general gesture recognizer.")
        print("For ASL-specific recognition, you may need to fine-tune or use")
        print("the custom classifier with your training data.")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nAlternative: You can manually download from:")
        print(model_url)
        print(f"And save it to: {model_path}")


if __name__ == "__main__":
    main()
