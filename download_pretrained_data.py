"""
Download pre-trained ASL datasets and models from Kaggle
"""
import os
import subprocess
import sys
from pathlib import Path


def check_kaggle_setup():
    """Check if Kaggle API is set up"""
    try:
        import kaggle
        print("✓ Kaggle API found")
        return True
    except ImportError:
        print("✗ Kaggle API not installed")
        print("\nInstalling kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✓ Kaggle installed")
        return True


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n" + "=" * 60)
        print("Kaggle API credentials not found!")
        print("=" * 60)
        print("\nTo download datasets from Kaggle, you need to:")
        print("\n1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to https://www.kaggle.com/settings/account")
        print("3. Scroll to 'API' section and click 'Create New Token'")
        print("4. This downloads 'kaggle.json' to your Downloads folder")
        print(f"5. Move it to: {kaggle_json.parent}")
        print("\n   On Windows:")
        print(f"   mkdir {kaggle_json.parent}")
        print(f"   move Downloads\\kaggle.json {kaggle_json}")
        print("\n6. Run this script again")
        print("\n" + "=" * 60)
        return False
    
    print(f"✓ Kaggle credentials found at {kaggle_json}")
    return True


def download_mediapipe_asl_dataset():
    """Download pre-processed MediaPipe ASL dataset from Kaggle"""
    print("\n" + "=" * 60)
    print("Downloading MediaPipe ASL Dataset (A-Z)")
    print("=" * 60)
    print("\nDataset: ASL Mediapipe Landmarked Dataset")
    print("Author: Granth Gaurav")
    print("Size: ~200MB (preprocessed landmarks)")
    print("\nThis dataset contains:")
    print("- All 26 ASL alphabet letters (A-Z)")
    print("- Pre-extracted MediaPipe hand landmarks")
    print("- Ready for training without preprocessing")
    
    data_dir = Path("data/pretrained")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import kaggle
        
        # Download dataset
        dataset_name = "granthgaurav/asl-mediapipe-converted-dataset"
        
        print(f"\nDownloading from Kaggle: {dataset_name}")
        print("This may take a few minutes...")
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        print(f"\n✓ Dataset downloaded to: {data_dir}")
        print("\nDataset structure:")
        for item in data_dir.iterdir():
            print(f"  - {item.name}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTry downloading manually:")
        print(f"https://www.kaggle.com/datasets/{dataset_name}")
        return False


def download_alternative_dataset():
    """Download alternative ASL dataset"""
    print("\n" + "=" * 60)
    print("Alternative: Download Kaggle ASL Signs Competition Data")
    print("=" * 60)
    
    print("\nYou can also try the Google ASL competition datasets:")
    print("\n1. ASL Signs (Isolated signs):")
    print("   kaggle competitions download -c asl-signs")
    print("\n2. ASL Fingerspelling:")
    print("   kaggle competitions download -c asl-fingerspelling")
    print("\nThese are larger datasets with video data.")


def create_sample_data():
    """Create sample training data for testing"""
    print("\n" + "=" * 60)
    print("Creating Sample Training Data")
    print("=" * 60)
    
    print("\nNo pre-trained data available. Options:")
    print("\n1. Use the data collector to create your own:")
    print("   .\\collect_data.bat")
    print("\n2. Download from Kaggle (requires Kaggle API setup)")
    print("\n3. Use the heuristic classifier (already working)")


def main():
    """Main function"""
    print("=" * 60)
    print("ASL Translator - Pre-trained Data Downloader")
    print("=" * 60)
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        return
    
    if not check_kaggle_credentials():
        return
    
    print("\n📦 Available Options:")
    print("\n1. Download MediaPipe ASL Dataset (A-Z) [Recommended]")
    print("2. View alternative datasets")
    print("3. Create sample data instructions")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        success = download_mediapipe_asl_dataset()
        if success:
            print("\n✓ Download complete!")
            print("\nNext steps:")
            print("1. Run: .\\train_model.bat")
            print("2. The trainer will automatically find the downloaded data")
            print("3. After training, run: .\\run.bat")
    
    elif choice == "2":
        download_alternative_dataset()
    
    elif choice == "3":
        create_sample_data()
    
    else:
        print("\nExiting...")


if __name__ == "__main__":
    main()
