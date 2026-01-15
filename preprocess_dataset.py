"""
CLI script to preprocess image datasets and extract hand landmarks
Converts images to landmark JSON files suitable for model training
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocess_images import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ASL gesture images to extract hand landmarks'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing image folders (e.g., data/pretrained/Data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save processed landmarks (e.g., data/dataset1_landmarks)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        sys.exit(1)
    
    print("=" * 60)
    print("ASL Image Dataset Preprocessing")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        auto_detect_gestures=True
    )
    
    # Process all gestures
    preprocessor.process_all_gestures()
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Train a model using the processed data:")
    print(f"   python train_model_cli.py --model-name my_model --data-dir {args.output_dir}")
    print(f"\n2. Start the server and select your model from the dropdown")


if __name__ == "__main__":
    main()
