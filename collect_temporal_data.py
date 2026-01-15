"""
Helper script to collect temporal gesture sequences for training
"""
import argparse
from src.data_collector import DataCollector


def main():
    parser = argparse.ArgumentParser(
        description="Collect temporal gesture sequences for ASL training"
    )
    
    parser.add_argument(
        '--target-samples',
        type=int,
        default=50,
        help='Number of sequences to collect per gesture (default: 50)'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Target sequence length in frames (default: 30)'
    )
    
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Use manual spacebar control instead of auto-segmentation'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Temporal Gesture Data Collection")
    print("=" * 60)
    print("\nYou're about to collect TEMPORAL (video) gesture sequences.")
    print("This is for dynamic gestures like 'J' and 'Z' that involve motion.")
    print()
    
    if args.manual:
        print("Mode: MANUAL (Spacebar Control)")
        print("Instructions:")
        print("  1. Press SPACE to START recording a gesture sequence")
        print("  2. Perform the gesture (with motion if dynamic)")
        print("  3. Press SPACE again to STOP recording")
        print(f"  4. Each sequence should be ~{args.sequence_length} frames (~1 second)")
        print("  5. Press 'n' to move to the next gesture")
        print("  6. Press 'q' when done")
    else:
        print("Mode: AUTO-SEGMENT (Hands-free)")
        print("Instructions:")
        print("  1. Perform gestures naturally - system auto-detects start/end")
        print("  2. Return hands to REST POSITION (at sides) between gestures")
        print("  3. System starts recording when hands move from rest")
        print("  4. System stops when hands return to rest for 10+ frames")
        print("  5. Press 'r' to reset if segmenter gets stuck")
        print("  6. Press 'n' to move to the next gesture")
        print("  7. Press 'q' when done")
    
    print()
    print("=" * 60)
    input("Press ENTER to start collecting data...")
    
    # Initialize data collector
    collector = DataCollector()
    collector.target_samples = args.target_samples
    collector.target_sequence_length = args.sequence_length
    
    # Show current dataset info
    info = collector.get_dataset_info()
    print("\nCurrent Dataset (static):")
    print(f"Total samples: {info['total_samples']}")
    print()
    
    # Start temporal collection
    collector.collect_data_interactive(
        temporal_mode=True,
        auto_segment=not args.manual  # Auto-segment by default
    )
    
    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train the temporal model:")
    print("     python train_temporal_model_cli.py --epochs 50")
    print()
    print("  2. The trained model will be saved to models/temporal_lstm.pth")
    print()
    print("  3. Restart the API server to use temporal predictions:")
    print("     python -m uvicorn src.main:app --reload")


if __name__ == "__main__":
    main()
