"""
Predict gestures from video file using trained temporal model
"""
import cv2
import argparse
import numpy as np
from pathlib import Path
from src.multi_hand_body_detector import MultiHandBodyDetector
from src.temporal_gesture_classifier import TemporalGestureClassifier
from src.gesture_segmenter import GestureSegmenter
import subprocess
import json


def detect_video_rotation(video_path: str):
    """
    Detect if video needs rotation based on metadata
    Returns cv2.ROTATE_* code or None
    """
    try:
        # Try using ffprobe to get rotation metadata
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
             '-show_streams', video_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    rotation = stream.get('tags', {}).get('rotate', 0)
                    rotation = int(rotation)
                    if rotation == 90:
                        return cv2.ROTATE_90_CLOCKWISE
                    elif rotation == 180:
                        return cv2.ROTATE_180
                    elif rotation == 270:
                        return cv2.ROTATE_90_COUNTERCLOCKWISE
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        # ffprobe not available or failed
        pass
    
    # Fallback: detect based on aspect ratio
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            height, width = frame.shape[:2]
            # If width > height significantly, it's sideways (need to rotate)
            if width > height * 1.5:  # Landscape ratio
                print(f"Detected sideways video ({width}x{height}), rotating counterclockwise")
                return cv2.ROTATE_90_COUNTERCLOCKWISE
    
    return None


def predict_video(video_path: str, model_path: str = "models/temporal_lstm.pth"):
    """
    Process video and predict gestures automatically
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model
    """
    print("=" * 60)
    print("Video Gesture Prediction")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Model: {model_path}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print(f"Train a model first: python train_temporal_model_cli.py --epochs 50")
        return
    
    # Initialize detector, segmenter, and classifier
    print("Loading model...")
    detector = MultiHandBodyDetector()
    segmenter = GestureSegmenter(
        movement_threshold=0.025,  # Higher threshold - less sensitive
        rest_duration_frames=15,   # Longer rest required (0.5 seconds at 30fps)
        min_gesture_frames=20,     # Minimum 20 frames (0.67 seconds)
        max_gesture_frames=90      # Maximum 90 frames (3 seconds)
    )
    classifier = TemporalGestureClassifier(model_path=model_path)
    
    print(f"Model loaded: {classifier.num_classes} classes")
    print(f"Gestures: {classifier.gesture_labels}")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Detect video orientation and rotation needed
    rotation_code = detect_video_rotation(video_path)
    
    print(f"Video Info:")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print()
    print("Processing video...")
    print()
    
    gestures_found = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation if needed
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
        
        # Detect hands and body
        _, detection_results = detector.detect_hands_and_body(frame)
        
        # Process frame through segmenter
        gesture_complete, gesture_sequence = segmenter.process_frame(
            detection_results, frame
        )
        
        # If gesture was completed, predict it
        if gesture_complete:
            # Extract features from sequence
            sequence_features = []
            for frame_data in gesture_sequence:
                features = detector.extract_features(frame_data['detection_results'])
                sequence_features.append(features)
            
            # Pad or truncate to 30 frames
            target_length = 30
            if len(sequence_features) < target_length:
                # Pad with last frame
                while len(sequence_features) < target_length:
                    sequence_features.append(sequence_features[-1])
            elif len(sequence_features) > target_length:
                # Truncate
                sequence_features = sequence_features[:target_length]
            
            # Convert to numpy array and predict
            sequence = np.array(sequence_features, dtype=np.float32)
            gesture, confidence = classifier.predict(sequence)
            
            # Check if two-handed
            two_hand_usage = np.mean(sequence[:, 143])
            gesture_type = "Two-Hand" if two_hand_usage > 0.3 else "Single-Hand"
            
            gestures_found.append({
                'gesture': gesture,
                'confidence': confidence,
                'type': gesture_type,
                'frame_count': len(gesture_sequence),
                'timestamp': frame_count / fps
            })
            
            print(f"\n✓ Detected: {gesture} (confidence={confidence:.2%}, type={gesture_type}, {len(gesture_sequence)} frames)")
    
    cap.release()
    detector.close()
    
    print("\n")
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    
    if not gestures_found:
        print("No gestures detected in video.")
        print()
        print("Tips:")
        print("  - Ensure hands are clearly visible")
        print("  - Return hands to rest position between gestures")
        print("  - Check lighting and camera quality")
        return
    
    print(f"Total gestures detected: {len(gestures_found)}")
    print()
    print("Gesture Sequence:")
    print("-" * 60)
    
    gesture_string = ""
    for i, result in enumerate(gestures_found, 1):
        timestamp_str = f"{result['timestamp']:.1f}s"
        print(f"{i:2d}. [{timestamp_str:>6s}] {result['gesture']:>3s} "
              f"({result['confidence']:>5.1%}) {result['type']:>11s} "
              f"({result['frame_count']:2d} frames)")
        
        if result['gesture'] not in ['UNKNOWN', 'SPACE']:
            gesture_string += result['gesture']
    
    print()
    print(f"Combined sequence: {gesture_string or '(no valid gestures)'}")
    print()
    
    # Statistics
    if gestures_found:
        avg_confidence = np.mean([g['confidence'] for g in gestures_found])
        gesture_counts = {}
        for g in gestures_found:
            gesture_counts[g['gesture']] = gesture_counts.get(g['gesture'], 0) + 1
        
        print("Statistics:")
        print(f"  Average confidence: {avg_confidence:.1%}")
        print(f"  Gesture breakdown:")
        for gesture, count in sorted(gesture_counts.items()):
            percentage = (count / len(gestures_found)) * 100
            print(f"    {gesture}: {count} ({percentage:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Predict gestures from video file using trained model"
    )
    
    parser.add_argument(
        'video',
        help='Path to video file (e.g., recording.mp4, video.mov)'
    )
    
    parser.add_argument(
        '--model',
        default='models/temporal_lstm.pth',
        help='Path to trained model (default: models/temporal_lstm.pth)'
    )
    
    args = parser.parse_args()
    
    predict_video(args.video, args.model)


if __name__ == "__main__":
    main()
