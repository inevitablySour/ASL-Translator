"""
Process video file to automatically extract gesture sequences
Segments gestures based on hand movement and rest position detection
"""
import cv2
import argparse
from pathlib import Path
from src.multi_hand_body_detector import MultiHandBodyDetector
from src.gesture_segmenter import GestureSegmenter
from src.data_collector import DataCollector
import subprocess
import json


def detect_video_rotation(video_path: str):
    """Detect if video needs rotation based on metadata"""
    try:
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
        pass
    
    # Fallback: detect based on aspect ratio
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            height, width = frame.shape[:2]
            if width > height * 1.5:  # Landscape ratio - sideways video
                print(f"Detected sideways video ({width}x{height}), rotating counterclockwise")
                return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def process_video(video_path: str, gesture_name: str, output_dir: str = "data/temporal"):
    """
    Process video and automatically segment gestures
    
    Args:
        video_path: Path to video file
        gesture_name: Name of the gesture being performed (e.g., 'A', 'B')
        output_dir: Directory to save extracted sequences
    """
    print("=" * 60)
    print("Automatic Gesture Segmentation from Video")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Gesture: {gesture_name}")
    print(f"Output: {output_dir}/{gesture_name}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize detector and segmenter
    detector = MultiHandBodyDetector()
    segmenter = GestureSegmenter(
        movement_threshold=0.025,  # Higher threshold - less sensitive
        rest_duration_frames=15,   # Longer rest required (0.5 seconds at 30fps)
        min_gesture_frames=20,     # Minimum 20 frames (0.67 seconds)
        max_gesture_frames=90      # Maximum 90 frames (3 seconds)
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation_code = detect_video_rotation(video_path)
    
    print(f"Video Info:")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print()
    print("Processing video...")
    print()
    
    sequences_found = 0
    frame_count = 0
    
    # Create output directory
    output_path = Path(output_dir) / gesture_name
    output_path.mkdir(parents=True, exist_ok=True)
    
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
        
        # If gesture was completed, save it
        if gesture_complete:
            sequences_found += 1
            save_gesture_sequence(
                gesture_sequence,
                gesture_name,
                output_path,
                sequences_found
            )
            print(f"\n✓ Extracted sequence {sequences_found} ({len(gesture_sequence)} frames)")
    
    cap.release()
    detector.close()
    
    print("\n")
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nExtracted {sequences_found} gesture sequences")
    print(f"Saved to: {output_path}")
    print()
    
    if sequences_found > 0:
        print("Next steps:")
        print("  1. Review the extracted sequences")
        print("  2. Train the model:")
        print("     python train_temporal_model_cli.py --epochs 50")
        print("  3. Restart the server")
    else:
        print("No gestures were detected. Try:")
        print("  - Ensure good lighting")
        print("  - Make clear gestures with pauses between")
        print("  - Check that hands are visible in frame")


def save_gesture_sequence(gesture_sequence, gesture_name, output_path, sequence_num):
    """Save a gesture sequence to disk"""
    from datetime import datetime
    import json
    
    # Create sequence directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sequence_id = f"{gesture_name}_{timestamp}_seq{sequence_num}"
    sequence_dir = output_path / sequence_id
    sequence_dir.mkdir(exist_ok=True)
    
    # Create frames subdirectory
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Save frames and metadata
    sequence_data = []
    
    for idx, frame_data in enumerate(gesture_sequence):
        # Save frame image
        frame = frame_data['frame']
        if frame is not None:
            frame_path = frames_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        
        # Store detection results
        detection_results = frame_data['detection_results']
        sequence_data.append({
            'frame_index': idx,
            'left_hand': detection_results.get('left_hand'),
            'right_hand': detection_results.get('right_hand'),
            'pose': detection_results.get('pose'),
            'spatial_features': detection_results.get('spatial_features', {}),
            'movement': frame_data.get('movement', 0.0),
            'is_resting': frame_data.get('is_resting', False)
        })
    
    # Save metadata
    metadata = {
        'gesture': gesture_name,
        'timestamp': timestamp,
        'sequence_length': len(gesture_sequence),
        'detector_type': 'multi_hand_body',
        'sequence_data': sequence_data
    }
    
    metadata_path = sequence_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract gesture sequences from video automatically"
    )
    
    parser.add_argument(
        'video',
        help='Path to video file (e.g., gestures.mp4)'
    )
    
    parser.add_argument(
        'gesture',
        help='Name of gesture in video (e.g., A, B, C)'
    )
    
    parser.add_argument(
        '--output',
        default='data/temporal',
        help='Output directory for extracted sequences (default: data/temporal)'
    )
    
    args = parser.parse_args()
    
    process_video(args.video, args.gesture, args.output)


if __name__ == "__main__":
    main()
