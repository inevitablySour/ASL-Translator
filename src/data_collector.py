"""
Data collection tool for ASL gesture training data
"""
import cv2
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from .hand_detector import HandDetector
from .multi_hand_body_detector import MultiHandBodyDetector
from .gesture_segmenter import GestureSegmenter


class DataCollector:
    """Collects and labels ASL gesture data for training"""
    
    def __init__(self, data_dir: str = "data/gestures", use_multi_hand: bool = True):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
            use_multi_hand: Use multi-hand + body detector (recommended for full ASL support)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use enhanced multi-hand detector by default
        self.use_multi_hand = use_multi_hand
        if use_multi_hand:
            self.detector = MultiHandBodyDetector()
        else:
            self.detector = HandDetector()  # Legacy single-hand support
        
        # ASL alphabet gestures to collect
        self.gestures = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        self.current_gesture = None
        self.samples_collected = 0
        self.target_samples = 100  # Samples per gesture
        self.blur_background = True  # Enable background blurring for privacy
        
        # Temporal data collection settings
        self.recording_mode = False  # Toggle for temporal recording
        self.recording = False
        self.recorded_frames = []
        self.target_sequence_length = 30  # Frames per sequence (~1 second at 30 FPS)
        
        # Automatic gesture segmentation
        self.use_auto_segment = False
        self.segmenter = GestureSegmenter(
            movement_threshold=0.02,
            rest_duration_frames=10,
            min_gesture_frames=15,
            max_gesture_frames=90
        )
    
    def collect_data_interactive(self, temporal_mode: bool = False, auto_segment: bool = True):
        """
        Interactive data collection using webcam
        
        Args:
            temporal_mode: If True, collect video sequences instead of single frames
            auto_segment: If True, automatically detect gesture start/end (no spacebar needed)
        """
        self.recording_mode = temporal_mode
        self.use_auto_segment = auto_segment and temporal_mode
        
        mode_name = "Temporal Sequence" if temporal_mode else "Static Frame"
        if self.use_auto_segment:
            mode_name += " (Auto-Segment)"
        
        print(f"ASL Gesture Data Collector - {mode_name} Mode")
        print("=" * 50)
        print("\nControls:")
        if temporal_mode:
            if self.use_auto_segment:
                print("- Perform gesture naturally, system auto-detects start/end")
                print("- Return hands to rest position (at sides) between gestures")
                print("- Press 'r' to reset segmenter if stuck")
            else:
                print("- Press SPACE to start/stop recording sequence")
        else:
            print("- Press SPACE to capture sample")
        print("- Press 'n' for next gesture")
        print("- Press 'q' to quit")
        print("\n" + "=" * 50 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        gesture_index = 0
        self.current_gesture = self.gestures[gesture_index]
        self.samples_collected = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands (and body if using multi-hand detector)
            if self.use_multi_hand:
                annotated_frame, detection_results = self.detector.detect_hands_and_body(frame)
                # Convert to list format for compatibility with existing code
                hand_landmarks = []
                if detection_results['left_hand']:
                    hand_landmarks.append(detection_results['left_hand'])
                if detection_results['right_hand']:
                    hand_landmarks.append(detection_results['right_hand'])
            else:
                annotated_frame, hand_landmarks = self.detector.detect_hands(frame)
            
            # Apply background blur if enabled and hand detected
            if self.blur_background and hand_landmarks:
                annotated_frame = self._blur_background(annotated_frame, hand_landmarks)
            
            # Automatic gesture segmentation (if enabled)
            if self.use_auto_segment and self.use_multi_hand:
                gesture_complete, gesture_sequence = self.segmenter.process_frame(
                    detection_results, frame
                )
                
                if gesture_complete:
                    # Save the automatically captured sequence
                    self._save_sequence(gesture_sequence)
                    self.samples_collected += 1
                    print(f"✓ Auto-saved sequence {self.samples_collected}/{self.target_samples}")
            
            # Manual recording mode (spacebar control)
            elif self.recording_mode and self.recording:
                # Store full detection results for temporal sequences
                if self.use_multi_hand:
                    self.recorded_frames.append({
                        'detection_results': detection_results,
                        'frame': frame.copy()
                    })
                else:
                    self.recorded_frames.append({
                        'landmarks': hand_landmarks[0] if hand_landmarks else None,
                        'frame': frame.copy()
                    })
            
            # Add UI overlay
            self._draw_ui(annotated_frame, hand_landmarks)
            
            # Show frame
            cv2.imshow('ASL Data Collector', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Next gesture
                gesture_index = (gesture_index + 1) % len(self.gestures)
                self.current_gesture = self.gestures[gesture_index]
                self.samples_collected = 0
                if self.use_auto_segment:
                    self.segmenter.reset()
                print(f"\nSwitched to gesture: {self.current_gesture}")
            elif key == ord('r') and self.use_auto_segment:
                # Reset segmenter
                self.segmenter.reset()
            elif key == ord(' '):
                if self.recording_mode:
                    # Temporal mode - start/stop recording
                    if not self.recording:
                        self.recording = True
                        self.recorded_frames = []
                        print(f"Recording sequence for '{self.current_gesture}'...")
                    else:
                        self.recording = False
                        if len(self.recorded_frames) >= self.target_sequence_length:
                            self._save_sequence(self.recorded_frames)
                            self.samples_collected += 1
                            print(f"Saved sequence {self.samples_collected}/{self.target_samples} ({len(self.recorded_frames)} frames)")
                        else:
                            print(f"Sequence too short ({len(self.recorded_frames)} frames), need at least {self.target_sequence_length}")
                        self.recorded_frames = []
                else:
                    # Static mode - capture single frame
                    if hand_landmarks:
                        self._save_sample(hand_landmarks[0], frame)
                        self.samples_collected += 1
                        print(f"Captured sample {self.samples_collected}/{self.target_samples} for '{self.current_gesture}'")
                    else:
                        print("No hand detected! Please position hand in frame.")
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        
        print("\nData collection complete!")
        print(f"Data saved to: {self.data_dir}")
    
    def _blur_background(self, frame: np.ndarray, hand_landmarks: List) -> np.ndarray:
        """
        Blur background while keeping hand region sharp
        
        Args:
            frame: Input frame
            hand_landmarks: Detected hand landmarks
        
        Returns:
            Frame with blurred background
        """
        if not hand_landmarks:
            return frame
        
        height, width = frame.shape[:2]
        
        # Create mask for hand region
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for hand in hand_landmarks:
            # Get bounding box from landmarks
            landmarks = hand['landmarks']
            
            # Extract x, y coordinates
            x_coords = [int(lm['x'] * width) for lm in landmarks]
            y_coords = [int(lm['y'] * height) for lm in landmarks]
            
            # Get bounding box with padding
            x_min = max(0, min(x_coords) - 50)
            x_max = min(width, max(x_coords) + 50)
            y_min = max(0, min(y_coords) - 50)
            y_max = min(height, max(y_coords) + 50)
            
            # Draw filled rectangle for hand region
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        # Apply Gaussian blur to entire frame
        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        
        # Create smooth mask edges
        mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_3channel = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend original and blurred based on mask
        result = (frame * mask_3channel + blurred * (1 - mask_3channel)).astype(np.uint8)
        
        return result
    
    def _draw_ui(self, frame: np.ndarray, hand_landmarks: List):
        """Draw UI overlay on frame"""
        height, width = frame.shape[:2]
        
        # Display detector mode
        mode_text = "Multi-Hand + Body" if self.use_multi_hand else "Single Hand"
        cv2.putText(frame, f"Mode: {mode_text}", (10, height - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show hand count
        hand_count = len(hand_landmarks) if hand_landmarks else 0
        cv2.putText(frame, f"Hands detected: {hand_count}", (10, height - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 120), (255, 255, 255), 2)
        
        # Current gesture
        cv2.putText(frame, f"Gesture: {self.current_gesture}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Progress
        progress = f"Samples: {self.samples_collected}/{self.target_samples}"
        cv2.putText(frame, progress, 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hand detection status
        status = "Hand Detected" if hand_landmarks else "No Hand"
        color = (0, 255, 0) if hand_landmarks else (0, 0, 255)
        cv2.putText(frame, status, 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Auto-segmentation status or manual recording indicator
        if self.use_auto_segment:
            status = self.segmenter.get_status()
            if status['is_active']:
                # Active gesture - show green indicator
                cv2.circle(frame, (width - 30, 30), 15, (0, 255, 0), -1)
                cv2.putText(frame, f"RECORDING {status['frame_count']}", 
                            (width - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Movement indicator
                movement_bar_width = int(status['avg_movement'] * 200)
                cv2.rectangle(frame, (width - 220, 50), (width - 20, 65), (50, 50, 50), -1)
                cv2.rectangle(frame, (width - 220, 50), (width - 220 + movement_bar_width, 65), (0, 255, 0), -1)
                cv2.putText(frame, "Movement", (width - 220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # Waiting for gesture - show blue indicator
                cv2.circle(frame, (width - 30, 30), 12, (255, 200, 0), -1)
                cv2.putText(frame, "READY", (width - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        elif self.recording_mode and self.recording:
            cv2.circle(frame, (width - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {len(self.recorded_frames)}", 
                        (width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions at bottom
        cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
        if self.use_auto_segment:
            cv2.putText(frame, "AUTO MODE: Perform gesture naturally | R: Reset | N: Next | Q: Quit", 
                        (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif self.recording_mode:
            cv2.putText(frame, "SPACE: Start/Stop Recording | N: Next | Q: Quit", 
                        (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "SPACE: Capture | N: Next | Q: Quit", 
                        (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _save_sample(self, hand_landmarks: Dict, frame: np.ndarray):
        """Save a single sample with blurred background"""
        # Create gesture directory
        gesture_dir = self.data_dir / self.current_gesture
        gesture_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.current_gesture}_{timestamp}"
        
        # Save landmarks as JSON
        landmarks_path = gesture_dir / f"{filename}.json"
        data = {
            "gesture": self.current_gesture,
            "timestamp": timestamp,
            "landmarks": hand_landmarks
        }
        
        with open(landmarks_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Apply background blur before saving image for privacy
        if self.blur_background:
            frame = self._blur_background(frame, [hand_landmarks])
        
        # Save image with blurred background
        image_path = gesture_dir / f"{filename}.jpg"
        cv2.imwrite(str(image_path), frame)
    
    def _save_sequence(self, frames_data: List[Dict]):
        """Save a temporal sequence of frames with multi-hand and body data"""
        # Create temporal data directory
        temporal_dir = self.data_dir.parent / "temporal" / self.current_gesture
        temporal_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sequence ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sequence_id = f"{self.current_gesture}_{timestamp}"
        sequence_dir = temporal_dir / sequence_id
        sequence_dir.mkdir(exist_ok=True)
        
        # Create frames subdirectory
        frames_dir = sequence_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Save each frame and its detection results
        sequence_data = []
        valid_frames = 0
        
        for idx, frame_data in enumerate(frames_data):
            frame = frame_data['frame']
            
            # Handle both old and new format
            if self.use_multi_hand:
                detection_results = frame_data['detection_results']
                
                # Check if we have any hand detected
                has_hand = (detection_results.get('left_hand') is not None or 
                           detection_results.get('right_hand') is not None)
                
                if not has_hand:
                    continue  # Skip frames without any hand
                
                # Save frame image
                frame_path = frames_dir / f"frame_{idx:04d}.jpg"
                
                # Apply background blur if enabled (for hands and body)
                if self.blur_background:
                    hand_landmarks = []
                    if detection_results.get('left_hand'):
                        hand_landmarks.append(detection_results['left_hand'])
                    if detection_results.get('right_hand'):
                        hand_landmarks.append(detection_results['right_hand'])
                    frame = self._blur_background(frame, hand_landmarks)
                
                cv2.imwrite(str(frame_path), frame)
                
                # Store full detection results
                sequence_data.append({
                    'frame_index': idx,
                    'left_hand': detection_results.get('left_hand'),
                    'right_hand': detection_results.get('right_hand'),
                    'pose': detection_results.get('pose'),
                    'spatial_features': detection_results.get('spatial_features', {})
                })
                valid_frames += 1
            else:
                # Legacy single-hand format
                landmarks = frame_data.get('landmarks')
                
                if landmarks is None:
                    continue
                
                # Save frame image
                frame_path = frames_dir / f"frame_{idx:04d}.jpg"
                
                if self.blur_background:
                    frame = self._blur_background(frame, [landmarks])
                
                cv2.imwrite(str(frame_path), frame)
                
                # Store landmarks (legacy format)
                sequence_data.append({
                    'frame_index': idx,
                    'landmarks': landmarks
                })
                valid_frames += 1
        
        # Determine gesture characteristics
        two_hand_frames = 0
        has_body_frames = 0
        if self.use_multi_hand:
            for frame in sequence_data:
                if frame.get('left_hand') and frame.get('right_hand'):
                    two_hand_frames += 1
                if frame.get('pose'):
                    has_body_frames += 1
        
        # Save sequence metadata
        metadata = {
            'gesture': self.current_gesture,
            'timestamp': timestamp,
            'sequence_id': sequence_id,
            'total_frames': len(frames_data),
            'valid_frames': valid_frames,
            'fps': 30,  # Approximate
            'detector_type': 'multi_hand_body' if self.use_multi_hand else 'single_hand',
            'gesture_type': 'dynamic' if self.current_gesture in ['J', 'Z'] else 'static',
            'sequence_data': sequence_data,  # Now includes left/right/pose/spatial data
            'statistics': {
                'two_hand_frames': two_hand_frames,
                'two_hand_ratio': two_hand_frames / valid_frames if valid_frames > 0 else 0,
                'body_detected_frames': has_body_frames,
                'body_detection_ratio': has_body_frames / valid_frames if valid_frames > 0 else 0
            } if self.use_multi_hand else {}
        }
        
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        if self.use_multi_hand:
            two_hand_pct = (two_hand_frames / valid_frames * 100) if valid_frames > 0 else 0
            print(f"  Saved to: {sequence_dir}")
            print(f"  Two-hand frames: {two_hand_frames}/{valid_frames} ({two_hand_pct:.0f}%)")
        else:
            print(f"  Saved to: {sequence_dir}")
    
    def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load collected dataset
        
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        for gesture in self.gestures:
            gesture_dir = self.data_dir / gesture
            
            if not gesture_dir.exists():
                continue
            
            for json_file in gesture_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract features
                landmarks = data['landmarks']['landmarks']
                feature_vector = []
                for lm in landmarks:
                    feature_vector.extend([lm['x'], lm['y'], lm['z']])
                
                features.append(feature_vector)
                labels.append(gesture)
        
        return np.array(features), np.array(labels)
    
    def get_dataset_info(self) -> Dict:
        """Get information about collected dataset"""
        info = {
            "total_samples": 0,
            "gestures": {}
        }
        
        for gesture in self.gestures:
            gesture_dir = self.data_dir / gesture
            
            if gesture_dir.exists():
                count = len(list(gesture_dir.glob("*.json")))
                info["gestures"][gesture] = count
                info["total_samples"] += count
        
        return info


def main():
    """Run data collector"""
    collector = DataCollector()
    
    # Show current dataset info
    info = collector.get_dataset_info()
    print("\nCurrent Dataset:")
    print(f"Total samples: {info['total_samples']}")
    
    if info['gestures']:
        print("\nSamples per gesture:")
        for gesture, count in sorted(info['gestures'].items()):
            print(f"  {gesture}: {count}")
    
    print("\n" + "=" * 50)
    
    # Start collection
    collector.collect_data_interactive()


if __name__ == "__main__":
    main()
