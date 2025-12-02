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


class DataCollector:
    """Collects and labels ASL gesture data for training"""
    
    def __init__(self, data_dir: str = "data/gestures"):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.hand_detector = HandDetector()
        
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
    
    def collect_data_interactive(self):
        """
        Interactive data collection using webcam
        """
        print("ASL Gesture Data Collector")
        print("=" * 50)
        print("\nControls:")
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
            
            # Detect hands
            annotated_frame, hand_landmarks = self.hand_detector.detect_hands(frame)
            
            # Apply background blur if enabled and hand detected
            if self.blur_background and hand_landmarks:
                annotated_frame = self._blur_background(annotated_frame, hand_landmarks)
            
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
                print(f"\nSwitched to gesture: {self.current_gesture}")
            elif key == ord(' '):
                # Capture sample
                if hand_landmarks:
                    self._save_sample(hand_landmarks[0], frame)
                    self.samples_collected += 1
                    print(f"Captured sample {self.samples_collected}/{self.target_samples} for '{self.current_gesture}'")
                else:
                    print("No hand detected! Please position hand in frame.")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()
        
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
        
        # Instructions at bottom
        cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
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
