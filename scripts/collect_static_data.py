"""
Interactive script to collect static ASL gesture data using webcam.
Press SPACE to capture a gesture pose, and type custom gesture names.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import json
import os
from datetime import datetime

# Add project paths (scripts/ is one level below project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "inference" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api" / "src"))

try:
    from hand_detector import HandDetector
except ImportError:
    try:
        from services.inference.src.hand_detector import HandDetector
    except ImportError:
        print("Error: Could not import HandDetector.")
        print("Make sure mediapipe is installed: pip install mediapipe")
        sys.exit(1)

try:
    from database import init_db, get_session, TrainingSample
    DB_AVAILABLE = True
except ImportError:
    print("Warning: Could not import database module. Will save to JSON files only.")
    DB_AVAILABLE = False


class StaticGestureCollector:
    def __init__(self, data_dir="data/gestures", target_samples=100, use_database=True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.target_samples = target_samples
        self.use_database = use_database and DB_AVAILABLE
        
        # Initialize database if available
        if self.use_database:
            self.db_engine = init_db()
            print("✓ Database initialized - samples will be saved to database")
        else:
            self.db_engine = None
            print("⚠ Saving to JSON files only (database not available)")
        
        self.hand_detector = HandDetector()
        self.current_gesture = None
        self.samples_collected = 0
        self.window_name = "ASL Static Gesture Collection"
        
        # UI state
        self.recording = False
        self.instruction_text = ""
        self.input_mode = False  # True when waiting for gesture name input
        self.input_text = ""  # Current input text
        self.suggest_next = None  # Suggested next gesture
        self.last_key = -1  # Track last key to prevent repeats
        self.key_debounce_frames = 0  # Debounce counter
        
    def get_dataset_info(self):
        """Get info about current dataset"""
        info = {
            'gestures': [],
            'total_samples': 0,
            'samples_per_gesture': {}
        }
        
        # Query database if available
        if self.use_database:
            try:
                from sqlalchemy import func
                session = get_session(self.db_engine)
                
                # Get count per gesture from database
                results = session.query(
                    TrainingSample.gesture,
                    func.count(TrainingSample.id)
                ).group_by(TrainingSample.gesture).all()
                
                for gesture_name, count in results:
                    info['gestures'].append(gesture_name)
                    info['samples_per_gesture'][gesture_name] = count
                    info['total_samples'] += count
                
                session.close()
            except Exception as e:
                print(f"Warning: Failed to query database: {e}")
                print("Falling back to JSON file counts...")
                # Fall through to file-based counting
        
        # Fallback to JSON files if database not available or query failed
        if not self.use_database or info['total_samples'] == 0:
            if self.data_dir.exists():
                for gesture_dir in self.data_dir.iterdir():
                    if gesture_dir.is_dir():
                        gesture_name = gesture_dir.name
                        json_files = list(gesture_dir.glob("*.json"))
                        count = len(json_files)
                        if gesture_name not in info['gestures']:
                            info['gestures'].append(gesture_name)
                        info['samples_per_gesture'][gesture_name] = count
                        info['total_samples'] += count
        
        info['gestures'].sort()
        return info
    
    def save_gesture_sample(self, gesture_name, hand_landmarks):
        """Save hand landmarks to database and/or JSON"""
        # Convert landmarks to list format (from dict format returned by HandDetector)
        landmarks_list = []
        if hand_landmarks and isinstance(hand_landmarks, dict):
            for landmark in hand_landmarks.get('landmarks', []):
                landmarks_list.extend([landmark['x'], landmark['y'], landmark['z']])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save to database if available
        if self.use_database:
            try:
                session = get_session(self.db_engine)
                sample = TrainingSample(
                    gesture=gesture_name,
                    landmarks=landmarks_list,
                    source='original',
                    collection_date=datetime.now(),
                    sample_metadata={'timestamp': timestamp}
                )
                session.add(sample)
                session.commit()
                session.close()
            except Exception as e:
                print(f"Warning: Failed to save to database: {e}")
                # Fall through to save JSON as backup
        
        # Always save JSON as backup
        gesture_dir = self.data_dir / gesture_name
        gesture_dir.mkdir(exist_ok=True)
        filename = gesture_dir / f"{gesture_name}_{timestamp}.json"
        
        data = {
            'gesture': gesture_name,
            'landmarks': landmarks_list,
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        self.samples_collected += 1
        return filename
    
    def draw_ui(self, frame, info_text=""):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # If in input mode, show gesture name input dialog
        if self.input_mode:
            return self.draw_input_dialog(frame)
        
        # Semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw text
        y_pos = 25
        line_height = 25
        
        # Title
        cv2.putText(frame, "ASL Static Gesture Collection", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_height
        
        # Current gesture and count
        if self.current_gesture:
            text = f"Gesture: {self.current_gesture} | Collected: {self.samples_collected}/{self.target_samples}"
            color = (0, 255, 0) if self.samples_collected < self.target_samples else (0, 165, 255)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "Press 'N' to select a gesture", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y_pos += line_height
        
        # Progress bar
        if self.current_gesture:
            bar_width = 300
            bar_height = 20
            bar_x = 10
            bar_y = y_pos + 5
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Progress fill
            progress = min(self.samples_collected / self.target_samples, 1.0)
            fill_width = int(bar_width * progress)
            color = (0, 255, 0) if progress < 1.0 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            
            # Progress text
            cv2.putText(frame, f"{self.samples_collected}/{self.target_samples}", 
                       (bar_x + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions at bottom
        instructions = [
            "SPACE: Capture gesture | N: Next gesture | Q: Quit | R: Reset counter"
        ]
        
        y_pos = h - 40
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos -= 25
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (w - 50, 50), 15, (0, 0, 255), -1)
            cv2.putText(frame, "CAPTURING", (w - 130, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def draw_input_dialog(self, frame):
        """Draw gesture name input dialog on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Dialog box
        dialog_width = 700
        dialog_height = 350
        dialog_x = (w - dialog_width) // 2
        dialog_y = (h - dialog_height) // 2
        
        cv2.rectangle(frame, (dialog_x, dialog_y), 
                     (dialog_x + dialog_width, dialog_y + dialog_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (dialog_x, dialog_y), 
                     (dialog_x + dialog_width, dialog_y + dialog_height),
                     (0, 255, 0), 3)
        
        # Title
        y_pos = dialog_y + 50
        cv2.putText(frame, "Enter Gesture Name", 
                   (dialog_x + 150, y_pos),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        
        # Input field
        y_pos += 70
        cv2.rectangle(frame, (dialog_x + 50, y_pos), 
                     (dialog_x + dialog_width - 50, y_pos + 60),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (dialog_x + 50, y_pos), 
                     (dialog_x + dialog_width - 50, y_pos + 60),
                     (0, 255, 0), 2)
        
        # Input text - larger font for better visibility
        cv2.putText(frame, self.input_text, 
                   (dialog_x + 70, y_pos + 45),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        
        # Cursor
        cursor_x = dialog_x + 70 + len(self.input_text) * 30
        cv2.line(frame, (cursor_x, y_pos + 15), (cursor_x, y_pos + 50), (255, 255, 255), 2)
        
        # Instructions
        y_pos += 100
        cv2.putText(frame, "Type gesture name, then press ENTER", 
                   (dialog_x + 70, y_pos),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)
        
        # Additional instructions
        y_pos += 35
        cv2.putText(frame, "Press ESC to cancel", 
                   (dialog_x + 70, y_pos),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main collection loop"""
        print("=" * 60)
        print("ASL Static Gesture Data Collection")
        print("=" * 60)
        print("\nInstructions:")
        print("  1. Press 'N' to select a gesture (dialog will appear)")
        print("  2. Type gesture name or press SPACE for suggested name")
        print("  3. Position your hand in front of the camera")
        print("  4. Press SPACE to capture the static gesture pose")
        print("  5. Repeat for ~50-100 samples per gesture")
        print("  6. Press 'N' to move to the next gesture")
        print("  7. Press 'Q' to quit and save all data")
        print("\n" + "=" * 60)
        print()
        
        # Show current dataset
        info = self.get_dataset_info()
        print("Current Dataset:")
        print(f"  Total samples: {info['total_samples']}")
        if info['samples_per_gesture']:
            print("  Samples per gesture:")
            for gesture, count in sorted(info['samples_per_gesture'].items()):
                print(f"    {gesture}: {count}")
        print()
        
        # Create resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam opened. Press 'N' to start or 'Q' to quit.\n")
        
        with self.hand_detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for selfie view
                frame = cv2.flip(frame, 1)
                
                # Detect hands if not in input mode
                if not self.input_mode:
                    frame, hands = self.hand_detector.detect_hands(frame)
                else:
                    hands = []
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle key presses with debouncing
                key = cv2.waitKey(1) & 0xFF
                
                # Debounce: ignore repeated keys within 3 frames
                if key == self.last_key and self.key_debounce_frames < 3:
                    self.key_debounce_frames += 1
                    key = -1  # Ignore this key press
                else:
                    self.last_key = key
                    self.key_debounce_frames = 0
                
                if self.input_mode:
                    # In input mode, handle text input
                    if key == 13:  # ENTER - confirm input
                        if self.input_text.strip():
                            self.current_gesture = self.input_text.strip()
                            self.input_mode = False
                            self.input_text = ""
                            
                            # Load existing samples count
                            self.samples_collected = 0
                            info = self.get_dataset_info()
                            existing_count = info['samples_per_gesture'].get(self.current_gesture, 0)
                            self.samples_collected = existing_count
                            
                            print(f"\n✓ Switched to gesture: {self.current_gesture}")
                            print(f"Already collected: {self.samples_collected} samples")
                            print(f"Target: {self.target_samples} samples\n")
                    
                    elif key == 27:  # ESC - cancel input
                        self.input_mode = False
                        self.input_text = ""
                    
                    elif key == -1 or key == 255:
                        # No key pressed (-1) or special key (255), skip
                        pass
                    
                    elif (65 <= key <= 90):  # Uppercase A-Z
                        self.input_text += chr(key)
                    
                    elif (97 <= key <= 122):  # Lowercase a-z
                        self.input_text += chr(key).upper()  # Convert to uppercase
                    
                    elif (48 <= key <= 57):  # Numbers 0-9
                        self.input_text += chr(key)
                    
                    elif key in [95, 45]:  # Underscore and hyphen
                        self.input_text += chr(key)
                    
                    elif key == 8:  # Explicit backspace only
                        self.input_text = self.input_text[:-1]
                
                else:
                    # Normal mode key handling
                    if key == ord('q'):  # Quit
                        print("\nExiting...")
                        break
                    
                    elif key == ord('n'):  # Next gesture
                        # Calculate suggested next gesture
                        if self.current_gesture is None:
                            self.suggest_next = 'A'
                        else:
                            next_ord = ord(self.current_gesture) + 1
                            if next_ord <= ord('Z'):
                                self.suggest_next = chr(next_ord)
                            else:
                                self.suggest_next = None
                        
                        self.input_mode = True
                        self.input_text = ""
                    
                    elif key == ord(' '):  # Spacebar - capture gesture
                        if self.current_gesture is None:
                            print("Please select a gesture first (press N)")
                            continue
                        
                        if hands and len(hands) > 0:
                            hand = hands[0]
                            filename = self.save_gesture_sample(self.current_gesture, hand)
                            print(f"✓ Captured {self.current_gesture} ({self.samples_collected}/{self.target_samples})")
                            
                            if self.samples_collected >= self.target_samples:
                                print(f"✓ Target reached for {self.current_gesture}! Press N to continue to next gesture.")
                        else:
                            print("✗ No hand detected. Try again with your hand visible.")
                    
                    elif key == ord('r'):  # Reset counter
                        if self.current_gesture:
                            self.samples_collected = 0
                            print(f"\n✓ Counter reset for {self.current_gesture}\n")
                        else:
                            print("Please select a gesture first (press N)")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("Data Collection Complete!")
        print("=" * 60)
        
        final_info = self.get_dataset_info()
        print("\nFinal Dataset Summary:")
        print(f"  Total samples: {final_info['total_samples']}")
        print("  Samples per gesture:")
        for gesture in sorted(final_info['gestures']):
            count = final_info['samples_per_gesture'][gesture]
            print(f"    {gesture}: {count}")
        
        print("\nNext steps:")
        print("  1. Preprocess the data (if using images):")
        print("     python scripts/preprocess_dataset.py --input-dir <images> --output-dir data/gestures")
        print("\n  2. Train a model:")
        print("     python scripts/train_model_cli.py --model-name my_model --data-dir data/gestures")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect static ASL gesture data using webcam"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/gestures',
        help='Directory to save gesture data (default: data/gestures)'
    )
    
    parser.add_argument(
        '--target-samples',
        type=int,
        default=100,
        help='Target number of samples per gesture (default: 100)'
    )
    
    args = parser.parse_args()
    
    collector = StaticGestureCollector(
        data_dir=args.data_dir,
        target_samples=args.target_samples
    )
    
    collector.run()


if __name__ == "__main__":
    main()
