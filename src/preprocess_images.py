"""
Preprocess pretrained image data to extract hand landmarks
"""
import cv2
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_detector import HandDetector
from src.config import settings


class ImagePreprocessor:
    """Preprocess images to extract hand landmarks"""
    
    def __init__(self, input_dir: str = "data/pretrained/Data", 
                 output_dir: str = "data/gestures"):
        """
        Initialize preprocessor
        
        Args:
            input_dir: Directory with pretrained images
            output_dir: Directory to save processed landmarks
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hand_detector = HandDetector()
        
        # ASL alphabet gestures
        self.gestures = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    
    def process_all_gestures(self):
        """Process all gesture directories"""
        print("=" * 60)
        print("Processing Pretrained ASL Image Data")
        print("=" * 60)
        
        total_processed = 0
        total_failed = 0
        
        for gesture in self.gestures:
            gesture_dir = self.input_dir / gesture
            
            if not gesture_dir.exists():
                print(f"\nSkipping {gesture} - directory not found")
                continue
            
            print(f"\nProcessing gesture: {gesture}")
            processed, failed = self.process_gesture(gesture, gesture_dir)
            
            total_processed += processed
            total_failed += failed
            
            print(f"  Processed: {processed}, Failed: {failed}")
        
        print("\n" + "=" * 60)
        print(f"Total processed: {total_processed}")
        print(f"Total failed: {total_failed}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        self.hand_detector.close()
    
    def process_gesture(self, gesture: str, gesture_dir: Path) -> tuple:
        """
        Process all images for a single gesture
        
        Args:
            gesture: Gesture label
            gesture_dir: Directory containing gesture images
        
        Returns:
            Tuple of (processed_count, failed_count)
        """
        # Create output directory
        output_gesture_dir = self.output_dir / gesture
        output_gesture_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = list(gesture_dir.glob("*.jpg")) + list(gesture_dir.glob("*.png"))
        
        processed = 0
        failed = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {gesture}"):
            try:
                # Read image
                image = cv2.imread(str(img_path))
                
                if image is None:
                    failed += 1
                    continue
                
                # Detect hands
                _, hand_landmarks = self.hand_detector.detect_hands(image)
                
                if not hand_landmarks:
                    failed += 1
                    continue
                
                # Save landmarks as JSON
                output_file = output_gesture_dir / f"{img_path.stem}.json"
                data = {
                    "gesture": gesture,
                    "source_image": str(img_path),
                    "landmarks": hand_landmarks[0]
                }
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                processed += 1
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                failed += 1
        
        return processed, failed


def main():
    """Main preprocessing pipeline"""
    preprocessor = ImagePreprocessor(
        input_dir="data/pretrained/Data",
        output_dir="data/gestures"
    )
    preprocessor.process_all_gestures()


if __name__ == "__main__":
    main()
