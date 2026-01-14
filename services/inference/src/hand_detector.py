"""
Hand detection service using MediaPipe Hands
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
from config import settings


class HandDetector:
    """Detects and extracts hand landmarks from images using MediaPipe"""

    def __init__(self):
        """Initialize MediaPipe Hands solution"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=settings.max_num_hands,
            min_detection_confidence=settings.mediapipe_min_detection_confidence,
            min_tracking_confidence=settings.mediapipe_min_tracking_confidence
        )

    def detect_hands(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], List[dict]]:
        """
        Detect hands in an image and extract landmarks

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            Tuple of (annotated_image, list of hand landmarks)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.hands.process(image_rgb)

        # Prepare output
        annotated_image = image.copy()
        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on image
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmark coordinates
                landmarks_dict = {
                    'landmarks': [
                        {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        }
                        for landmark in hand_landmarks.landmark
                    ]
                }
                hand_landmarks_list.append(landmarks_dict)

        return annotated_image, hand_landmarks_list

    def extract_features(self, hand_landmarks: dict) -> np.ndarray:
        """
        Extract feature vector from hand landmarks

        Args:
            hand_landmarks: Dictionary containing landmark coordinates

        Returns:
            Feature vector as numpy array (63 features: 21 landmarks * 3 coordinates)
        """
        landmarks = hand_landmarks['landmarks']
        features = []

        for landmark in landmarks:
            features.extend([landmark['x'], landmark['y'], landmark['z']])

        return np.array(features, dtype=np.float32)

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position (landmark 0)

        Args:
            landmarks: Array of shape (63,) representing 21 landmarks

        Returns:
            Normalized landmarks
        """
        # Reshape to (21, 3)
        landmarks_reshaped = landmarks.reshape(21, 3)

        # Get wrist position (first landmark)
        wrist = landmarks_reshaped[0].copy()

        # Normalize relative to wrist
        normalized = landmarks_reshaped - wrist

        # Flatten back to (63,)
        return normalized.flatten()

    def close(self):
        """Clean up resources"""
        self.hands.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()