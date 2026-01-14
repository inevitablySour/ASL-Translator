"""
Gesture classification for ASL hand signs
"""
import numpy as np
import joblib
import mediapipe as mp
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict
from config import settings


class GestureClassifier:
    """
    Classifies hand gestures into ASL letters/signs

    Note: This is a starter implementation using a simple rule-based classifier.
    In production, this should be replaced with a trained ML model (CNN, Random Forest, etc.)
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize classifier"""
        # ASL alphabet letters (subset for demonstration)
        self.gesture_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            'SPACE', 'UNKNOWN'
        ]

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_recognizer = None

        # Try to load MediaPipe gesture recognizer
        # Note: MediaPipe's pre-trained model is for general gestures, not ASL alphabet
        # Disabling for now - using heuristic classifier instead
        # mediapipe_model = Path(settings.models_dir) / "gesture_recognizer.task"
        # if mediapipe_model.exists():
        #     self._load_mediapipe_model(str(mediapipe_model))

        # Try to load custom trained model if available
        if model_path is None:
            model_path = Path(settings.models_dir) / "asl_classifier.pkl"

        if Path(model_path).exists():
            self.load_model(model_path)

    def predict(self, features: np.ndarray, image: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Predict gesture from hand landmark features

        Args:
            features: Feature vector from hand landmarks (63 features)
            image: Optional image for MediaPipe recognizer

        Returns:
            Tuple of (predicted_gesture, confidence_score)
        """
        if self.model is not None:
            # Use custom trained model for prediction
            return self._predict_with_model(features)
        elif self.gesture_recognizer is not None and image is not None:
            # Use MediaPipe gesture recognizer
            return self._predict_with_mediapipe(image)
        else:
            # Use simple heuristic classifier as fallback
            return self._predict_heuristic(features)

    def _load_mediapipe_model(self, model_path: str):
        """Load MediaPipe gesture recognizer model"""
        try:
            # Convert to absolute path
            model_path = str(Path(model_path).resolve())

            BaseOptions = mp.tasks.BaseOptions
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE
            )
            self.gesture_recognizer = GestureRecognizer.create_from_options(options)
            print(f"MediaPipe gesture recognizer loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load MediaPipe model: {e}")
            self.gesture_recognizer = None

    def _predict_with_mediapipe(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict using MediaPipe gesture recognizer"""
        try:
            # Convert image to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Recognize gestures
            result = self.gesture_recognizer.recognize(mp_image)

            if result.gestures:
                # Get top gesture
                gesture = result.gestures[0][0]
                return gesture.category_name.upper(), gesture.score
            else:
                return 'UNKNOWN', 0.0

        except Exception as e:
            print(f"MediaPipe prediction error: {e}")
            return 'UNKNOWN', 0.0

    def _predict_with_model(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict using trained ML model

        Args:
            features: Feature vector

        Returns:
            Tuple of (predicted_gesture, confidence_score)
        """
        # Preprocess features with scaler
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        # Decode label
        if self.label_encoder is not None:
            gesture = self.label_encoder.inverse_transform([prediction])[0]
        else:
            gesture = self.gesture_labels[prediction]

        return gesture, float(confidence)

    def _predict_heuristic(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Simple heuristic-based prediction (for demonstration/testing)

        This is a placeholder that uses basic finger position analysis
        In practice, you should train a proper ML model

        Args:
            features: Feature vector (63 values)

        Returns:
            Tuple of (predicted_gesture, confidence_score)
        """
        # Reshape features to (21, 3) for easier analysis
        landmarks = features.reshape(21, 3)

        # Calculate basic hand shape metrics
        # These are simplified heuristics for demonstration

        # Finger tips (indices in MediaPipe hand landmarks)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Palm base points
        wrist = landmarks[0]
        index_base = landmarks[5]

        # Calculate distances (basic gesture detection)
        fingers_extended = self._count_extended_fingers(landmarks)
        hand_is_closed = fingers_extended <= 1

        # Simple gesture mapping (very basic)
        if hand_is_closed:
            gesture = 'A'  # Closed fist typically represents 'A' in ASL
            confidence = 0.7
        elif fingers_extended == 2:
            # Check if index and middle are extended (could be 'V')
            gesture = 'V'
            confidence = 0.6
        elif fingers_extended == 5:
            # All fingers extended (could be 'B' or '5')
            gesture = 'B'
            confidence = 0.6
        else:
            gesture = 'UNKNOWN'
            confidence = 0.3

        return gesture, confidence

    def _count_extended_fingers(self, landmarks: np.ndarray) -> int:
        """
        Count how many fingers are extended

        Args:
            landmarks: Reshaped landmarks (21, 3)

        Returns:
            Number of extended fingers
        """
        extended = 0

        # Finger tip and middle joint indices
        fingers = [
            (4, 3),  # Thumb
            (8, 6),  # Index
            (12, 10),  # Middle
            (16, 14),  # Ring
            (20, 18)  # Pinky
        ]

        wrist_y = landmarks[0][1]

        for tip_idx, mid_idx in fingers:
            tip_y = landmarks[tip_idx][1]
            mid_y = landmarks[mid_idx][1]

            # If tip is higher (lower y value) than middle joint, finger is extended
            if tip_y < mid_y:
                extended += 1

        return extended

    def load_model(self, model_path: str):
        """
        Load a trained model from disk

        Args:
            model_path: Path to saved model file
        """
        try:
            model_path = Path(model_path)
            model_dir = model_path.parent

            # Load model
            self.model = joblib.load(model_path)

            # Load scaler if exists
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            # Load label encoder if exists
            encoder_path = model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                # Update gesture labels from encoder
                self.gesture_labels = list(self.label_encoder.classes_)

            print(f"Model loaded successfully from {model_path}")

        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Falling back to heuristic-based classification")
            self.model = None

    def is_confident(self, confidence: float) -> bool:
        """
        Check if prediction confidence meets threshold

        Args:
            confidence: Confidence score

        Returns:
            True if confidence is above threshold
        """
        return confidence >= settings.classifier_confidence_threshold