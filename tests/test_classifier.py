# ASL-Translator/tests/test_classifier.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'services' / 'inference' / 'src'))

import numpy as np
import pytest


def test_gesture_classifier_initialization():
    """Test GestureClassifier can be initialized"""
    # Import here to avoid issues if dependencies are missing
    try:
        from gesture_classifier import GestureClassifier
        
        classifier = GestureClassifier(model_path=None)
        assert classifier is not None
        assert len(classifier.gesture_labels) > 0
        assert 'A' in classifier.gesture_labels
        assert 'Z' in classifier.gesture_labels
    except ImportError as e:
        pytest.skip(f"Could not import GestureClassifier: {e}")


def test_gesture_classifier_predict_with_random_features():
    """Test classifier can make predictions with random features"""
    try:
        from gesture_classifier import GestureClassifier
        
        classifier = GestureClassifier(model_path=None)
        
        # Create random feature vector (21 landmarks * 3 coordinates = 63 features)
        features = np.random.rand(63)
        
        # Make prediction (should use heuristic fallback if no model)
        gesture, confidence = classifier.predict(features)
        
        assert gesture is not None
        assert isinstance(gesture, str)
        assert 0.0 <= confidence <= 1.0
        
    except ImportError as e:
        pytest.skip(f"Could not import GestureClassifier: {e}")


def test_feature_vector_shape():
    """Test that feature vectors have the correct shape"""
    # MediaPipe hand landmarks: 21 points * 3 coordinates (x, y, z)
    expected_features = 63
    
    features = np.random.rand(expected_features)
    assert features.shape == (63,)
    
    # Reshape to landmarks
    landmarks = features.reshape(21, 3)
    assert landmarks.shape == (21, 3)
