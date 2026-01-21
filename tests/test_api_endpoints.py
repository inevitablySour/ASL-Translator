# ASL-Translator/tests/test_api_endpoints.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'services' / 'api' / 'src'))

import pytest
from unittest.mock import Mock, patch
import base64
import numpy as np
import cv2


def test_prediction_request_model():
    """Test PredictionRequest model validation"""
    try:
        from api import PredictionRequest
        
        # Valid request
        request = PredictionRequest(
            image="base64_encoded_image_data",
            model="test_model"
        )
        assert request.image == "base64_encoded_image_data"
        assert request.model == "test_model"
        
        # Request without model (optional)
        request2 = PredictionRequest(image="data")
        assert request2.model is None
        
    except ImportError as e:
        pytest.skip(f"Could not import API models: {e}")


def test_prediction_response_model():
    """Test PredictionResponse model validation"""
    try:
        from api import PredictionResponse
        
        response = PredictionResponse(
            job_id="test-123",
            gesture="A",
            translation="A",
            confidence=0.89,
            language="en",
            processing_time_ms=45.2
        )
        
        assert response.job_id == "test-123"
        assert response.gesture == "A"
        assert response.confidence == 0.89
        assert 0.0 <= response.confidence <= 1.0
        
    except ImportError as e:
        pytest.skip(f"Could not import API models: {e}")


def test_feedback_request_model():
    """Test FeedbackRequest model validation"""
    try:
        from api import FeedbackRequest
        
        # Accepted feedback
        feedback1 = FeedbackRequest(
            job_id="test-456",
            accepted=True
        )
        assert feedback1.accepted is True
        assert feedback1.corrected_gesture is None
        
        # Corrected feedback
        feedback2 = FeedbackRequest(
            job_id="test-789",
            accepted=False,
            corrected_gesture="B"
        )
        assert feedback2.accepted is False
        assert feedback2.corrected_gesture == "B"
        
    except ImportError as e:
        pytest.skip(f"Could not import API models: {e}")


def test_create_id():
    """Test unique ID generation"""
    try:
        from api import create_ID
        
        id1 = create_ID()
        id2 = create_ID()
        
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0
        
    except ImportError as e:
        pytest.skip(f"Could not import create_ID: {e}")


def test_image_encoding():
    """Test image can be base64 encoded for API"""
    # Create a small test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    assert len(img_base64) > 0
    assert isinstance(img_base64, str)
    
    # Test decoding
    decoded = base64.b64decode(img_base64)
    assert len(decoded) > 0
