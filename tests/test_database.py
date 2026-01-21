# ASL-Translator/tests/test_database.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'services' / 'api' / 'src'))

from database import User, Model, Prediction, Feedback, TrainingSample, TrainingRun
from datetime import datetime
import pytest


def test_user_creation():
    """Test User model can be instantiated"""
    user = User(
        username="test_user",
        email="test@example.com",
        created_at=datetime.utcnow()
    )
    assert user.username == "test_user"
    assert user.email == "test@example.com"


def test_model_creation():
    """Test Model model can be instantiated"""
    model = Model(
        name="test_model",
        version="1.0",
        file_path="/models/test_model.pkl",
        accuracy=0.95,
        is_active=True,
        created_at=datetime.utcnow()
    )
    assert model.name == "test_model"
    assert model.accuracy == 0.95
    assert model.is_active is True


def test_prediction_creation():
    """Test Prediction model can be instantiated"""
    prediction = Prediction(
        job_id="test-job-123",
        gesture="A",
        translation="A",
        confidence=0.89,
        language="en",
        processing_time_ms=45.2,
        created_at=datetime.utcnow()
    )
    assert prediction.job_id == "test-job-123"
    assert prediction.gesture == "A"
    assert prediction.confidence == 0.89


def test_feedback_creation():
    """Test Feedback model can be instantiated"""
    feedback = Feedback(
        prediction_id=1,
        accepted=True,
        corrected_gesture=None,
        created_at=datetime.utcnow()
    )
    assert feedback.prediction_id == 1
    assert feedback.accepted is True


def test_training_sample_creation():
    """Test TrainingSample model can be instantiated"""
    sample = TrainingSample(
        gesture="B",
        landmarks=[0.1] * 63,
        source="user_feedback",
        created_at=datetime.utcnow()
    )
    assert sample.gesture == "B"
    assert len(sample.landmarks) == 63
    assert sample.source == "user_feedback"


def test_training_run_creation():
    """Test TrainingRun model can be instantiated"""
    run = TrainingRun(
        model_id="test-model-id",
        samples_used=500,
        feedback_samples=50,
        status="completed",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )
    assert run.model_id == "test-model-id"
    assert run.samples_used == 500
    assert run.feedback_samples == 50
    assert run.status == "completed"
