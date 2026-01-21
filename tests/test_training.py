# ASL-Translator/tests/test_training.py

import sys
from pathlib import Path
import numpy as np
import tempfile
import pytest

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'services' / 'training' / 'src'))

def test_model_trainer_can_train_on_small_dataset():
    """Test that ModelTrainer can train on a small synthetic dataset"""
    from model_trainer import ModelTrainer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Small synthetic dataset with more samples for better training
    np.random.seed(42)
    X = np.random.rand(50, 63)  # 50 samples, 63 features (hand landmarks)
    y = np.array(["A"] * 25 + ["B"] * 25)

    # Use temporary directory instead of /app/models
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trainer and override model_path
        trainer = ModelTrainer(model_type="random_forest", model_name="test_model")
        trainer.model_path = Path(tmpdir) / "test_model"
        trainer.model_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)
        
        # Train model
        trainer.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Assertions
        assert trainer.model is not None, "Model should be trained"
        assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy should be between 0 and 1"
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
