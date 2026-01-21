# ASL-Translator/tests/test_training.py

import numpy as np
from services.training.src.model_trainer import ModelTrainer

def test_model_trainer_can_train_on_small_dataset():
    # Small synthetic dataset
    X = np.random.rand(20, 5)
    y = np.array(["A"] * 10 + ["B"] * 10)

    trainer = ModelTrainer(model_type="random_forest", model_name="test_model")
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)

    trainer.train_model(X_train, y_train)
    metrics = trainer.evaluate_model(X_test, y_test)

    # Basic assertion: accuracy is a float between 0 and 1
    assert 0.0 <= metrics["accuracy"] <= 1.0