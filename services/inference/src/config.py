"""
Configuration management for ASL Translator application
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    model_confidence_threshold: float = 0.6

    # Application
    app_name: str = "ASL Translator"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Model Configuration
    classifier_confidence_threshold: float = 0.6
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    max_num_hands: int = 1

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "asl-gesture-recognition"

    # Paths
    models_dir: str = "models/"
    data_dir: str = "data/"

    # Language
    default_language: str = "english"  # or "dutch"

    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()