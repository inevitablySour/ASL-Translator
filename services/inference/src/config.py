"""
Configuration management for ASL Translator application
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    
    model_confidence_threshold: float = 0.6

    # Application
    app_name: str = "ASL Translator"
    app_version: str = "1.0.0"

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
    models_dir: str = "/app/models/"
    data_dir: str = "/app/data/"

    # Language
    default_language: str = "english"  # Supported: "english", "dutch"

    # Feedback and Retraining
    feedback_confidence_threshold: float = 0.9  # Min confidence to request feedback
    retraining_sample_threshold: int = 200  # Number of feedback samples before retraining
    database_url: str = "sqlite:///data/feedback.db"  # Feedback database URL

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def validate_production_settings(self):
        """Validate settings for production deployment"""
        if self.is_production:
            if self.debug:
                raise ValueError("Debug mode must be disabled in production")
            if not self.database_url or "sqlite" in self.database_url.lower():
                raise ValueError("Production must use PostgreSQL, not SQLite")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()

# Validate production settings on import
try:
    settings.validate_production_settings()
except ValueError as e:
    if settings.is_production:
        raise
    # In development, just log a warning
    import logging
    logging.warning(f"Production validation warning: {e}")