"""
MLflow integration for model tracking and versioning
"""
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from datetime import datetime
from .config import settings


class MLflowManager:
    """Manages MLflow experiment tracking and model registry"""
    
    def __init__(self):
        """Initialize MLflow manager"""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        self.experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_inference_metrics(self, prediction_time_ms: float, confidence: float):
        """Log inference metrics"""
        metrics = {
            "inference_time_ms": prediction_time_ms,
            "confidence": confidence
        }
        self.log_metrics(metrics)
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()


mlflow_manager = MLflowManager()
