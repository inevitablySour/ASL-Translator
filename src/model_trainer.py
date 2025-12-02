"""
Model training for ASL gesture classification
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Tuple, Dict

from .data_collector import DataCollector
from .config import settings
from .mlflow_manager import mlflow_manager


class ModelTrainer:
    """Train and evaluate ASL gesture classification models"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model ('random_forest', 'svm', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_path = Path(settings.models_dir)
        self.model_path.mkdir(exist_ok=True)
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray, 
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training
        
        Args:
            features: Feature matrix
            labels: Label vector
            test_size: Proportion of test set
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\nPreparing data...")
        print(f"Total samples: {len(features)}")
        print(f"Feature dimensions: {features.shape[1]}")
        print(f"Unique labels: {len(np.unique(labels))}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, 
            test_size=test_size, 
            random_state=42,
            stratify=y_encoded
        )
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   hyperparameters: Dict = None) -> None:
        """
        Train the classification model
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameters: Optional hyperparameters for model
        """
        print(f"\nTraining {self.model_type} model...")
        
        if hyperparameters is None:
            hyperparameters = {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
        
        # Create model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(**hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        print("Training complete!")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Classification report
        labels = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred, 
            target_names=labels,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
        
        Returns:
            Dictionary with CV scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, n_jobs=-1)
        
        print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist()
        }
    
    def save_model(self, filename: str = "asl_classifier.pkl"):
        """
        Save trained model and preprocessing objects
        
        Args:
            filename: Name of model file
        """
        model_file = self.model_path / filename
        scaler_file = self.model_path / "scaler.pkl"
        encoder_file = self.model_path / "label_encoder.pkl"
        
        # Save model
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.label_encoder, encoder_file)
        
        print(f"\nModel saved to: {model_file}")
        print(f"Scaler saved to: {scaler_file}")
        print(f"Label encoder saved to: {encoder_file}")
    
    def train_with_mlflow(self, X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         hyperparameters: Dict = None):
        """
        Train model with MLflow tracking
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            hyperparameters: Model hyperparameters
        """
        with mlflow_manager.start_run(run_name=f"asl_training_{self.model_type}"):
            # Log parameters
            if hyperparameters:
                mlflow_manager.log_params(hyperparameters)
            
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.log_param("num_classes", len(np.unique(y_train)))
            
            # Train model
            self.train_model(X_train, y_train, hyperparameters)
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["classification_report"]["weighted avg"]["precision"])
            mlflow.log_metric("recall", metrics["classification_report"]["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", metrics["classification_report"]["weighted avg"]["f1-score"])
            
            # Save model
            self.save_model()
            
            # Log model to MLflow
            mlflow.sklearn.log_model(self.model, "model")
            
            print("\nMLflow run completed!")
            mlflow_manager.end_run()


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("ASL Gesture Model Training")
    print("=" * 50)
    
    # Load data
    print("\nLoading dataset...")
    collector = DataCollector()
    features, labels = collector.load_dataset()
    
    if len(features) == 0:
        print("Error: No training data found!")
        print("Please run data_collector.py first to collect training data.")
        return
    
    # Show dataset info
    info = collector.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"Total samples: {info['total_samples']}")
    print(f"Gestures: {len(info['gestures'])}")
    
    # Initialize trainer
    trainer = ModelTrainer(model_type="random_forest")
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)
    
    # Hyperparameters
    hyperparameters = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Train with MLflow tracking
    trainer.train_with_mlflow(X_train, X_test, y_train, y_test, hyperparameters)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
