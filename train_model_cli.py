"""
CLI script for training ASL gesture recognition models
Supports multiple models and custom data directories
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.model_trainer import ModelTrainer
    from src.db_data_loader import DatabaseDataLoader as DataLoader
    DATA_SOURCE = 'database'
except ImportError:
    # Fallback: try old data collector if it exists
    try:
        from src.model_trainer import ModelTrainer
        from src.data_collector import DataCollector as DataLoader
        DATA_SOURCE = 'json'
    except ImportError:
        print("Error: Could not import required modules")
        print("Make sure src/model_trainer.py and src/db_data_loader.py exist")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train ASL Gesture Recognition Model')
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name for the model (will be saved in models/{model_name}/)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/gestures',
        help='Directory containing training data (default: data/gestures)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='random_forest',
        choices=['random_forest'],
        help='Type of model to train (default: random_forest)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of trees in random forest (default: 200)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Max depth of trees (default: 20)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ASL Gesture Model Training")
    print("=" * 60)
    print(f"Model Name: {args.model_name}")
    print(f"Data Source: {DATA_SOURCE}")
    if DATA_SOURCE == 'json':
        print(f"Data Directory: {args.data_dir}")
    print(f"Model Type: {args.model_type}")
    print("=" * 60)
    
    # Load data
    if DATA_SOURCE == 'database':
        print(f"\nLoading dataset from database...")
        loader = DataLoader()
    else:
        print(f"\nLoading dataset from {args.data_dir}...")
        loader = DataLoader(data_dir=args.data_dir)
    
    features, labels = loader.load_dataset()
    
    if len(features) == 0:
        print(f"Error: No training data found in {args.data_dir}!")
        print("Please ensure the data directory contains gesture folders with JSON files.")
        sys.exit(1)
    
    # Show dataset info
    info = loader.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"Total samples: {info['total_samples']}")
    print(f"Gestures: {len(info['gestures'])}")
    
    if DATA_SOURCE == 'database' and 'samples_per_source' in info:
        print(f"\nSamples by source:")
        for source, count in info['samples_per_source'].items():
            print(f"  {source}: {count}")
    
    # Initialize trainer
    trainer = ModelTrainer(model_type=args.model_type, model_name=args.model_name)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, labels, test_size=args.test_size
    )
    
    # Hyperparameters
    hyperparameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Train with MLflow tracking
    trainer.train_with_mlflow(X_train, X_test, y_train, y_test, hyperparameters)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Model saved as: {args.model_name}")
    print("=" * 60)
    print(f"\nTo use this model:")
    print(f"1. Start the server: uvicorn src.main:app --reload")
    print(f"2. Select '{args.model_name}' from the model dropdown in the web interface")


if __name__ == "__main__":
    main()
