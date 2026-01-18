#!/usr/bin/env python3
"""
Script to register an existing model in the database
"""
import sys
from pathlib import Path

# Add API src directory to path
# Handle both local and Docker container paths
if Path('/src/src').exists():
    # Running in Docker container
    api_src_path = Path('/src/src')
else:
    # Running locally
    api_src_path = Path(__file__).parent / 'services' / 'api' / 'src'
sys.path.insert(0, str(api_src_path))

from database import get_session, init_db, Model
from datetime import datetime


def register_model(model_dir: str, set_active: bool = False):
    """
    Register an existing model directory in the database
    
    Args:
        model_dir: Path to model directory (e.g., "models/asl_model_20260118_141758")
        set_active: Whether to set this model as active
    """
    model_path = Path(model_dir)
    
    # Check if directory exists
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        return False
    
    # Check if model file exists
    model_file = model_path / "asl_classifier.pkl"
    if not model_file.exists():
        print(f"Error: Model file not found: {model_file}")
        return False
    
    # Extract model name from directory
    model_name = model_path.name
    
    print(f"Registering model: {model_name}")
    print(f"Model path: {model_file}")
    
    # Initialize database
    engine = init_db()
    session = get_session(engine)
    
    # Check if model already exists
    existing_model = session.query(Model).filter_by(version=model_name).first()
    if existing_model:
        print(f"Model {model_name} already exists in database (ID: {existing_model.id})")
        
        if set_active:
            # Deactivate all models
            session.query(Model).update({"is_active": False})
            # Activate this model
            existing_model.is_active = True
            session.commit()
            print(f"Set {model_name} as active model")
        
        session.close()
        return True
    
    # If setting as active, deactivate all other models
    if set_active:
        session.query(Model).update({"is_active": False})
    
    # Create new model record
    model = Model(
        version=model_name,
        name=f"ASL Classifier {model_name}",
        file_path=str(model_file.resolve()),
        accuracy=None,  # Unknown for manually registered models
        is_active=set_active,
        model_metadata={
            "manually_registered": True,
            "registration_date": datetime.utcnow().isoformat()
        }
    )
    
    session.add(model)
    session.commit()
    
    print(f"✓ Model registered successfully!")
    print(f"  ID: {model.id}")
    print(f"  Version: {model.version}")
    print(f"  Active: {model.is_active}")
    
    session.close()
    return True


def list_models():
    """List all registered models"""
    engine = init_db()
    session = get_session(engine)
    
    models = session.query(Model).order_by(Model.created_at.desc()).all()
    
    if not models:
        print("No models registered in database")
        session.close()
        return
    
    print("\nRegistered Models:")
    print("-" * 80)
    for model in models:
        status = "ACTIVE" if model.is_active else "Inactive"
        accuracy = f"{model.accuracy*100:.2f}%" if model.accuracy else "N/A"
        print(f"{status:8} | {model.version:30} | Accuracy: {accuracy:6} | ID: {model.id}")
    print("-" * 80)
    
    session.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Register models in the database')
    parser.add_argument('--list', action='store_true', help='List all registered models')
    parser.add_argument('--register', type=str, help='Register a model directory')
    parser.add_argument('--activate', action='store_true', help='Set registered model as active')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.register:
        register_model(args.register, set_active=args.activate)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
