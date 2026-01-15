"""
CLI script to train temporal LSTM model for dynamic gesture recognition
"""
import argparse
from pathlib import Path
import torch

from src.temporal_gesture_classifier import GestureLSTM
from src.temporal_model_trainer import TemporalModelTrainer
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal LSTM model for ASL gesture recognition"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/temporal',
        help='Directory containing temporal gesture sequences (default: data/temporal)'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Number of frames in each sequence (default: 30)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='LSTM hidden layer size (default: 128)'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of LSTM layers (default: 2)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/temporal_lstm.pth',
        help='Output path for trained model (default: models/temporal_lstm.pth)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to train on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("Temporal Gesture Recognition - Model Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Sequence length: {args.sequence_length} frames")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Device: {device}")
    print(f"  Output: {args.output}")
    print()
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print(f"\nTo collect temporal data, run:")
        print(f"  python collect_temporal_data.py")
        return
    
    # Count available gestures
    gesture_count = sum(1 for d in data_path.iterdir() if d.is_dir())
    if gesture_count == 0:
        print(f"Error: No gesture directories found in {args.data_dir}")
        print(f"\nCollect training data first using:")
        print(f"  python collect_temporal_data.py")
        return
    
    print(f"Found {gesture_count} gesture types in dataset\n")
    
    # Create temporary model (will be updated after loading data)
    temp_model = GestureLSTM(
        input_size=63,  # Temporary - will be updated
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=gesture_count,
        dropout=args.dropout
    )
    
    # Initialize trainer
    trainer = TemporalModelTrainer(temp_model, device=device)
    
    try:
        # Prepare data (this will detect feature size)
        train_loader, test_loader = trainer.prepare_data(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            test_split=args.test_split
        )
        
        # Get detected feature size from loaded data
        sample_batch = next(iter(train_loader))
        detected_feature_size = sample_batch[0].shape[2]  # (batch, seq_len, features)
        num_classes = len(trainer.label_encoder.classes_)
        
        # Create model with correct input size
        print(f"\nCreating model with {detected_feature_size} input features...")
        
        if detected_feature_size == 151:
            # Use spatial-temporal model for multi-hand + body data
            from src.spatial_temporal_classifier import SpatialTemporalLSTM
            model = SpatialTemporalLSTM(
                input_size=151,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_classes=num_classes,
                dropout=args.dropout
            )
            print("  Using Spatial-Temporal LSTM (multi-hand + body support)")
        else:
            # Use standard temporal model for single-hand data
            model = GestureLSTM(
                input_size=detected_feature_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_classes=num_classes,
                dropout=args.dropout
            )
            print(f"  Using standard Temporal LSTM ({detected_feature_size} features)")
        
        print(f"\nFinal Model Architecture:")
        print(f"  Input: (batch_size, {args.sequence_length}, {detected_feature_size})")
        print(f"  LSTM layers: {args.num_layers} × {args.hidden_size} units")
        print(f"  Output classes: {num_classes}")
        print()
        
        # Update trainer with correct model
        trainer.model = model.to(device)
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience
        )
        
        # Evaluate model
        print("\n" + "=" * 60)
        print("Evaluating Model on Test Data")
        print("=" * 60)
        metrics = trainer.evaluate(test_loader)
        
        # Print detailed per-class results
        print("\nPer-Class Performance:")
        print("-" * 60)
        report = metrics['classification_report']
        for gesture_name in trainer.label_encoder.classes_:
            if gesture_name in report:
                class_metrics = report[gesture_name]
                print(f"  {gesture_name}:")
                print(f"    Precision: {class_metrics['precision']:.3f}")
                print(f"    Recall:    {class_metrics['recall']:.3f}")
                print(f"    F1-Score:  {class_metrics['f1-score']:.3f}")
                print(f"    Support:   {int(class_metrics['support'])} samples")
        
        # Print overall metrics
        print("\n" + "-" * 60)
        print(f"Overall Test Accuracy: {metrics['accuracy']:.4f}")
        if 'macro avg' in report:
            macro = report['macro avg']
            print(f"Macro Avg F1-Score:    {macro['f1-score']:.4f}")
        if 'weighted avg' in report:
            weighted = report['weighted avg']
            print(f"Weighted Avg F1-Score: {weighted['f1-score']:.4f}")
        
        # Save model
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(
            save_path=str(output_path),
            metadata={
                'accuracy': metrics['accuracy'],
                'sequence_length': args.sequence_length,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
            }
        )
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Model saved to: {output_path}")
        print(f"\nTo use this model for inference:")
        print(f"  1. Copy {output_path} to models/temporal_lstm.pth")
        print(f"  2. Restart the API server")
        print(f"  3. Use POST /predict/temporal endpoint")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print(f"\nMake sure you have collected temporal training data.")
        print(f"Run: python collect_temporal_data.py")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
