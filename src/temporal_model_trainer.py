"""
PyTorch training for temporal LSTM gesture classifier
Trains on sequences of hand landmarks for dynamic gesture recognition
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .temporal_gesture_classifier import GestureLSTM
from .config import settings
from .mlflow_manager import mlflow_manager


class TemporalGestureDataset(Dataset):
    """PyTorch Dataset for temporal gesture sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            sequences: Array of shape (num_samples, sequence_length, 63)
            labels: Array of shape (num_samples,) with integer labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TemporalModelTrainer:
    """Trainer for LSTM temporal gesture classifier"""
    
    def __init__(self, model: GestureLSTM, device: str = None):
        """
        Initialize trainer
        
        Args:
            model: LSTM model to train
            device: Device to train on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.label_encoder = LabelEncoder()
        
        print(f"Training on device: {self.device}")
    
    def prepare_data(self, data_dir: str, sequence_length: int = 30, 
                    test_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare temporal gesture data
        
        Args:
            data_dir: Directory containing temporal sequences
            sequence_length: Target sequence length
            test_split: Proportion of data for testing
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        print(f"\nLoading temporal gesture data from {data_dir}...")
        
        data_path = Path(data_dir)
        sequences = []
        labels = []
        
        # Load all gesture sequences
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        for gesture_dir in data_path.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            gesture_name = gesture_dir.name
            print(f"Loading gesture: {gesture_name}")
            
            for sequence_dir in gesture_dir.iterdir():
                if not sequence_dir.is_dir():
                    continue
                
                # Load metadata
                metadata_path = sequence_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Determine data format (old vs new)
                detector_type = metadata.get('detector_type', 'single_hand')
                
                # Extract sequence data
                if 'sequence_data' in metadata:
                    # New format (multi-hand + body)
                    sequence_data = metadata['sequence_data']
                elif 'landmarks_sequence' in metadata:
                    # Old format (single hand)
                    sequence_data = metadata['landmarks_sequence']
                else:
                    print(f"  Skipping {sequence_dir.name}: no sequence data found")
                    continue
                
                if len(sequence_data) < sequence_length:
                    # Skip sequences that are too short
                    continue
                
                # Convert to feature array based on format
                sequence_features = []
                
                if detector_type == 'multi_hand_body':
                    # New format: extract from multi-hand + body data
                    from .multi_hand_body_detector import MultiHandBodyDetector
                    detector = MultiHandBodyDetector()
                    
                    for frame_data in sequence_data[:sequence_length]:
                        # Reconstruct detection results format
                        detection_results = {
                            'left_hand': frame_data.get('left_hand'),
                            'right_hand': frame_data.get('right_hand'),
                            'pose': frame_data.get('pose'),
                            'spatial_features': frame_data.get('spatial_features', {})
                        }
                        # Extract 151-feature vector
                        features = detector.extract_features(detection_results)
                        sequence_features.append(features)
                    
                    detector.close()
                else:
                    # Old format: single hand (63 features)
                    for frame_data in sequence_data[:sequence_length]:
                        landmarks = frame_data['landmarks']['landmarks']
                        features = []
                        for lm in landmarks:
                            features.extend([lm['x'], lm['y'], lm['z']])
                        sequence_features.append(features)
                
                # Pad if necessary
                while len(sequence_features) < sequence_length:
                    sequence_features.append(sequence_features[-1])  # Repeat last frame
                
                sequences.append(sequence_features)
                labels.append(gesture_name)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences found in data directory")
        
        print(f"\nLoaded {len(sequences)} sequences")
        print(f"Unique gestures: {len(set(labels))}")
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels)
        
        # Detect feature size
        feature_size = sequences.shape[2]
        print(f"Feature size detected: {feature_size}")
        if feature_size == 151:
            print("  Using multi-hand + body features (151 features)")
        elif feature_size == 63:
            print("  Using single-hand features (63 features)")
        else:
            print(f"  Warning: Unexpected feature size {feature_size}")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Create dataset
        dataset = TemporalGestureDataset(sequences, labels_encoded)
        
        # Split into train and test
        test_size = int(len(dataset) * test_split)
        train_size = len(dataset) - test_size
        
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader,
             epochs: int = 50, learning_rate: float = 0.001,
             patience: int = 10) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
        
        Returns:
            Dictionary with training history
        """
        print(f"\nTraining model for {epochs} epochs...")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self._validate(test_loader, criterion)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_temporal_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_temporal_model.pth', weights_only=False))
        
        return history
    
    def _validate(self, data_loader: DataLoader, criterion) -> Tuple[float, float]:
        """
        Validate the model
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        return val_loss / val_total, val_correct / val_total
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model and generate detailed metrics
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(
            all_labels, all_predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
        }
    
    def save_model(self, save_path: str, metadata: Dict = None):
        """
        Save trained model with metadata
        
        Args:
            save_path: Path to save model
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'gesture_labels': self.label_encoder.classes_.tolist(),  # Also save as gesture_labels for classifier
            'num_classes': len(self.label_encoder.classes_),
            'input_size': self.model.input_size,  # Use actual model input size
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, save_path)
        print(f"\nModel saved to: {save_path}")
