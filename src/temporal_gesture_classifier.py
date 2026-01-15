"""
LSTM-based temporal gesture classifier for dynamic ASL sign recognition
Uses PyTorch for sequence modeling with all 21 hand landmarks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from .config import settings


class GestureLSTM(nn.Module):
    """
    LSTM neural network for temporal gesture classification
    
    Architecture:
    - Input: (batch_size, sequence_length, 63) - all 21 landmarks × 3 coordinates
    - LSTM layers with dropout for regularization
    - Dense layers for classification
    - Output: (batch_size, num_classes)
    """
    
    def __init__(self, input_size: int = 63, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 28, dropout: float = 0.3):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of features per frame (63 = 21 landmarks × 3)
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            num_classes: Number of gesture classes to predict
            dropout: Dropout probability for regularization
        """
        super(GestureLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers for classification
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last time step
        # Shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # Dense layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer (no softmax here, will be applied in loss function)
        x = self.fc2(x)
        
        return x


class TemporalGestureClassifier:
    """
    Wrapper class for temporal gesture classification using LSTM
    Handles model loading, inference, and prediction
    """
    
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 28):
        """
        Initialize temporal gesture classifier
        
        Args:
            model_path: Path to saved model weights
            num_classes: Number of gesture classes (only used if no model loaded)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default gesture labels (26 letters + SPACE + UNKNOWN)
        self.gesture_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            'SPACE', 'UNKNOWN'
        ]
        
        # Try to load model info from checkpoint first
        loaded_config = None
        if model_path and Path(model_path).exists():
            loaded_config = self._load_model_config(model_path)
        
        # Use loaded config or defaults
        if loaded_config:
            self.num_classes = loaded_config['num_classes']
            self.gesture_labels = loaded_config.get('gesture_labels', self.gesture_labels[:self.num_classes])
            input_size = loaded_config.get('input_size', 63)
            hidden_size = loaded_config.get('hidden_size', 128)
            num_layers = loaded_config.get('num_layers', 2)
            print(f"Loaded model config: {self.num_classes} classes, {len(self.gesture_labels)} labels")
        else:
            self.num_classes = num_classes
            input_size = 63
            hidden_size = 128
            num_layers = 2
        
        # Initialize model with correct architecture
        self.model = GestureLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=0.3
        ).to(self.device)
        
        # Load model weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("No pre-trained model loaded. Model initialized with random weights.")
            print("Train the model using temporal_model_trainer.py")
        
        # Set to evaluation mode
        self.model.eval()
    
    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from a sequence of hand landmarks
        
        Args:
            sequence: Landmark sequence of shape (window_size, 63)
        
        Returns:
            Tuple of (predicted_gesture, confidence_score)
        """
        # Convert to tensor and add batch dimension
        # Shape: (1, window_size, 63)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Make prediction (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        # Get gesture label
        if predicted_idx < len(self.gesture_labels):
            gesture = self.gesture_labels[predicted_idx]
        else:
            gesture = 'UNKNOWN'
        
        return gesture, confidence
    
    def predict_batch(self, sequences: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict gestures for a batch of sequences
        
        Args:
            sequences: Batch of sequences, shape (batch_size, window_size, 63)
        
        Returns:
            List of (gesture, confidence) tuples
        """
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)
        
        # Convert to list of predictions
        results = []
        for idx, conf in zip(predicted_indices.cpu().numpy(), confidences.cpu().numpy()):
            if idx < len(self.gesture_labels):
                gesture = self.gesture_labels[idx]
            else:
                gesture = 'UNKNOWN'
            results.append((gesture, float(conf)))
        
        return results
    
    def _load_model_config(self, model_path: str) -> Optional[dict]:
        """
        Load model configuration from checkpoint without loading full model
        
        Args:
            model_path: Path to model checkpoint
        
        Returns:
            Dictionary with model config or None
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                return {
                    'num_classes': checkpoint.get('num_classes', 28),
                    'gesture_labels': checkpoint.get('gesture_labels'),
                    'input_size': checkpoint.get('input_size', 63),
                    'hidden_size': checkpoint.get('hidden_size', 128),
                    'num_layers': checkpoint.get('num_layers', 2)
                }
        except Exception as e:
            print(f"Warning: Could not load model config from {model_path}: {e}")
        return None
    
    def is_confident(self, confidence: float) -> bool:
        """
        Check if prediction confidence meets threshold
        
        Args:
            confidence: Confidence score
        
        Returns:
            True if confidence is above threshold
        """
        return confidence >= settings.temporal_confidence_threshold
    
    def load_model(self, model_path: str):
        """
        Load model weights from file
        
        Args:
            model_path: Path to saved model weights (.pth file)
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
                
                # Load additional info if available
                if 'gesture_labels' in checkpoint:
                    self.gesture_labels = checkpoint['gesture_labels']
                if 'num_classes' in checkpoint:
                    self.num_classes = checkpoint['num_classes']
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Model loaded from {model_path}")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Using randomly initialized weights.")
    
    def save_model(self, save_path: str, epoch: Optional[int] = None, 
                   loss: Optional[float] = None, accuracy: Optional[float] = None):
        """
        Save model weights and metadata
        
        Args:
            save_path: Path to save model
            epoch: Training epoch (optional)
            loss: Training loss (optional)
            accuracy: Training accuracy (optional)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'gesture_labels': self.gesture_labels,
            'num_classes': self.num_classes,
            'input_size': 63,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def get_model_info(self) -> dict:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'architecture': 'LSTM',
            'input_size': 63,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'gesture_labels': self.gesture_labels,
        }
    
    def to_device(self, device: str):
        """Move model to specified device (cuda/cpu)"""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
