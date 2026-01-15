"""
Spatial-Temporal LSTM classifier for two-hand and body-aware ASL recognition
Supports gestures requiring two hands and spatial body context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from .config import settings


class SpatialTemporalLSTM(nn.Module):
    """
    Enhanced LSTM for two-hand + body spatial gestures
    
    Architecture:
    - Input: (batch_size, sequence_length, 151)
      - 63: left hand landmarks
      - 63: right hand landmarks  
      - 15: body pose landmarks
      - 10: spatial relationship features
    - LSTM layers with dropout
    - Dense layers for classification
    - Output: (batch_size, num_classes)
    """
    
    def __init__(self, input_size: int = 151, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 28, dropout: float = 0.3):
        """
        Initialize spatial-temporal LSTM model
        
        Args:
            input_size: Features per frame (151 = 63+63+15+10)
            hidden_size: LSTM hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of gesture classes
            dropout: Dropout probability
        """
        super(SpatialTemporalLSTM, self).__init__()
        
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
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # Dense layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


class SpatialTemporalClassifier:
    """
    Wrapper for spatial-temporal gesture classification
    Handles two-hand and body-aware predictions
    """
    
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 28):
        """
        Initialize classifier
        
        Args:
            model_path: Path to saved model weights
            num_classes: Number of gesture classes
        """
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Gesture labels - expanded for two-hand gestures
        self.gesture_labels = [
            # Single hand letters
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            
            # Two-hand gestures (examples)
            'CLAP', 'TOGETHER',
            
            # Common
            'SPACE', 'UNKNOWN'
        ]
        
        # Initialize model with expanded input
        self.model = SpatialTemporalLSTM(
            input_size=151,  # Two hands + pose + spatial features
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)
        
        # Load model weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("No pre-trained model loaded. Model initialized with random weights.")
            print("Train the model using spatial_temporal_trainer.py")
        
        # Set to evaluation mode
        self.model.eval()
    
    def predict(self, sequence: np.ndarray) -> Tuple[str, float, dict]:
        """
        Predict gesture from sequence
        
        Args:
            sequence: Landmark sequence of shape (window_size, 151)
        
        Returns:
            Tuple of (gesture, confidence, metadata)
            metadata includes: requires_two_hands, has_body_context
        """
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        # Get gesture label
        if predicted_idx < len(self.gesture_labels):
            gesture = self.gesture_labels[predicted_idx]
        else:
            gesture = 'UNKNOWN'
        
        # Analyze sequence metadata
        metadata = self._analyze_sequence(sequence)
        metadata['predicted_index'] = predicted_idx
        
        return gesture, confidence, metadata
    
    def _analyze_sequence(self, sequence: np.ndarray) -> dict:
        """
        Analyze sequence to determine characteristics
        
        Args:
            sequence: Shape (window_size, 151)
        
        Returns:
            Dict with metadata about the gesture
        """
        # Extract spatial features from first frame
        # Features 126-135 are spatial features
        spatial_features = sequence[0, 141:151]
        
        has_left = bool(spatial_features[0])
        has_right = bool(spatial_features[1])
        has_both = bool(spatial_features[2])
        
        # Check for two-hand usage across sequence
        two_hand_frames = np.sum(sequence[:, 143] > 0)  # has_both_hands feature
        two_hand_ratio = two_hand_frames / len(sequence)
        
        # Check for body pose detection
        # Features 126-140 are pose features
        pose_present = np.any(sequence[:, 126:141] != 0)
        
        return {
            'has_left_hand': has_left,
            'has_right_hand': has_right,
            'requires_two_hands': two_hand_ratio > 0.5,  # More than 50% of frames have both hands
            'has_body_context': pose_present,
            'two_hand_ratio': float(two_hand_ratio)
        }
    
    def predict_batch(self, sequences: np.ndarray) -> List[Tuple[str, float, dict]]:
        """
        Predict gestures for a batch of sequences
        
        Args:
            sequences: Batch of sequences, shape (batch_size, window_size, 151)
        
        Returns:
            List of (gesture, confidence, metadata) tuples
        """
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)
        
        results = []
        for idx, conf, seq in zip(
            predicted_indices.cpu().numpy(),
            confidences.cpu().numpy(),
            sequences
        ):
            if idx < len(self.gesture_labels):
                gesture = self.gesture_labels[idx]
            else:
                gesture = 'UNKNOWN'
            
            metadata = self._analyze_sequence(seq)
            metadata['predicted_index'] = int(idx)
            
            results.append((gesture, float(conf), metadata))
        
        return results
    
    def is_confident(self, confidence: float) -> bool:
        """Check if prediction confidence meets threshold"""
        return confidence >= settings.temporal_confidence_threshold
    
    def load_model(self, model_path: str):
        """Load model weights from file"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Spatial-temporal model loaded from {model_path}")
                
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
        """Save model weights and metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'gesture_labels': self.gesture_labels,
            'num_classes': self.num_classes,
            'input_size': 151,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'model_type': 'spatial_temporal_lstm'
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        torch.save(checkpoint, save_path)
        print(f"Spatial-temporal model saved to {save_path}")
    
    def get_model_info(self) -> dict:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Spatial-Temporal LSTM',
            'input_size': 151,
            'input_breakdown': {
                'left_hand': 63,
                'right_hand': 63,
                'body_pose': 15,
                'spatial_features': 10
            },
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'gesture_labels': self.gesture_labels,
            'supports_two_hands': True,
            'supports_body_context': True
        }
    
    def to_device(self, device: str):
        """Move model to specified device"""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
