"""
Temporal feature extraction for dynamic gesture recognition
Manages sliding window buffer and extracts sequences of hand landmarks
"""
import numpy as np
from collections import deque
from typing import Optional, List, Tuple
from .config import settings


class TemporalFeatureExtractor:
    """
    Extracts temporal features from sequences of hand landmarks
    Maintains a sliding window buffer for continuous gesture recognition
    """
    
    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize temporal feature extractor
        
        Args:
            window_size: Number of frames in sliding window (default from settings)
        """
        self.window_size = window_size or settings.temporal_window_size
        
        # Sliding window buffer - stores landmark features for each frame
        # Each element is a numpy array of shape (63,) representing 21 landmarks × 3 coordinates
        self.frame_buffer = deque(maxlen=self.window_size)
        
        # Track frame count and timing
        self.frame_count = 0
        self.frames_since_last_prediction = 0
        
    def add_frame(self, landmarks: np.ndarray) -> bool:
        """
        Add a new frame of hand landmarks to the buffer
        
        Args:
            landmarks: Hand landmark features, shape (63,) - all 21 landmarks preserved
        
        Returns:
            True if buffer is full and ready for prediction
        """
        if landmarks.shape[0] != 63:
            raise ValueError(f"Expected 63 features (21 landmarks × 3), got {landmarks.shape[0]}")
        
        self.frame_buffer.append(landmarks.copy())
        self.frame_count += 1
        self.frames_since_last_prediction += 1
        
        # Buffer is ready when it has window_size frames
        return len(self.frame_buffer) == self.window_size
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current sequence from the buffer
        
        Returns:
            Sequence array of shape (window_size, 63) or None if buffer not full
        """
        if len(self.frame_buffer) < self.window_size:
            return None
        
        # Convert deque to numpy array
        # Shape: (window_size, 63) - preserves all 21 landmarks per frame
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)
        return sequence
    
    def should_predict(self) -> bool:
        """
        Check if enough frames have passed to make a new prediction
        Uses stride to avoid predicting every frame
        
        Returns:
            True if should make a prediction now
        """
        if len(self.frame_buffer) < self.window_size:
            return False
        
        # Predict every N frames based on stride
        if self.frames_since_last_prediction >= settings.temporal_stride:
            return True
        
        return False
    
    def reset_prediction_timer(self):
        """Reset the frame counter for predictions"""
        self.frames_since_last_prediction = 0
    
    def clear(self):
        """Clear the frame buffer and reset counters"""
        self.frame_buffer.clear()
        self.frame_count = 0
        self.frames_since_last_prediction = 0
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames for prediction"""
        return len(self.frame_buffer) >= self.window_size
    
    def get_buffer_size(self) -> int:
        """Get current number of frames in buffer"""
        return len(self.frame_buffer)
    
    def compute_temporal_features(self, sequence: np.ndarray) -> dict:
        """
        Compute additional temporal features from a sequence
        These can be used alongside raw landmarks for enhanced detection
        
        Args:
            sequence: Landmark sequence of shape (window_size, 63)
        
        Returns:
            Dictionary with temporal features
        """
        # Reshape to (window_size, 21, 3) for easier computation
        landmarks_3d = sequence.reshape(self.window_size, 21, 3)
        
        # Compute velocity (change between consecutive frames)
        velocity = np.diff(landmarks_3d, axis=0)  # Shape: (window_size-1, 21, 3)
        
        # Compute acceleration (change in velocity)
        acceleration = np.diff(velocity, axis=0)  # Shape: (window_size-2, 21, 3)
        
        # Aggregate statistics
        velocity_magnitude = np.linalg.norm(velocity, axis=2)  # (window_size-1, 21)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=2)  # (window_size-2, 21)
        
        features = {
            'velocity': velocity,
            'acceleration': acceleration,
            'avg_velocity': np.mean(velocity_magnitude),
            'max_velocity': np.max(velocity_magnitude),
            'avg_acceleration': np.mean(acceleration_magnitude),
            'max_acceleration': np.max(acceleration_magnitude),
            
            # Per-finger statistics (useful for distinguishing gestures)
            'fingertip_velocities': {
                'thumb': np.mean(velocity_magnitude[:, 4]),      # Landmark 4: thumb tip
                'index': np.mean(velocity_magnitude[:, 8]),      # Landmark 8: index tip
                'middle': np.mean(velocity_magnitude[:, 12]),    # Landmark 12: middle tip
                'ring': np.mean(velocity_magnitude[:, 16]),      # Landmark 16: ring tip
                'pinky': np.mean(velocity_magnitude[:, 20]),     # Landmark 20: pinky tip
            },
            
            # Wrist movement (important for dynamic gestures like J, Z)
            'wrist_velocity': np.mean(velocity_magnitude[:, 0]),  # Landmark 0: wrist
        }
        
        return features
    
    def extract_trajectory(self, sequence: np.ndarray, landmark_idx: int = 8) -> np.ndarray:
        """
        Extract movement trajectory for a specific landmark
        Useful for analyzing motion patterns
        
        Args:
            sequence: Landmark sequence of shape (window_size, 63)
            landmark_idx: Index of landmark to track (default 8: index finger tip)
        
        Returns:
            Trajectory array of shape (window_size, 3)
        """
        landmarks_3d = sequence.reshape(self.window_size, 21, 3)
        trajectory = landmarks_3d[:, landmark_idx, :]
        return trajectory
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize a sequence relative to first frame's wrist position
        Helps make gestures translation-invariant
        
        Args:
            sequence: Landmark sequence of shape (window_size, 63)
        
        Returns:
            Normalized sequence of same shape
        """
        # Reshape to (window_size, 21, 3)
        landmarks_3d = sequence.reshape(self.window_size, 21, 3)
        
        # Get first frame's wrist position (landmark 0)
        initial_wrist = landmarks_3d[0, 0, :].copy()
        
        # Normalize all frames relative to initial wrist
        normalized = landmarks_3d - initial_wrist
        
        # Flatten back to (window_size, 63)
        return normalized.reshape(self.window_size, 63)
    
    def get_frame_info(self) -> dict:
        """Get information about current buffer state"""
        return {
            'total_frames_processed': self.frame_count,
            'buffer_size': len(self.frame_buffer),
            'buffer_capacity': self.window_size,
            'is_ready': self.is_ready(),
            'frames_since_prediction': self.frames_since_last_prediction,
        }
