"""
Gesture buffer for smoothing predictions and managing gesture segmentation
Reduces jitter and implements cooldown logic for temporal gesture recognition
"""
from collections import deque, Counter
from typing import Optional, Tuple, List
from .config import settings


class GestureBuffer:
    """
    Buffers and smooths gesture predictions over time
    Implements gesture segmentation and cooldown logic
    """
    
    def __init__(self, buffer_size: int = 5):
        """
        Initialize gesture buffer
        
        Args:
            buffer_size: Number of recent predictions to keep for smoothing
        """
        self.buffer_size = buffer_size
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Cooldown management
        self.last_gesture = None
        self.last_gesture_confidence = 0.0
        self.frames_since_last_gesture = 0
        self.cooldown_frames = settings.gesture_cooldown
        
        # Gesture history for analysis
        self.gesture_history = []
        
    def add_prediction(self, gesture: str, confidence: float) -> Optional[Tuple[str, float]]:
        """
        Add a new prediction to the buffer and return smoothed result
        
        Args:
            gesture: Predicted gesture label
            confidence: Prediction confidence score
        
        Returns:
            Smoothed (gesture, confidence) tuple if a gesture is detected, None otherwise
        """
        # Add to buffer
        self.prediction_buffer.append((gesture, confidence))
        self.frames_since_last_gesture += 1
        
        # Need minimum number of predictions for smoothing
        if len(self.prediction_buffer) < self.buffer_size:
            return None
        
        # Get smoothed prediction
        smoothed_gesture, smoothed_confidence = self._get_smoothed_prediction()
        
        # Check if we should output this gesture
        if self._should_output_gesture(smoothed_gesture, smoothed_confidence):
            # Update state
            self.last_gesture = smoothed_gesture
            self.last_gesture_confidence = smoothed_confidence
            self.frames_since_last_gesture = 0
            
            # Add to history
            self.gesture_history.append({
                'gesture': smoothed_gesture,
                'confidence': smoothed_confidence,
            })
            
            return (smoothed_gesture, smoothed_confidence)
        
        return None
    
    def _get_smoothed_prediction(self) -> Tuple[str, float]:
        """
        Smooth predictions using majority voting
        
        Returns:
            Tuple of (smoothed_gesture, average_confidence)
        """
        # Extract gestures and confidences from buffer
        gestures = [pred[0] for pred in self.prediction_buffer]
        confidences = [pred[1] for pred in self.prediction_buffer]
        
        # Majority voting for gesture
        gesture_counts = Counter(gestures)
        most_common_gesture = gesture_counts.most_common(1)[0][0]
        
        # Average confidence for the most common gesture
        gesture_confidences = [
            conf for gest, conf in self.prediction_buffer 
            if gest == most_common_gesture
        ]
        avg_confidence = sum(gesture_confidences) / len(gesture_confidences)
        
        return most_common_gesture, avg_confidence
    
    def _should_output_gesture(self, gesture: str, confidence: float) -> bool:
        """
        Determine if a gesture should be output based on confidence and cooldown
        
        Args:
            gesture: Predicted gesture
            confidence: Prediction confidence
        
        Returns:
            True if gesture should be output
        """
        # Filter out UNKNOWN and low confidence predictions
        if gesture == 'UNKNOWN':
            return False
        
        if confidence < settings.temporal_confidence_threshold:
            return False
        
        # Cooldown logic: prevent repeated detections of same gesture
        if self.last_gesture == gesture:
            # Same gesture detected again - check cooldown
            if self.frames_since_last_gesture < self.cooldown_frames:
                return False  # Still in cooldown period
        
        # Check consistency: gesture should appear in majority of recent predictions
        gestures = [pred[0] for pred in self.prediction_buffer]
        gesture_count = gestures.count(gesture)
        consistency_ratio = gesture_count / len(gestures)
        
        # Require at least 60% consistency
        if consistency_ratio < 0.6:
            return False
        
        return True
    
    def clear(self):
        """Clear all buffers and reset state"""
        self.prediction_buffer.clear()
        self.last_gesture = None
        self.last_gesture_confidence = 0.0
        self.frames_since_last_gesture = 0
    
    def reset_cooldown(self):
        """Reset cooldown timer (useful for manual gesture transitions)"""
        self.frames_since_last_gesture = self.cooldown_frames
    
    def get_current_state(self) -> dict:
        """
        Get current buffer state information
        
        Returns:
            Dictionary with buffer state
        """
        return {
            'buffer_size': len(self.prediction_buffer),
            'last_gesture': self.last_gesture,
            'last_confidence': self.last_gesture_confidence,
            'frames_since_last': self.frames_since_last_gesture,
            'in_cooldown': self.frames_since_last_gesture < self.cooldown_frames,
            'history_length': len(self.gesture_history),
        }
    
    def get_gesture_history(self, limit: Optional[int] = None) -> List[dict]:
        """
        Get recent gesture history
        
        Args:
            limit: Maximum number of recent gestures to return (None for all)
        
        Returns:
            List of gesture dictionaries
        """
        if limit:
            return self.gesture_history[-limit:]
        return self.gesture_history.copy()
    
    def get_gesture_sequence(self, limit: Optional[int] = None) -> str:
        """
        Get detected gestures as a string sequence
        
        Args:
            limit: Maximum number of recent gestures (None for all)
        
        Returns:
            String of detected gestures (e.g., "HELLO")
        """
        history = self.get_gesture_history(limit)
        gestures = [item['gesture'] for item in history if item['gesture'] != 'SPACE']
        return ''.join(gestures)


class GestureSegmenter:
    """
    Segments continuous gesture stream into discrete gesture events
    Detects when a gesture starts and ends
    """
    
    def __init__(self, min_duration: Optional[int] = None):
        """
        Initialize gesture segmenter
        
        Args:
            min_duration: Minimum frames for a valid gesture
        """
        self.min_duration = min_duration or settings.min_gesture_duration
        
        self.current_gesture = None
        self.current_gesture_start = 0
        self.current_gesture_frames = 0
        self.gesture_active = False
        
        self.detected_gestures = []
    
    def process_frame(self, gesture: str, confidence: float, 
                     frame_idx: int) -> Optional[dict]:
        """
        Process a frame and detect gesture boundaries
        
        Args:
            gesture: Predicted gesture for this frame
            confidence: Prediction confidence
            frame_idx: Current frame index
        
        Returns:
            Dictionary with gesture info if a complete gesture is detected, None otherwise
        """
        # Ignore UNKNOWN predictions
        if gesture == 'UNKNOWN' or confidence < settings.temporal_confidence_threshold:
            # If gesture was active, check if it should end
            if self.gesture_active:
                return self._end_gesture(frame_idx)
            return None
        
        # Check if this is a new gesture or continuation
        if not self.gesture_active:
            # Start new gesture
            self._start_gesture(gesture, frame_idx)
            return None
        
        elif gesture == self.current_gesture:
            # Continue current gesture
            self.current_gesture_frames += 1
            return None
        
        else:
            # Different gesture detected - end current and start new
            result = self._end_gesture(frame_idx)
            self._start_gesture(gesture, frame_idx)
            return result
    
    def _start_gesture(self, gesture: str, frame_idx: int):
        """Start tracking a new gesture"""
        self.current_gesture = gesture
        self.current_gesture_start = frame_idx
        self.current_gesture_frames = 1
        self.gesture_active = True
    
    def _end_gesture(self, frame_idx: int) -> Optional[dict]:
        """
        End current gesture and return it if valid
        
        Returns:
            Gesture info dict if valid, None otherwise
        """
        if not self.gesture_active:
            return None
        
        # Check if gesture duration meets minimum
        if self.current_gesture_frames >= self.min_duration:
            gesture_info = {
                'gesture': self.current_gesture,
                'start_frame': self.current_gesture_start,
                'end_frame': frame_idx - 1,
                'duration_frames': self.current_gesture_frames,
            }
            self.detected_gestures.append(gesture_info)
            
            # Reset state
            self.gesture_active = False
            self.current_gesture = None
            
            return gesture_info
        
        # Gesture too short - discard
        self.gesture_active = False
        self.current_gesture = None
        return None
    
    def force_end_current(self, frame_idx: int) -> Optional[dict]:
        """Force end current gesture (useful when stream ends)"""
        return self._end_gesture(frame_idx)
    
    def clear(self):
        """Clear segmenter state"""
        self.current_gesture = None
        self.current_gesture_start = 0
        self.current_gesture_frames = 0
        self.gesture_active = False
        self.detected_gestures.clear()
