"""
Automatic gesture segmentation based on hand movement and rest position detection
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class GestureSegmenter:
    """
    Automatically detects when a gesture starts and ends by analyzing hand movement
    and detecting rest positions (hands at sides or in neutral position)
    """
    
    def __init__(
        self,
        movement_threshold: float = 0.02,
        rest_duration_frames: int = 10,
        min_gesture_frames: int = 15,
        max_gesture_frames: int = 90,
        history_size: int = 10
    ):
        """
        Initialize gesture segmenter
        
        Args:
            movement_threshold: Minimum movement to consider active gesture (normalized 0-1)
            rest_duration_frames: Frames of rest needed to trigger gesture end
            min_gesture_frames: Minimum frames for valid gesture
            max_gesture_frames: Maximum frames before forcing segment
            history_size: Number of frames to track for movement analysis
        """
        self.movement_threshold = movement_threshold
        self.rest_duration_frames = rest_duration_frames
        self.min_gesture_frames = min_gesture_frames
        self.max_gesture_frames = max_gesture_frames
        
        # State tracking
        self.is_active_gesture = False
        self.gesture_frames: List[Dict] = []
        self.rest_frame_count = 0
        
        # Movement history for smoothing
        self.movement_history = deque(maxlen=history_size)
        self.position_history = deque(maxlen=2)  # Track last 2 frames
        
        # Rest position detection
        self.rest_positions = {
            'low': (0.6, 1.0),    # Hands at sides (y coordinate range)
            'center': (0.3, 0.7),  # Hands near center body
        }
    
    def process_frame(
        self,
        detection_results: Dict,
        frame: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Process a frame and determine if a gesture segment is complete
        
        Args:
            detection_results: Hand/body detection results from MultiHandBodyDetector
            frame: Optional frame data to store with gesture
        
        Returns:
            Tuple of (gesture_complete, gesture_sequence)
            - gesture_complete: True if a gesture segment just finished
            - gesture_sequence: List of frames if complete, None otherwise
        """
        # Calculate movement and rest state
        movement = self._calculate_movement(detection_results)
        is_resting = self._is_rest_position(detection_results)
        
        # Update movement history
        self.movement_history.append(movement)
        avg_movement = np.mean(list(self.movement_history))
        
        # Detect gesture start - only start if coming from rest position
        if not self.is_active_gesture:
            if avg_movement > self.movement_threshold and not is_resting:
                # Check that we actually came from rest (not mid-gesture)
                # Only start if hands were detected (not starting from nothing)
                has_hands = (detection_results.get('left_hand') or detection_results.get('right_hand'))
                if has_hands:
                    # Gesture starting
                    self.is_active_gesture = True
                    self.gesture_frames = []
                    self.rest_frame_count = 0
                    print("🟢 Gesture started (hands moved from rest)")
        
        # During active gesture
        if self.is_active_gesture:
            # Store frame data
            frame_data = {
                'detection_results': detection_results,
                'frame': frame.copy() if frame is not None else None,
                'movement': movement,
                'is_resting': is_resting
            }
            self.gesture_frames.append(frame_data)
            
            # Check for rest state (gesture ending)
            # Only increment rest count if ACTUALLY at rest (not just low movement)
            if is_resting:
                self.rest_frame_count += 1
            else:
                # Reset count if not at rest - requires CONTINUOUS rest
                self.rest_frame_count = 0
            
            # Check if gesture should end
            gesture_length = len(self.gesture_frames)
            
            # End conditions
            if self.rest_frame_count >= self.rest_duration_frames and gesture_length >= self.min_gesture_frames:
                # Natural end - hands returned to rest
                print(f"🔴 Gesture ended (rest position, {gesture_length} frames)")
                return self._finalize_gesture()
            
            elif gesture_length >= self.max_gesture_frames:
                # Force end - too long
                print(f"🟡 Gesture ended (max length, {gesture_length} frames)")
                return self._finalize_gesture()
        
        return False, None
    
    def _calculate_movement(self, detection_results: Dict) -> float:
        """
        Calculate total movement in frame compared to previous frame
        
        Returns:
            Movement magnitude (0.0 = no movement, 1.0 = large movement)
        """
        # Get current hand positions
        current_positions = []
        
        if detection_results.get('left_hand'):
            left = detection_results['left_hand']['landmarks']
            # Use wrist position (landmark 0)
            current_positions.append((left[0]['x'], left[0]['y']))
        
        if detection_results.get('right_hand'):
            right = detection_results['right_hand']['landmarks']
            current_positions.append((right[0]['x'], right[0]['y']))
        
        if not current_positions:
            return 0.0
        
        # Store current position
        self.position_history.append(current_positions)
        
        # Need at least 2 frames to calculate movement
        if len(self.position_history) < 2:
            return 0.0
        
        # Calculate movement between frames
        prev_positions = self.position_history[0]
        curr_positions = self.position_history[1]
        
        total_movement = 0.0
        num_hands = min(len(prev_positions), len(curr_positions))
        
        for i in range(num_hands):
            dx = curr_positions[i][0] - prev_positions[i][0]
            dy = curr_positions[i][1] - prev_positions[i][1]
            movement = np.sqrt(dx**2 + dy**2)
            total_movement += movement
        
        return total_movement / max(num_hands, 1)
    
    def _is_rest_position(self, detection_results: Dict) -> bool:
        """
        Detect if BOTH hands are in rest position relative to body (at waist/hips)
        Uses pose landmarks for body-relative positioning
        
        Returns:
            True if both hands are in rest position
        """
        left_hand = detection_results.get('left_hand')
        right_hand = detection_results.get('right_hand')
        pose = detection_results.get('pose')
        
        # No hands detected = NOT resting (need to see hands at rest)
        if not left_hand and not right_hand:
            return False
        
        # Need pose data for body-relative positioning
        if not pose or not pose.get('landmarks'):
            # No pose - can't determine rest position reliably
            return False
        
        try:
            landmarks = pose['landmarks']
            # Get body keypoints (0=nose, 3=left_hip, 4=right_hip)
            left_hip_y = landmarks[3]['y']
            right_hip_y = landmarks[4]['y']
            nose_y = landmarks[0]['y']
            
            # Calculate body height for normalization
            body_height = abs(left_hip_y - nose_y)
            if body_height < 0.1:  # Too small, unreliable
                return False
            
            # Hip level (where hands should rest)
            hip_level = (left_hip_y + right_hip_y) / 2
            
            # Check if BOTH hands are at or below hip level
            hands_at_rest = []
            
            if left_hand:
                left_wrist_y = left_hand['landmarks'][0]['y']
                # Hand must be at hip level or lower (with small tolerance)
                left_at_rest = left_wrist_y >= (hip_level - 0.05)
                hands_at_rest.append(left_at_rest)
            
            if right_hand:
                right_wrist_y = right_hand['landmarks'][0]['y']
                right_at_rest = right_wrist_y >= (hip_level - 0.05)
                hands_at_rest.append(right_at_rest)
            
            # For two-hand gestures: BOTH hands must be at rest
            # For one-hand: that hand must be at rest
            return all(hands_at_rest)
            
        except (KeyError, IndexError, TypeError):
            # Pose data incomplete - can't determine rest
            return False
    
    def _finalize_gesture(self) -> Tuple[bool, List[Dict]]:
        """
        Finalize the current gesture and return the sequence
        
        Returns:
            Tuple of (True, gesture_frames)
        """
        gesture_sequence = self.gesture_frames.copy()
        
        # Reset state
        self.is_active_gesture = False
        self.gesture_frames = []
        self.rest_frame_count = 0
        self.movement_history.clear()
        self.position_history.clear()
        
        return True, gesture_sequence
    
    def reset(self):
        """Reset segmenter state (call when switching gestures)"""
        self.is_active_gesture = False
        self.gesture_frames = []
        self.rest_frame_count = 0
        self.movement_history.clear()
        self.position_history.clear()
        print("🔄 Segmenter reset")
    
    def get_status(self) -> Dict:
        """Get current segmenter status for UI display"""
        return {
            'is_active': self.is_active_gesture,
            'frame_count': len(self.gesture_frames),
            'rest_count': self.rest_frame_count,
            'avg_movement': np.mean(list(self.movement_history)) if self.movement_history else 0.0
        }
