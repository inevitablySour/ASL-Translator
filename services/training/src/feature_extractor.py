"""
Enhanced feature extraction for ASL gestures with hand orientation
Features are scale-invariant (independent of distance from camera)
"""
import numpy as np
from typing import Dict, List


class EnhancedFeatureExtractor:
    """Extract rich features including hand orientation and direction"""
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(self, include_orientation=True):
        """
        Initialize feature extractor
        
        Args:
            include_orientation: Include hand orientation features
        """
        self.include_orientation = include_orientation
    
    def extract_features(self, hand_landmarks: Dict) -> np.ndarray:
        """
        Extract comprehensive feature vector from hand landmarks
        All features are scale-invariant (distance from camera doesn't matter)
        
        Args:
            hand_landmarks: Dictionary containing landmark coordinates
        
        Returns:
            Feature vector as numpy array
        """
        landmarks = hand_landmarks['landmarks']
        
        # Convert to numpy array (21 landmarks x 3 coordinates)
        landmark_array = np.array([
            [lm['x'], lm['y'], lm['z']] for lm in landmarks
        ], dtype=np.float32)
        
        features = []
        
        # 1. Scale-invariant normalized landmarks (63 features)
        # Normalize by hand size AND translate to wrist
        normalized_landmarks = self._normalize_landmarks(landmark_array)
        features.extend(normalized_landmarks.flatten())
        
        # 2. Hand orientation features (if enabled)
        if self.include_orientation:
            orientation_features = self._extract_orientation_features(landmark_array)
            features.extend(orientation_features)
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be scale-invariant
        
        Steps:
        1. Translate so wrist is at origin
        2. Scale by hand size (palm width)
        
        Args:
            landmarks: Array of shape (21, 3)
        
        Returns:
            Normalized landmarks of shape (21, 3)
        """
        # Step 1: Translate to wrist
        wrist = landmarks[self.WRIST].copy()
        translated = landmarks - wrist
        
        # Step 2: Compute hand size (use palm width as reference)
        # Distance from wrist to middle finger MCP
        palm_size = np.linalg.norm(translated[self.MIDDLE_MCP])
        
        # Avoid division by zero
        if palm_size < 1e-6:
            palm_size = 1.0
        
        # Scale by palm size
        normalized = translated / palm_size
        
        return normalized
    
    def _extract_orientation_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract hand orientation and direction features
        All features are directional (angles and unit vectors) so inherently scale-invariant
        
        Args:
            landmarks: Array of shape (21, 3) representing hand landmarks
        
        Returns:
            List of orientation features (15 features)
        """
        features = []
        
        # 1. Palm normal vector (3 features)
        # Direction the palm is facing
        palm_normal = self._compute_palm_normal(landmarks)
        features.extend(palm_normal)
        
        # 2. Index finger pointing direction (3 features)
        # Critical for distinguishing D (pointing up) vs G (pointing sideways)
        index_direction = self._compute_finger_direction(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        features.extend(index_direction)
        
        # 3. Middle finger pointing direction (3 features)
        middle_direction = self._compute_finger_direction(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        features.extend(middle_direction)
        
        # 4. Hand tilt angles (3 features: pitch, yaw, roll)
        # Pitch: up/down tilt (critical for D vs G)
        # Yaw: left/right rotation
        # Roll: hand rotation
        tilt_angles = self._compute_tilt_angles(landmarks)
        features.extend(tilt_angles)
        
        # 5. Finger spread angles (3 features)
        # Angle between adjacent fingers
        spread_angles = self._compute_finger_spread(landmarks)
        features.extend(spread_angles)
        
        # Total: 3 + 3 + 3 + 3 + 3 = 15 orientation features
        return features
    
    def _compute_palm_normal(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute the normal vector to the palm plane
        This indicates which direction the palm is facing
        """
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        
        # Two vectors in the palm plane
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        # Cross product gives normal to the plane
        normal = np.cross(v1, v2)
        
        # Normalize to unit vector (scale-invariant)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 1.0])  # Default forward
        
        return normal
    
    def _compute_finger_direction(self, landmarks: np.ndarray, 
                                   base_idx: int, tip_idx: int) -> np.ndarray:
        """
        Compute the direction vector of a finger (unit vector)
        """
        base = landmarks[base_idx]
        tip = landmarks[tip_idx]
        
        # Direction from base to tip
        direction = tip - base
        
        # Normalize to unit vector (scale-invariant)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([0.0, 1.0, 0.0])  # Default up
        
        return direction
    
    def _compute_tilt_angles(self, landmarks: np.ndarray) -> List[float]:
        """
        Compute hand tilt angles (pitch, yaw, roll) in radians
        
        These are crucial for distinguishing gestures like:
        - D (index finger pointing UP, pitch ≈ 0°)
        - G (index finger pointing SIDEWAYS, pitch ≈ 90°)
        
        Returns:
            [pitch, yaw, roll] in radians
        """
        # Use wrist and middle finger tip to define hand axis
        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        # Hand axis vector
        hand_axis = middle_tip - wrist
        
        # Normalize
        norm = np.linalg.norm(hand_axis)
        if norm < 1e-6:
            return [0.0, 0.0, 0.0]
        
        hand_axis = hand_axis / norm
        
        # Pitch: rotation around X-axis (up/down tilt)
        # Positive pitch = pointing up, Negative pitch = pointing down
        # For D: pitch ≈ 0 (finger vertical), For G: pitch ≈ ±π/2 (finger horizontal)
        pitch = np.arcsin(np.clip(hand_axis[1], -1.0, 1.0))  # y-component
        
        # Yaw: rotation around Y-axis (left/right turn)
        # Which horizontal direction the hand is facing
        yaw = np.arctan2(hand_axis[0], hand_axis[2])
        
        # Roll: rotation around Z-axis (palm rotation)
        # Use palm normal for roll
        palm_normal = self._compute_palm_normal(landmarks)
        roll = np.arctan2(palm_normal[0], palm_normal[1])
        
        return [float(pitch), float(yaw), float(roll)]
    
    def _compute_finger_spread(self, landmarks: np.ndarray) -> List[float]:
        """
        Compute angles between adjacent fingers (in radians)
        Helps distinguish gestures with different finger configurations
        """
        angles = []
        
        # Get finger tip positions
        fingers = [
            (self.INDEX_MCP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_TIP)
        ]
        
        # Compute angles between adjacent fingers
        for i in range(len(fingers) - 1):
            dir1 = self._compute_finger_direction(landmarks, *fingers[i])
            dir2 = self._compute_finger_direction(landmarks, *fingers[i+1])
            
            # Angle between vectors
            dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            angles.append(float(angle))
        
        return angles
    
    def get_feature_count(self) -> int:
        """Get total number of features"""
        base_features = 63  # 21 landmarks × 3 coordinates
        orientation_features = 15 if self.include_orientation else 0
        return base_features + orientation_features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for debugging"""
        names = []
        
        # Landmark features
        landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
            'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
            'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        for lm in landmark_names:
            names.extend([f'{lm}_x', f'{lm}_y', f'{lm}_z'])
        
        # Orientation features
        if self.include_orientation:
            names.extend(['palm_normal_x', 'palm_normal_y', 'palm_normal_z'])
            names.extend(['index_dir_x', 'index_dir_y', 'index_dir_z'])
            names.extend(['middle_dir_x', 'middle_dir_y', 'middle_dir_z'])
            names.extend(['pitch', 'yaw', 'roll'])
            names.extend(['spread_index_middle', 'spread_middle_ring', 
                         'spread_ring_pinky'])
        
        return names
