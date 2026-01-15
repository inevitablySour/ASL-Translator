"""
Multi-hand and body pose detector for spatial ASL gesture recognition
Detects both hands + upper body landmarks for context-aware recognition
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict
from .config import settings


class MultiHandBodyDetector:
    """
    Detects both hands and upper body pose for context-aware gesture recognition
    Provides spatial relationships between hands and body
    """
    
    def __init__(self):
        """Initialize MediaPipe Hands and Pose solutions"""
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector (detect up to 2 hands)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect both hands
            min_detection_confidence=settings.mediapipe_min_detection_confidence,
            min_tracking_confidence=settings.mediapipe_min_tracking_confidence
        )
        
        # Initialize pose detector (for body context)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=lite (fastest), 1=full, 2=heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False  # Disable segmentation for speed
        )
    
    def detect_hands_and_body(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Detect both hands and upper body pose
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            Tuple of (annotated_image, detection_results)
            
            detection_results contains:
            - 'left_hand': landmarks dict or None
            - 'right_hand': landmarks dict or None
            - 'pose': pose landmarks dict or None
            - 'spatial_features': dict with relative positions
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hands_results = self.hands.process(image_rgb)
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        
        # Prepare annotated image
        annotated_image = image.copy()
        
        # Initialize results
        results = {
            'left_hand': None,
            'right_hand': None,
            'pose': None,
            'spatial_features': {}
        }
        
        # Extract pose landmarks
        if pose_results.pose_landmarks:
            results['pose'] = self._extract_pose_landmarks(pose_results.pose_landmarks)
            
            # Draw pose on image
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Extract hand landmarks
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            ):
                # Determine if left or right hand
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks
                landmarks_dict = self._extract_hand_landmarks(hand_landmarks)
                
                # Store based on hand
                if hand_label == 'Left':
                    results['left_hand'] = landmarks_dict
                else:
                    results['right_hand'] = landmarks_dict
        
        # Compute spatial features (relationships between hands and body)
        if results['left_hand'] or results['right_hand']:
            results['spatial_features'] = self._compute_spatial_features(
                results['left_hand'],
                results['right_hand'],
                results['pose']
            )
        
        return annotated_image, results
    
    def _extract_hand_landmarks(self, hand_landmarks) -> Dict:
        """Extract hand landmarks as dictionary"""
        return {
            'landmarks': [
                {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                }
                for landmark in hand_landmarks.landmark
            ]
        }
    
    def _extract_pose_landmarks(self, pose_landmarks) -> Dict:
        """Extract relevant upper body pose landmarks"""
        # We only need upper body landmarks for gesture context
        # Key landmarks: 0=nose, 11=left_shoulder, 12=right_shoulder, 
        # 23=left_hip, 24=right_hip
        relevant_indices = [0, 11, 12, 23, 24]
        
        return {
            'landmarks': [
                {
                    'x': pose_landmarks.landmark[i].x,
                    'y': pose_landmarks.landmark[i].y,
                    'z': pose_landmarks.landmark[i].z,
                    'visibility': pose_landmarks.landmark[i].visibility
                }
                for i in relevant_indices
            ],
            'landmark_names': ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        }
    
    def _compute_spatial_features(self, left_hand: Optional[Dict], 
                                  right_hand: Optional[Dict], 
                                  pose: Optional[Dict]) -> Dict:
        """
        Compute spatial relationships between hands and body
        
        Returns:
            Dictionary with spatial features:
            - hands_distance: distance between hands (if both present)
            - left_hand_to_body: relative position of left hand to torso
            - right_hand_to_body: relative position of right hand to torso
            - hands_vertical_offset: vertical difference between hands
            - hands_horizontal_offset: horizontal difference between hands
        """
        spatial_features = {
            'has_left_hand': left_hand is not None,
            'has_right_hand': right_hand is not None,
            'has_both_hands': left_hand is not None and right_hand is not None,
            'has_pose': pose is not None
        }
        
        # If both hands present, compute inter-hand features
        if left_hand and right_hand:
            left_wrist = np.array([
                left_hand['landmarks'][0]['x'],
                left_hand['landmarks'][0]['y'],
                left_hand['landmarks'][0]['z']
            ])
            right_wrist = np.array([
                right_hand['landmarks'][0]['x'],
                right_hand['landmarks'][0]['y'],
                right_hand['landmarks'][0]['z']
            ])
            
            # Distance between hands (wrist to wrist)
            hands_distance = np.linalg.norm(left_wrist - right_wrist)
            spatial_features['hands_distance'] = float(hands_distance)
            
            # Vertical and horizontal offsets
            spatial_features['hands_vertical_offset'] = float(right_wrist[1] - left_wrist[1])
            spatial_features['hands_horizontal_offset'] = float(right_wrist[0] - left_wrist[0])
            
            # Midpoint between hands
            hands_midpoint = (left_wrist + right_wrist) / 2
            spatial_features['hands_midpoint'] = hands_midpoint.tolist()
        
        # If pose available, compute hand-to-body relationships
        if pose:
            # Get torso center (midpoint between shoulders)
            left_shoulder = np.array([
                pose['landmarks'][1]['x'],  # left_shoulder
                pose['landmarks'][1]['y'],
                pose['landmarks'][1]['z']
            ])
            right_shoulder = np.array([
                pose['landmarks'][2]['x'],  # right_shoulder
                pose['landmarks'][2]['y'],
                pose['landmarks'][2]['z']
            ])
            torso_center = (left_shoulder + right_shoulder) / 2
            
            spatial_features['torso_center'] = torso_center.tolist()
            
            # Compute hand positions relative to torso
            if left_hand:
                left_wrist = np.array([
                    left_hand['landmarks'][0]['x'],
                    left_hand['landmarks'][0]['y'],
                    left_hand['landmarks'][0]['z']
                ])
                left_to_torso = left_wrist - torso_center
                spatial_features['left_hand_to_torso'] = {
                    'distance': float(np.linalg.norm(left_to_torso)),
                    'offset_x': float(left_to_torso[0]),
                    'offset_y': float(left_to_torso[1]),
                    'offset_z': float(left_to_torso[2])
                }
            
            if right_hand:
                right_wrist = np.array([
                    right_hand['landmarks'][0]['x'],
                    right_hand['landmarks'][0]['y'],
                    right_hand['landmarks'][0]['z']
                ])
                right_to_torso = right_wrist - torso_center
                spatial_features['right_hand_to_torso'] = {
                    'distance': float(np.linalg.norm(right_to_torso)),
                    'offset_x': float(right_to_torso[0]),
                    'offset_y': float(right_to_torso[1]),
                    'offset_z': float(right_to_torso[2])
                }
        
        return spatial_features
    
    def extract_features(self, detection_results: Dict) -> np.ndarray:
        """
        Extract complete feature vector for model input
        
        Feature structure:
        - Left hand: 63 features (21 landmarks × 3) or zeros if not present
        - Right hand: 63 features (21 landmarks × 3) or zeros if not present
        - Pose: 15 features (5 landmarks × 3) or zeros if not present
        - Spatial: 10 features (inter-hand and hand-to-body relationships)
        
        Total: 151 features
        
        Args:
            detection_results: Results from detect_hands_and_body()
        
        Returns:
            Feature vector of shape (151,)
        """
        features = []
        
        # Extract left hand features (63)
        if detection_results['left_hand']:
            for lm in detection_results['left_hand']['landmarks']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * 63)
        
        # Extract right hand features (63)
        if detection_results['right_hand']:
            for lm in detection_results['right_hand']['landmarks']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * 63)
        
        # Extract pose features (15)
        if detection_results['pose']:
            for lm in detection_results['pose']['landmarks']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * 15)
        
        # Extract spatial features (10)
        spatial = detection_results['spatial_features']
        features.extend([
            float(spatial.get('has_left_hand', 0)),
            float(spatial.get('has_right_hand', 0)),
            float(spatial.get('has_both_hands', 0)),
            spatial.get('hands_distance', 0.0),
            spatial.get('hands_vertical_offset', 0.0),
            spatial.get('hands_horizontal_offset', 0.0),
            spatial.get('left_hand_to_torso', {}).get('distance', 0.0),
            spatial.get('left_hand_to_torso', {}).get('offset_y', 0.0),
            spatial.get('right_hand_to_torso', {}).get('distance', 0.0),
            spatial.get('right_hand_to_torso', {}).get('offset_y', 0.0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def close(self):
        """Clean up resources"""
        self.hands.close()
        self.pose.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
