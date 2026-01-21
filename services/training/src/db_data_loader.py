"""
Database-based data loader for ASL gesture training
Loads training samples from database (TrainingSample table)
"""
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Add database path - use mounted API source
sys.path.insert(0, '/app/api_src')
sys.path.insert(0, '/app/inference_src')  # For feature extractor

from database import init_db, get_session, TrainingSample
from feature_extractor import EnhancedFeatureExtractor


class DatabaseDataLoader:
    """Load training data from database"""
    
    def __init__(self, database_url=None, use_enhanced_features=True):
        """
        Initialize data loader
        
        Args:
            database_url: Optional database URL (default: uses DATABASE_URL env or default)
            use_enhanced_features: Use EnhancedFeatureExtractor (79 features) vs raw landmarks (63 features)
        """
        self.engine = init_db(database_url)
        self.use_enhanced_features = use_enhanced_features
        
        if use_enhanced_features:
            self.feature_extractor = EnhancedFeatureExtractor(include_orientation=True)
            print(f"Using EnhancedFeatureExtractor ({self.feature_extractor.get_feature_count()} features)")
        else:
            self.feature_extractor = None
            print("Using raw landmarks (63 features)")
    
    def load_dataset(self, source_filter=None, include_feedback=True):
        """
        Load training dataset from database
        
        Args:
            source_filter: Optional list of sources to include (e.g., ['original', 'feedback'])
            include_feedback: Include feedback samples (default: True)
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        session = get_session(self.engine)
        
        try:
            # Build query
            query = session.query(TrainingSample)
            
            if source_filter:
                query = query.filter(TrainingSample.source.in_(source_filter))
            elif not include_feedback:
                query = query.filter(TrainingSample.source == 'original')
            
            # Fetch all samples
            samples = query.all()
            
            if len(samples) == 0:
                print("Warning: No training samples found in database")
                return np.array([]), np.array([])
            
            # Convert to numpy arrays
            features = []
            labels = []
            
            for sample in samples:
                if sample.landmarks and len(sample.landmarks) == 63:
                    if self.use_enhanced_features:
                        # Convert raw landmarks to enhanced features
                        landmarks_dict = self._convert_landmarks_to_dict(sample.landmarks)
                        enhanced_features = self.feature_extractor.extract_features(landmarks_dict)
                        features.append(enhanced_features)
                    else:
                        # Use raw landmarks
                        features.append(sample.landmarks)
                    labels.append(sample.gesture)
            
            features = np.array(features)
            labels = np.array(labels)
            
            print(f"Loaded {len(features)} samples from database")
            
            return features, labels
            
        finally:
            session.close()
    
    def get_dataset_info(self):
        """Get information about dataset in database"""
        session = get_session(self.engine)
        
        try:
            from sqlalchemy import func
            
            # Total samples
            total = session.query(TrainingSample).count()
            
            # Samples by gesture
            gesture_counts = session.query(
                TrainingSample.gesture,
                func.count(TrainingSample.id)
            ).group_by(TrainingSample.gesture).all()
            
            # Samples by source
            source_counts = session.query(
                TrainingSample.source,
                func.count(TrainingSample.id)
            ).group_by(TrainingSample.source).all()
            
            gestures = [g for g, _ in gesture_counts]
            samples_per_gesture = {g: c for g, c in gesture_counts}
            samples_per_source = {s: c for s, c in source_counts}
            
            return {
                'total_samples': total,
                'gestures': sorted(gestures),
                'samples_per_gesture': samples_per_gesture,
                'samples_per_source': samples_per_source
            }
            
        finally:
            session.close()
    
    def load_feedback_samples(self):
        """Load only feedback samples"""
        return self.load_dataset(source_filter=['feedback'])
    
    def load_original_samples(self):
        """Load only original training samples"""
        return self.load_dataset(source_filter=['original'])
    
    def _convert_landmarks_to_dict(self, landmarks_list):
        """Convert flat landmark list to dict format for feature extraction"""
        landmarks = []
        for i in range(0, len(landmarks_list), 3):
            landmarks.append({
                'x': landmarks_list[i],
                'y': landmarks_list[i+1],
                'z': landmarks_list[i+2]
            })
        return {'landmarks': landmarks}
