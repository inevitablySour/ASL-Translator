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

from database import init_db, get_session, TrainingSample


class DatabaseDataLoader:
    """Load training data from database"""
    
    def __init__(self, database_url=None):
        """
        Initialize data loader
        
        Args:
            database_url: Optional database URL (default: uses DATABASE_URL env or default)
        """
        self.engine = init_db(database_url)
    
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
