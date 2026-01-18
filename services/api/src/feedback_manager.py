"""
Feedback manager for collecting user feedback and triggering model retraining
"""
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, List
from database import (
    get_session, init_db, 
    Prediction, Feedback, Model, TrainingRun, TrainingSample
)

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages feedback collection and retraining triggers"""
    
    def __init__(self, retraining_threshold=200, feedback_data_dir="data/gestures/feedback"):
        self.retraining_threshold = retraining_threshold
        self.feedback_data_dir = Path(feedback_data_dir)
        self.feedback_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = init_db()
        logger.info(f"FeedbackManager initialized with threshold={retraining_threshold}")
    
    def store_prediction(self, job_id: str, gesture: str, translation: str, 
                        confidence: float, language: str, processing_time_ms: float,
                        landmarks: Optional[List[float]] = None, 
                        model_id: Optional[str] = None) -> bool:
        """
        Store a prediction in the database
        
        Args:
            job_id: Unique prediction ID
            gesture: Predicted gesture
            translation: Translated text
            confidence: Confidence score (0-1)
            language: Language code
            processing_time_ms: Processing time
            landmarks: Hand landmarks (if confidence >= threshold)
            model_id: Model version ID that made the prediction
            
        Returns:
            True if stored successfully
        """
        try:
            session = get_session(self.engine)
            
            prediction = Prediction(
                job_id=job_id,
                gesture=gesture,
                translation=translation,
                confidence=confidence,
                language=language,
                processing_time_ms=processing_time_ms,
                landmarks=landmarks,
                model_id=model_id
            )
            
            session.add(prediction)
            session.commit()
            logger.info(f"Stored prediction {job_id} (gesture={gesture}, conf={confidence:.2f})")
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction {job_id}: {e}", exc_info=True)
            session.rollback()
            session.close()
            return False
    
    def submit_feedback(self, job_id: str, accepted: bool, 
                       corrected_gesture: Optional[str] = None) -> Dict:
        """
        Submit user feedback for a prediction
        
        Args:
            job_id: Prediction job ID
            accepted: Whether user accepted the prediction
            corrected_gesture: If rejected, the correct gesture
            
        Returns:
            Dict with status and info
        """
        try:
            session = get_session(self.engine)
            
            # Check if prediction exists
            prediction = session.query(Prediction).filter_by(job_id=job_id).first()
            if not prediction:
                logger.warning(f"Prediction {job_id} not found for feedback")
                session.close()
                return {"success": False, "error": "Prediction not found"}
            
            # Check if landmarks exist (needed for training)
            if not prediction.landmarks:
                logger.warning(f"No landmarks available for prediction {job_id}")
                session.close()
                return {"success": False, "error": "No landmarks available for training"}
            
            # Create feedback entry
            feedback_type = "correct" if accepted else "incorrect"
            feedback = Feedback(
                prediction_id=job_id,
                accepted=accepted,
                feedback_type=feedback_type,
                corrected_gesture=corrected_gesture,
                used_for_training=False
            )
            
            session.add(feedback)
            session.commit()
            
            logger.info(f"Feedback stored for {job_id}: accepted={accepted}")
            
            # If feedback is accepted, also save to TrainingSample for training
            if accepted and prediction.landmarks:
                gesture_for_training = corrected_gesture if corrected_gesture else prediction.gesture
                try:
                    training_sample = TrainingSample(
                        gesture=gesture_for_training,
                        landmarks=prediction.landmarks,
                        source='feedback',
                        source_id=job_id,
                        confidence=prediction.confidence,
                        collection_date=prediction.created_at,
                        sample_metadata={
                            'feedback_id': feedback.id,
                            'original_gesture': prediction.gesture,
                            'corrected': corrected_gesture is not None
                        }
                    )
                    session.add(training_sample)
                    session.commit()
                    logger.info(f"Added feedback to training samples: {gesture_for_training}")
                except Exception as e:
                    logger.error(f"Failed to save training sample: {e}")
                    # Don't fail the feedback submission if training sample fails
            
            # Check if we need to trigger retraining
            should_retrain = self.check_retraining_threshold(session)
            
            result = {
                "success": True,
                "feedback_id": feedback.id,
                "should_retrain": should_retrain
            }
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error submitting feedback for {job_id}: {e}", exc_info=True)
            session.rollback()
            session.close()
            return {"success": False, "error": str(e)}
    
    def check_retraining_threshold(self, session) -> bool:
        """
        Check if we have enough unused feedback samples to trigger retraining
        
        Args:
            session: Database session
            
        Returns:
            True if threshold is reached
        """
        # Count feedback that hasn't been used for training yet
        unused_count = session.query(Feedback).filter_by(used_for_training=False).count()
        
        logger.info(f"Unused feedback samples: {unused_count}/{self.retraining_threshold}")
        
        return unused_count >= self.retraining_threshold
    
    def get_unused_feedback_count(self) -> int:
        """Get count of feedback samples not yet used for training"""
        try:
            session = get_session(self.engine)
            count = session.query(Feedback).filter_by(used_for_training=False).count()
            session.close()
            return count
        except Exception as e:
            logger.error(f"Error getting unused feedback count: {e}")
            return 0
    
    def export_feedback_for_training(self) -> Dict:
        """
        Export unused feedback data to JSON files for training
        
        Returns:
            Dict with export info (file paths, count, etc.)
        """
        try:
            session = get_session(self.engine)
            
            # Get all unused feedback with predictions
            feedbacks = session.query(Feedback).filter_by(used_for_training=False).all()
            
            if not feedbacks:
                logger.info("No unused feedback to export")
                session.close()
                return {"success": True, "count": 0, "message": "No data to export"}
            
            # Create export directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = self.feedback_data_dir / f"export_{timestamp}"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Group by gesture
            gesture_data = {}
            exported_count = 0
            
            for feedback in feedbacks:
                prediction = feedback.prediction
                
                # Determine which gesture to use
                gesture = feedback.corrected_gesture if feedback.corrected_gesture else prediction.gesture
                
                # Skip if no landmarks
                if not prediction.landmarks:
                    logger.warning(f"Skipping feedback {feedback.id}: no landmarks")
                    continue
                
                if gesture not in gesture_data:
                    gesture_data[gesture] = []
                
                # Create data entry
                entry = {
                    "gesture": gesture,
                    "landmarks": prediction.landmarks,
                    "timestamp": prediction.created_at.isoformat(),
                    "confidence": prediction.confidence,
                    "feedback_id": feedback.id,
                    "source": "user_feedback"
                }
                
                gesture_data[gesture].append(entry)
                exported_count += 1
            
            # Save to JSON files (one per gesture)
            saved_files = []
            for gesture, samples in gesture_data.items():
                gesture_dir = export_dir / gesture
                gesture_dir.mkdir(exist_ok=True)
                
                for i, sample in enumerate(samples):
                    filename = gesture_dir / f"{gesture}_feedback_{timestamp}_{i:04d}.json"
                    with open(filename, 'w') as f:
                        json.dump(sample, f, indent=2)
                    saved_files.append(str(filename))
            
            logger.info(f"Exported {exported_count} feedback samples to {export_dir}")
            
            session.close()
            
            return {
                "success": True,
                "count": exported_count,
                "export_dir": str(export_dir),
                "gestures": list(gesture_data.keys()),
                "files": saved_files
            }
            
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def mark_feedback_as_used(self, feedback_ids: Optional[List[str]] = None):
        """
        Mark feedback samples as used for training
        
        Args:
            feedback_ids: List of feedback IDs to mark, or None to mark all unused
        """
        try:
            session = get_session(self.engine)
            
            if feedback_ids:
                feedbacks = session.query(Feedback).filter(Feedback.id.in_(feedback_ids)).all()
            else:
                feedbacks = session.query(Feedback).filter_by(used_for_training=False).all()
            
            for feedback in feedbacks:
                feedback.used_for_training = True
            
            session.commit()
            logger.info(f"Marked {len(feedbacks)} feedback samples as used")
            session.close()
            
        except Exception as e:
            logger.error(f"Error marking feedback as used: {e}", exc_info=True)
            session.rollback()
            session.close()
    
    def get_stats(self) -> Dict:
        """Get feedback statistics"""
        try:
            session = get_session(self.engine)
            
            total_predictions = session.query(Prediction).count()
            total_feedback = session.query(Feedback).count()
            unused_feedback = session.query(Feedback).filter_by(used_for_training=False).count()
            accepted_feedback = session.query(Feedback).filter_by(accepted=True).count()
            
            stats = {
                "total_predictions": total_predictions,
                "total_feedback": total_feedback,
                "unused_feedback": unused_feedback,
                "used_feedback": total_feedback - unused_feedback,
                "accepted_feedback": accepted_feedback,
                "rejected_feedback": total_feedback - accepted_feedback,
                "threshold": self.retraining_threshold,
                "progress_to_threshold": f"{unused_feedback}/{self.retraining_threshold}"
            }
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
