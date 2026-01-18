"""
Automated retraining worker for ASL Translator
Monitors feedback database and triggers retraining when threshold is reached
"""
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add paths to find modules
sys.path.insert(0, '/app/api_src')  # Mounted API source
sys.path.insert(0, '/app/src')  # Training service source

from feedback_manager import FeedbackManager
from database import get_session, Model, TrainingRun
from db_data_loader import DatabaseDataLoader
from model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingWorker:
    """Worker that monitors feedback and triggers model retraining"""
    
    def __init__(self, check_interval=300, retraining_threshold=200):
        """
        Initialize retraining worker
        
        Args:
            check_interval: Seconds between checks (default 5 minutes)
            retraining_threshold: Number of feedback samples needed
        """
        self.check_interval = check_interval
        self.feedback_manager = FeedbackManager(retraining_threshold=retraining_threshold)
        self.project_root = Path(__file__).parent.parent.parent
        logger.info(f"RetrainingWorker initialized (interval={check_interval}s, threshold={retraining_threshold})")
    
    def should_retrain(self) -> bool:
        """Check if we have enough feedback to trigger retraining"""
        unused_count = self.feedback_manager.get_unused_feedback_count()
        threshold = self.feedback_manager.retraining_threshold
        logger.info(f"Unused feedback: {unused_count}/{threshold}")
        return unused_count >= threshold
    
    def trigger_retraining(self):
        """Export feedback data and trigger model retraining"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL RETRAINING")
            logger.info("=" * 60)
            
            # Create training run record
            session = get_session()
            training_run = TrainingRun(
                samples_used=0,
                feedback_samples=0,
                status='running'
            )
            session.add(training_run)
            session.commit()
            run_id = training_run.id
            logger.info(f"Created training run: {run_id}")
            session.close()
            
            # Count feedback samples available for training
            session = get_session()
            unused_count = self.feedback_manager.get_unused_feedback_count()
            logger.info(f"Using {unused_count} feedback samples for training")
            
            # Update training run with sample count
            training_run = session.query(TrainingRun).filter_by(id=run_id).first()
            training_run.feedback_samples = unused_count
            session.commit()
            session.close()
            
            # Train new model directly using database
            model_name = f"asl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Training new model: {model_name} (using database)")
            
            # Load data from database
            logger.info("Loading training data from database...")
            data_loader = DatabaseDataLoader()
            features, labels = data_loader.load_dataset()
            
            if len(features) == 0:
                raise Exception("No training data found in database")
            
            # Show dataset info
            info = data_loader.get_dataset_info()
            logger.info(f"Total samples: {info['total_samples']}")
            logger.info(f"Gestures: {len(info['gestures'])}")
            if 'samples_per_source' in info:
                for source, count in info['samples_per_source'].items():
                    logger.info(f"  {source}: {count} samples")
            
            # Initialize trainer
            trainer = ModelTrainer(model_type="random_forest", model_name=model_name)
            
            # Prepare data
            X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)
            
            # Hyperparameters
            hyperparameters = {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
            
            # Train with MLflow tracking
            metrics = trainer.train_with_mlflow(X_train, X_test, y_train, y_test, hyperparameters)
            
            logger.info("Training completed successfully!")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            
            total_samples = info['total_samples']
            
            # Create model record in database
            model_path = Path("/app/models") / model_name / "asl_classifier.pkl"
            
            session = get_session()
            model = Model(
                version=model_name,
                name=f"ASL Classifier {model_name}",
                file_path=str(model_path),
                accuracy=metrics['accuracy'],
                is_active=False,  # Don't auto-activate, let admin review
                model_metadata={
                    "trained_from_feedback": True,
                    "feedback_samples": unused_count,
                    "total_samples": total_samples,
                    "training_run_id": run_id
                }
            )
            session.add(model)
            session.commit()
            model_id = model.id
            logger.info(f"Created model record: {model_id}")
            
            # Update training run
            training_run = session.query(TrainingRun).filter_by(id=run_id).first()
            training_run.model_id = model_id
            training_run.samples_used = total_samples
            training_run.status = 'completed'
            training_run.completed_at = datetime.utcnow()
            session.commit()
            session.close()
            
            # Mark feedback as used
            logger.info("Marking feedback as used...")
            self.feedback_manager.mark_feedback_as_used()
            
            logger.info("=" * 60)
            logger.info(f"RETRAINING COMPLETE - Model: {model_name}")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "model_name": model_name,
                "model_id": model_id,
                "samples_used": unused_count
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def run(self):
        """Main loop - check for feedback and retrain when threshold reached"""
        logger.info("Starting retraining worker...")
        logger.info(f"Will check every {self.check_interval} seconds")
        
        while True:
            try:
                if self.should_retrain():
                    logger.info("Retraining threshold reached!")
                    result = self.trigger_retraining()
                    
                    if result["success"]:
                        logger.info(f"Successfully trained model: {result['model_name']}")
                    else:
                        logger.error(f"Retraining failed: {result.get('error')}")
                else:
                    logger.debug("Threshold not reached yet, waiting...")
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down retraining worker...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(self.check_interval)


def main():
    """Entry point for retraining worker"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Model Retraining Worker')
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=200,
        help='Feedback samples needed to trigger retraining (default: 200)'
    )
    
    args = parser.parse_args()
    
    worker = RetrainingWorker(
        check_interval=args.interval,
        retraining_threshold=args.threshold
    )
    worker.run()


if __name__ == "__main__":
    main()
