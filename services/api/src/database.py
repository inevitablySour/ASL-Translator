"""
Database models and connection for ASL Translator feedback system
"""
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid
import os

Base = declarative_base()


def generate_uuid():
    """Generate UUID as string"""
    return str(uuid.uuid4())


class User(Base):
    """User accounts (for future authentication)"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")


class Model(Base):
    """Model versions and metadata"""
    __tablename__ = 'models'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    version = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    file_path = Column(String, nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)
    model_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")


class Prediction(Base):
    """Prediction logs from the API"""
    __tablename__ = 'predictions'
    
    job_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=True)
    model_id = Column(String, ForeignKey('models.id'), nullable=True)
    gesture = Column(String, nullable=False)
    translation = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    language = Column(String, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    landmarks = Column(JSON, nullable=True)  # Hand landmarks for high-confidence predictions
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="predictions")
    model = relationship("Model", back_populates="predictions")
    feedback = relationship("Feedback", back_populates="prediction", uselist=False)


class Feedback(Base):
    """User feedback on predictions"""
    __tablename__ = 'feedback'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    prediction_id = Column(String, ForeignKey('predictions.job_id'), nullable=False)
    user_id = Column(String, ForeignKey('users.id'), nullable=True)
    accepted = Column(Boolean, nullable=False)  # Did user accept the prediction?
    feedback_type = Column(String, nullable=True)  # 'correct', 'incorrect', 'unsure'
    corrected_gesture = Column(String, nullable=True)  # If user corrects the gesture
    used_for_training = Column(Boolean, default=False)  # Has this been used in retraining?
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="feedback")
    user = relationship("User", back_populates="feedback")


class TrainingSample(Base):
    """Training samples for model training (original data collection + feedback)"""
    __tablename__ = 'training_samples'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    gesture = Column(String, nullable=False, index=True)
    landmarks = Column(JSON, nullable=False)  # Hand landmarks [x,y,z, ...]
    source = Column(String, default='original')  # 'original', 'feedback', 'synthetic'
    source_id = Column(String, nullable=True)  # Reference to prediction_id if from feedback
    confidence = Column(Float, nullable=True)  # Confidence if from prediction
    collection_date = Column(DateTime, default=datetime.utcnow)
    sample_metadata = Column(JSON, nullable=True)  # Additional metadata (e.g., original filename)
    used_in_training = Column(Boolean, default=False, index=True)  # Track which samples were used
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    """Track model retraining runs"""
    __tablename__ = 'training_runs'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey('models.id'), nullable=True)
    samples_used = Column(Integer, nullable=False)
    feedback_samples = Column(Integer, nullable=False)
    mlflow_run_id = Column(String, nullable=True)
    status = Column(String, default='pending')  # pending, running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    
# Database connection and session management
def get_database_url():
    """Get database URL from environment or use default"""
    default_url = os.getenv('DATABASE_URL')
    
    if default_url:
        return default_url
    
    # Auto-detect: Docker container vs local
    if os.path.exists('/app/data'):
        # Running in Docker container
        return 'sqlite:////app/data/feedback.db'
    else:
        # Running locally - use relative path from project root
        # Find project root (has data/ directory)
        from pathlib import Path
        current = Path(__file__).parent
        while current.parent != current:
            if (current / 'data').exists():
                db_path = current / 'data' / 'feedback.db'
                return f'sqlite:///{db_path}'
            current = current.parent
        
        # Fallback to relative path
        return 'sqlite:///data/feedback.db'


def init_db(database_url=None):
    """Initialize database and create tables"""
    if database_url is None:
        database_url = get_database_url()
    
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    """Get a database session"""
    if engine is None:
        engine = init_db()
    Session = sessionmaker(bind=engine)
    return Session()


