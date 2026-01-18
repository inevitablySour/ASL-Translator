#!/usr/bin/env python3
"""
Database initialization script
Automatically migrates data from SQLite to PostgreSQL if PostgreSQL is empty
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, User, Model, Prediction, Feedback, TrainingSample, TrainingRun, get_database_url
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLITE_BACKUP = '/app/data/feedback.db'


def is_database_empty(session):
    """Check if PostgreSQL database is empty"""
    # Check if any core tables have data
    model_count = session.query(Model).count()
    prediction_count = session.query(Prediction).count()
    training_count = session.query(TrainingSample).count()
    
    return model_count == 0 and prediction_count == 0 and training_count == 0


def migrate_from_sqlite():
    """Migrate data from SQLite backup to PostgreSQL"""
    logger.info("Starting automatic migration from SQLite backup...")
    
    # Check if SQLite backup exists
    if not Path(SQLITE_BACKUP).exists():
        logger.warning(f"SQLite backup not found at {SQLITE_BACKUP}")
        logger.info("Starting with empty database")
        return
    
    try:
        # Connect to SQLite
        sqlite_url = f'sqlite:///{SQLITE_BACKUP}'
        logger.info(f"Connecting to SQLite: {sqlite_url}")
        sqlite_engine = create_engine(sqlite_url, echo=False)
        SqliteSession = sessionmaker(bind=sqlite_engine)
        sqlite_session = SqliteSession()
        
        # Connect to PostgreSQL
        postgres_url = get_database_url()
        logger.info(f"Connecting to PostgreSQL...")
        postgres_engine = create_engine(postgres_url, echo=False)
        PostgresSession = sessionmaker(bind=postgres_engine)
        postgres_session = PostgresSession()
        
        # Migration order (respects foreign key constraints)
        tables = [
            (User, "Users"),
            (Model, "Models"),
            (Prediction, "Predictions"),
            (Feedback, "Feedback"),
            (TrainingSample, "Training Samples"),
            (TrainingRun, "Training Runs")
        ]
        
        total_migrated = 0
        
        for table_class, table_name in tables:
            # Query all records from SQLite
            records = sqlite_session.query(table_class).all()
            count = len(records)
            
            if count == 0:
                continue
            
            logger.info(f"Migrating {count} {table_name}...")
            
            # Bulk insert into PostgreSQL
            for record in records:
                try:
                    # Convert to dict
                    record_dict = {}
                    for column in table_class.__table__.columns:
                        record_dict[column.name] = getattr(record, column.name)
                    
                    # Create new object
                    new_record = table_class(**record_dict)
                    postgres_session.add(new_record)
                    
                except Exception as e:
                    logger.error(f"Error migrating {table_name} record: {e}")
                    postgres_session.rollback()
                    continue
            
            # Commit batch
            try:
                postgres_session.commit()
                logger.info(f"✓ Migrated {count} {table_name}")
                total_migrated += count
            except Exception as e:
                logger.error(f"Error committing {table_name}: {e}")
                postgres_session.rollback()
        
        sqlite_session.close()
        postgres_session.close()
        
        logger.info(f"✓ Migration complete! Total records: {total_migrated}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()


def init_database():
    """Initialize database and auto-migrate if needed"""
    logger.info("Initializing database...")
    
    # Get database URL
    db_url = get_database_url()
    logger.info(f"Database URL: {db_url.split('@')[0]}@***")  # Hide password
    
    # Create engine and tables
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    logger.info("✓ Database tables created")
    
    # Check if we need to migrate
    Session = sessionmaker(bind=engine)
    session = Session()
    
    if is_database_empty(session):
        logger.info("Database is empty, checking for SQLite backup...")
        session.close()
        migrate_from_sqlite()
    else:
        logger.info("Database already contains data, skipping migration")
        session.close()


if __name__ == '__main__':
    try:
        init_database()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
