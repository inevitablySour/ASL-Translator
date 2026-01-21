#!/usr/bin/env python3
"""
Check if SQLite has newer data than PostgreSQL and sync if needed
"""
import sys
from pathlib import Path

# Add API source to path (scripts/ is one level below project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'services' / 'api' / 'src'))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from database import Base, User, Model, Prediction, Feedback, TrainingSample, TrainingRun
import os

# Database URLs
SQLITE_URL = 'sqlite:///data/feedback.db'
POSTGRES_URL = os.getenv('DATABASE_URL', 'postgresql://asl_user:asl_password@localhost:5432/asl_translator')

def get_latest_timestamp(session, tables):
    """Get the most recent timestamp from all tables"""
    latest = None
    
    for table_class, _ in tables:
        # Check if table has a timestamp column (created_at, updated_at, etc.)
        timestamp_cols = []
        for col in table_class.__table__.columns:
            if 'timestamp' in col.name.lower() or 'created' in col.name.lower() or 'updated' in col.name.lower():
                timestamp_cols.append(col)
        
        if timestamp_cols:
            for col in timestamp_cols:
                result = session.query(func.max(col)).scalar()
                if result and (latest is None or result > latest):
                    latest = result
    
    return latest

def get_record_counts(session, tables):
    """Get total record count across all tables"""
    total = 0
    for table_class, _ in tables:
        total += session.query(table_class).count()
    return total

def needs_sync():
    """Check if SQLite has newer or more data than PostgreSQL"""
    tables = [
        (User, "Users"),
        (Model, "Models"),
        (Prediction, "Predictions"),
        (Feedback, "Feedback"),
        (TrainingSample, "Training Samples"),
        (TrainingRun, "Training Runs")
    ]
    
    try:
        # Connect to SQLite
        sqlite_engine = create_engine(SQLITE_URL, echo=False)
        SqliteSession = sessionmaker(bind=sqlite_engine)
        sqlite_session = SqliteSession()
        
        # Connect to PostgreSQL
        postgres_engine = create_engine(POSTGRES_URL, echo=False)
        
        # Check if PostgreSQL tables exist
        if not postgres_engine.dialect.has_table(postgres_engine.connect(), 'users'):
            print("PostgreSQL tables don't exist - will sync")
            sqlite_session.close()
            return True
        
        PostgresSession = sessionmaker(bind=postgres_engine)
        postgres_session = PostgresSession()
        
        # Get record counts
        sqlite_count = get_record_counts(sqlite_session, tables)
        postgres_count = get_record_counts(postgres_session, tables)
        
        print(f"SQLite records: {sqlite_count}")
        print(f"PostgreSQL records: {postgres_count}")
        
        # If SQLite has more records, sync
        if sqlite_count > postgres_count:
            print("SQLite has more records - will sync")
            sqlite_session.close()
            postgres_session.close()
            return True
        
        # If counts are equal, check timestamps
        if sqlite_count == postgres_count and sqlite_count > 0:
            sqlite_latest = get_latest_timestamp(sqlite_session, tables)
            postgres_latest = get_latest_timestamp(postgres_session, tables)
            
            if sqlite_latest and postgres_latest:
                print(f"SQLite latest: {sqlite_latest}")
                print(f"PostgreSQL latest: {postgres_latest}")
                
                if sqlite_latest > postgres_latest:
                    print("SQLite has newer data - will sync")
                    sqlite_session.close()
                    postgres_session.close()
                    return True
        
        print("PostgreSQL is up-to-date - no sync needed")
        sqlite_session.close()
        postgres_session.close()
        return False
        
    except Exception as e:
        print(f"Error checking databases: {e}")
        print("Will attempt sync to be safe")
        return True

if __name__ == '__main__':
    # Exit code 0 means sync needed, 1 means no sync needed
    sys.exit(0 if needs_sync() else 1)
