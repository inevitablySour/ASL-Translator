#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL
"""
import sys
from pathlib import Path

# Add API source to path
sys.path.insert(0, str(Path(__file__).parent / 'services' / 'api' / 'src'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, User, Model, Prediction, Feedback, TrainingSample, TrainingRun
import os

# Database URLs
SQLITE_URL = 'sqlite:///data/feedback.db'
POSTGRES_URL = os.getenv('DATABASE_URL', 'postgresql://asl_user:asl_password@localhost:5432/asl_translator')

def migrate_data():
    """Migrate all data from SQLite to PostgreSQL"""
    print("=" * 60)
    print("ASL Translator: SQLite to PostgreSQL Migration")
    print("=" * 60)
    print()
    
    # Connect to SQLite
    print(f"Connecting to SQLite: {SQLITE_URL}")
    sqlite_engine = create_engine(SQLITE_URL, echo=False)
    SqliteSession = sessionmaker(bind=sqlite_engine)
    sqlite_session = SqliteSession()
    
    # Connect to PostgreSQL
    print(f"Connecting to PostgreSQL: {POSTGRES_URL}")
    postgres_engine = create_engine(POSTGRES_URL, echo=False)
    
    # Create all tables in PostgreSQL
    print("Creating PostgreSQL tables...")
    Base.metadata.create_all(postgres_engine)
    
    PostgresSession = sessionmaker(bind=postgres_engine)
    postgres_session = PostgresSession()
    
    # Migration order (to respect foreign key constraints)
    tables = [
        (User, "Users"),
        (Model, "Models"),
        (Prediction, "Predictions"),
        (Feedback, "Feedback"),
        (TrainingSample, "Training Samples"),
        (TrainingRun, "Training Runs")
    ]
    
    print()
    print("Starting data migration...")
    print("-" * 60)
    
    total_migrated = 0
    
    for table_class, table_name in tables:
        print(f"\nMigrating {table_name}...")
        
        # Query all records from SQLite
        records = sqlite_session.query(table_class).all()
        count = len(records)
        
        if count == 0:
            print(f"  No records to migrate")
            continue
        
        print(f"  Found {count} records")
        
        # Bulk insert into PostgreSQL
        migrated = 0
        failed = 0
        
        for record in records:
            try:
                # Convert SQLAlchemy object to dict
                record_dict = {}
                for column in table_class.__table__.columns:
                    record_dict[column.name] = getattr(record, column.name)
                
                # Create new object for PostgreSQL
                new_record = table_class(**record_dict)
                postgres_session.add(new_record)
                migrated += 1
                
                # Commit in batches
                if migrated % 100 == 0:
                    postgres_session.commit()
                    print(f"  Migrated {migrated}/{count}...", end='\r')
            
            except Exception as e:
                failed += 1
                print(f"\n  Error migrating record: {e}")
                postgres_session.rollback()
        
        # Final commit
        try:
            postgres_session.commit()
            print(f"  ✓ Successfully migrated {migrated} records")
            if failed > 0:
                print(f"  ✗ Failed to migrate {failed} records")
            total_migrated += migrated
        except Exception as e:
            print(f"  ✗ Error committing batch: {e}")
            postgres_session.rollback()
    
    # Close connections
    sqlite_session.close()
    postgres_session.close()
    
    print()
    print("-" * 60)
    print(f"Migration complete! Total records migrated: {total_migrated}")
    print()
    
    # Verify migration
    print("Verifying migration...")
    postgres_session = PostgresSession()
    
    for table_class, table_name in tables:
        count = postgres_session.query(table_class).count()
        print(f"  {table_name}: {count} records")
    
    postgres_session.close()
    print()
    print("=" * 60)


if __name__ == '__main__':
    try:
        migrate_data()
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
