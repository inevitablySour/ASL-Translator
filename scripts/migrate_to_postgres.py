#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL
"""
import sys
from pathlib import Path

# Add API source to path (scripts/ is one level below project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'services' / 'api' / 'src'))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
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
        
        # Upsert records into PostgreSQL (insert or update on conflict)
        migrated = 0
        updated = 0
        failed = 0
        
        # Get primary key columns for the table
        pk_columns = [key.name for key in inspect(table_class).primary_key]
        
        # Process in batches
        batch_size = 100
        for i in range(0, count, batch_size):
            batch = records[i:i + batch_size]
            
            try:
                for record in batch:
                    # Convert SQLAlchemy object to dict
                    record_dict = {}
                    for column in table_class.__table__.columns:
                        record_dict[column.name] = getattr(record, column.name)
                    
                    # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE (upsert)
                    stmt = insert(table_class.__table__).values(record_dict)
                    
                    # Create update dict excluding primary keys
                    update_dict = {k: v for k, v in record_dict.items() if k not in pk_columns}
                    
                    if update_dict:
                        # Update all columns except primary keys on conflict
                        stmt = stmt.on_conflict_do_update(
                            index_elements=pk_columns,
                            set_=update_dict
                        )
                    else:
                        # If only primary keys exist, do nothing on conflict
                        stmt = stmt.on_conflict_do_nothing(index_elements=pk_columns)
                    
                    postgres_session.execute(stmt)
                
                # Commit batch
                postgres_session.commit()
                batch_migrated = len(batch)
                migrated += batch_migrated
                print(f"  Migrated {migrated}/{count}...", end='\r')
            
            except Exception as e:
                failed += len(batch)
                print(f"\n  Error migrating batch: {e}")
                postgres_session.rollback()
        
        print(f"\n  ✓ Successfully migrated/updated {migrated} records")
        if failed > 0:
            print(f"  ✗ Failed to migrate {failed} records")
        total_migrated += migrated
    
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
