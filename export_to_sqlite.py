#!/usr/bin/env python3
"""
Export data from PostgreSQL to SQLite
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

def export_data():
    """Export all data from PostgreSQL to SQLite"""
    print("=" * 60)
    print("ASL Translator: PostgreSQL to SQLite Export")
    print("=" * 60)
    print()
    
    # Connect to PostgreSQL
    print(f"Connecting to PostgreSQL: {POSTGRES_URL}")
    postgres_engine = create_engine(POSTGRES_URL, echo=False)
    PostgresSession = sessionmaker(bind=postgres_engine)
    postgres_session = PostgresSession()
    
    # Connect to SQLite
    print(f"Connecting to SQLite: {SQLITE_URL}")
    sqlite_engine = create_engine(SQLITE_URL, echo=False)
    
    # Create all tables in SQLite
    print("Creating SQLite tables...")
    Base.metadata.create_all(sqlite_engine)
    
    SqliteSession = sessionmaker(bind=sqlite_engine)
    sqlite_session = SqliteSession()
    
    # Clear existing SQLite data
    print("Clearing existing SQLite data...")
    for table in reversed(Base.metadata.sorted_tables):
        sqlite_session.execute(table.delete())
    sqlite_session.commit()
    
    # Export order (to respect foreign key constraints)
    tables = [
        (User, "Users"),
        (Model, "Models"),
        (Prediction, "Predictions"),
        (Feedback, "Feedback"),
        (TrainingSample, "Training Samples"),
        (TrainingRun, "Training Runs")
    ]
    
    print()
    print("Starting data export...")
    print("-" * 60)
    
    total_exported = 0
    
    for table_class, table_name in tables:
        print(f"\nExporting {table_name}...")
        
        # Query all records from PostgreSQL
        records = postgres_session.query(table_class).all()
        count = len(records)
        
        if count == 0:
            print(f"  No records to export")
            continue
        
        print(f"  Found {count} records")
        
        # Bulk insert into SQLite
        exported = 0
        failed = 0
        
        for record in records:
            try:
                # Convert SQLAlchemy object to dict
                record_dict = {}
                for column in table_class.__table__.columns:
                    record_dict[column.name] = getattr(record, column.name)
                
                # Create new object for SQLite
                new_record = table_class(**record_dict)
                sqlite_session.add(new_record)
                exported += 1
                
                # Commit in batches
                if exported % 100 == 0:
                    sqlite_session.commit()
                    print(f"  Exported {exported}/{count}...", end='\r')
            
            except Exception as e:
                failed += 1
                print(f"\n  Error exporting record: {e}")
                sqlite_session.rollback()
        
        # Final commit
        try:
            sqlite_session.commit()
            print(f"  ✓ Successfully exported {exported} records")
            if failed > 0:
                print(f"  ✗ Failed to export {failed} records")
            total_exported += exported
        except Exception as e:
            print(f"  ✗ Error committing batch: {e}")
            sqlite_session.rollback()
    
    # Close connections
    postgres_session.close()
    sqlite_session.close()
    
    print()
    print("-" * 60)
    print(f"Export complete! Total records exported: {total_exported}")
    print()
    
    # Verify export
    print("Verifying export...")
    sqlite_session = SqliteSession()
    
    for table_class, table_name in tables:
        count = sqlite_session.query(table_class).count()
        print(f"  {table_name}: {count} records")
    
    sqlite_session.close()
    print()
    print("=" * 60)


if __name__ == '__main__':
    try:
        export_data()
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
