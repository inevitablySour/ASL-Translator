"""
Migrate existing JSON training data to database
Imports all gesture JSON files from data/gestures into TrainingSample table
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "services" / "api" / "src"))

from database import init_db, get_session, TrainingSample


def parse_timestamp(filename):
    """Extract timestamp from filename like 'A_20260118_131816_419.json'"""
    try:
        parts = filename.stem.split('_')
        if len(parts) >= 4:
            date_str = parts[1]  # 20260118
            time_str = parts[2]  # 131816
            ms_str = parts[3]    # 419
            
            # Parse as datetime
            dt_str = f"{date_str}_{time_str}_{ms_str}"
            return datetime.strptime(dt_str, "%Y%m%d_%H%M%S_%f")
    except:
        pass
    
    # Fallback to file modification time
    return datetime.fromtimestamp(filename.stat().st_mtime)


def migrate_json_files(data_dir="data/gestures", dry_run=False):
    """
    Migrate all JSON training files to database
    
    Args:
        data_dir: Directory containing gesture folders
        dry_run: If True, only print what would be done
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Initialize database
    print("Initializing database...")
    engine = init_db()
    
    # Collect all JSON files
    json_files = list(data_path.glob("*/*.json"))
    print(f"\nFound {len(json_files)} JSON files")
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No data will be written to database\n")
    
    # Group by gesture
    gesture_counts = {}
    imported_count = 0
    skipped_count = 0
    error_count = 0
    
    session = get_session(engine)
    
    try:
        for json_file in json_files:
            try:
                # Get gesture from parent directory name
                gesture = json_file.parent.name
                
                # Load JSON data
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Validate required fields
                if 'landmarks' not in data:
                    print(f"⚠ Skipping {json_file.name}: no landmarks")
                    skipped_count += 1
                    continue
                
                landmarks = data['landmarks']
                if not isinstance(landmarks, list) or len(landmarks) != 63:
                    print(f"⚠ Skipping {json_file.name}: invalid landmarks (expected 63 values, got {len(landmarks)})")
                    skipped_count += 1
                    continue
                
                # Parse timestamp
                collection_date = parse_timestamp(json_file)
                
                # Create training sample
                sample = TrainingSample(
                    gesture=gesture,
                    landmarks=landmarks,
                    source='original',
                    collection_date=collection_date,
                    metadata={
                        'original_filename': json_file.name,
                        'original_path': str(json_file.relative_to(Path.cwd()))
                    }
                )
                
                if not dry_run:
                    session.add(sample)
                
                # Track counts
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                imported_count += 1
                
                # Commit in batches
                if not dry_run and imported_count % 100 == 0:
                    session.commit()
                    print(f"  Imported {imported_count} samples...")
            
            except Exception as e:
                print(f"✗ Error processing {json_file.name}: {e}")
                error_count += 1
                continue
        
        # Final commit
        if not dry_run:
            session.commit()
            print(f"\n✓ Successfully imported {imported_count} samples")
        else:
            print(f"\n[DRY RUN] Would import {imported_count} samples")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"Total files processed: {len(json_files)}")
        print(f"Successfully imported: {imported_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        print("\nSamples per gesture:")
        for gesture in sorted(gesture_counts.keys()):
            print(f"  {gesture}: {gesture_counts[gesture]}")
        print("=" * 60)
        
    finally:
        session.close()


def verify_migration():
    """Verify the migration by querying the database"""
    print("\n" + "=" * 60)
    print("Verifying Migration")
    print("=" * 60)
    
    engine = init_db()
    session = get_session(engine)
    
    try:
        # Count total samples
        total = session.query(TrainingSample).count()
        print(f"Total training samples in database: {total}")
        
        # Count by gesture
        from sqlalchemy import func
        gesture_counts = session.query(
            TrainingSample.gesture,
            func.count(TrainingSample.id)
        ).group_by(TrainingSample.gesture).all()
        
        print("\nSamples per gesture:")
        for gesture, count in sorted(gesture_counts):
            print(f"  {gesture}: {count}")
        
        # Count by source
        source_counts = session.query(
            TrainingSample.source,
            func.count(TrainingSample.id)
        ).group_by(TrainingSample.source).all()
        
        print("\nSamples by source:")
        for source, count in source_counts:
            print(f"  {source}: {count}")
        
    finally:
        session.close()
    
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate JSON training data to database')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/gestures',
        help='Directory containing gesture folders (default: data/gestures)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview migration without writing to database'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing migration (no import)'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_migration()
    else:
        print("=" * 60)
        print("Training Data Migration to Database")
        print("=" * 60)
        print(f"Data directory: {args.data_dir}")
        print(f"Dry run: {args.dry_run}")
        print("=" * 60)
        
        if not args.dry_run:
            response = input("\nThis will import training data into the database. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                return
        
        migrate_json_files(args.data_dir, args.dry_run)
        
        if not args.dry_run:
            verify_migration()


if __name__ == "__main__":
    main()
