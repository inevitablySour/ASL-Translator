#!/bin/bash
set -e

echo "Starting ASL Translator API..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
until python -c "from sqlalchemy import create_engine; import os; engine = create_engine(os.getenv('DATABASE_URL', 'postgresql://asl_user:asl_password@postgres:5432/asl_translator')); engine.connect()" 2>/dev/null; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
done

echo "PostgreSQL is up!"

# Initialize database and auto-migrate if needed
echo "Initializing database..."
python init_db.py

# Start the API
echo "Starting API server..."
exec python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
