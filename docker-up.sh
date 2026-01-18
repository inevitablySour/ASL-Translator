#!/usr/bin/env bash

# Wrapper script for docker-compose up that syncs SQLite to PostgreSQL if needed

echo "Starting Docker containers..."
docker-compose up -d postgres

echo ""
echo "Waiting for PostgreSQL to be ready..."
sleep 5

echo ""
echo "🔍 Checking if database sync is needed..."

# Detect Python command - use active venv or find one
if [ -n "$VIRTUAL_ENV" ]; then
    # Already in a virtual environment
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    # Local .venv directory
    PYTHON_CMD=".venv/bin/python"
elif command -v python &> /dev/null && python -c "import sqlalchemy, psycopg2" 2>/dev/null; then
    # System python with required packages
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null && python3 -c "import sqlalchemy, psycopg2" 2>/dev/null; then
    # System python3 with required packages
    PYTHON_CMD="python3"
else
    echo "⚠️  Cannot find Python with sqlalchemy and psycopg2 installed"
    echo "   Install with: pip install sqlalchemy psycopg2-binary"
    PYTHON_CMD="python3"
fi

# Check if sync is needed
if $PYTHON_CMD check_and_sync.py; then
    echo ""
    echo "Syncing SQLite to PostgreSQL..."
    if $PYTHON_CMD migrate_to_postgres.py; then
        echo "Sync completed successfully"
    else
        echo "Sync failed, but containers are running"
    fi
else
    echo "Database is already up-to-date"
fi

echo ""
echo "Starting remaining services..."
docker-compose up -d "$@"

echo ""
echo "All services are running!"
echo ""
echo "Services:"
docker-compose ps
