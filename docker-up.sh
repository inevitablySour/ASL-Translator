#!/usr/bin/env bash

# Wrapper script for docker-compose up that syncs SQLite to PostgreSQL if needed

echo "Starting Docker containers..."
docker-compose up -d postgres

echo ""
echo "Waiting for PostgreSQL to be ready..."
sleep 5

echo ""
echo "🔍 Checking if database sync is needed..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
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

# Deactivate venv if it was activated
if [ -d ".venv" ]; then
    deactivate
fi

echo ""
echo "Starting remaining services..."
docker-compose up -d "$@"

echo ""
echo "All services are running!"
echo ""
echo "Services:"
docker-compose ps
