#!/usr/bin/env bash

# Wrapper script for docker-compose down that exports PostgreSQL to SQLite first

echo "Exporting PostgreSQL data to SQLite before shutdown..."

# Check if PostgreSQL container is running
if docker ps | grep -q "asl_postgres"; then
    # Activate virtual environment and run export
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        if python export_to_sqlite.py; then
            echo "Data exported successfully"
        else
            echo "Export failed, but continuing with shutdown..."
        fi
        deactivate
    else
        # Try without virtual environment
        if python3 export_to_sqlite.py 2>/dev/null; then
            echo "✓ Data exported successfully"
        else
            echo "Could not export data (missing dependencies or venv)"
            echo "   Install: pip install sqlalchemy psycopg2-binary"
        fi
    fi
else
    echo "ℹ️  PostgreSQL container not running, skipping export"
fi

echo ""
echo "Stopping Docker containers..."
docker-compose down "$@"

echo ""
echo "Done!"
