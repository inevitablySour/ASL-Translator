#!/usr/bin/env bash

# Wrapper script for docker-compose down that exports PostgreSQL to SQLite first

echo "Exporting PostgreSQL data to SQLite before shutdown..."

# Check if PostgreSQL container is running
if docker ps | grep -q "asl_postgres"; then
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
    
    if $PYTHON_CMD scripts/export_to_sqlite.py; then
        echo "Data exported successfully"
    else
        echo "Export failed, but continuing with shutdown..."
        echo "   Make sure dependencies are installed: pip install sqlalchemy psycopg2-binary"
    fi
else
    echo "PostgreSQL container not running, skipping export"
fi

echo ""
echo "Stopping Docker containers..."
docker-compose down "$@"

echo ""
echo "Done!"
