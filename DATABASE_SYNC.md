# Database Sync Setup

This project includes automatic SQLite to PostgreSQL synchronization when pushing to GitHub.

## Starting Docker Services

Use the wrapper script to start services with intelligent sync:

```bash
./docker-up.sh
```

This script:
1. Starts PostgreSQL container
2. Checks if SQLite has newer/more data than PostgreSQL
3. Syncs only if SQLite has updates
4. Starts remaining services

### How it decides:
- Compares record counts between SQLite and PostgreSQL
- Checks timestamps to find newer data
- Only syncs when SQLite has changes
- Skips sync if databases are already in sync

### Manual sync:
```bash
python3 migrate_to_postgres.py
```

## Local Setup (Git Hook)

A pre-push hook automatically syncs `data/feedback.db` to PostgreSQL before each push.

### How it works:
1. When you run `git push`, the hook checks if `feedback.db` has changes
2. If changes are detected and PostgreSQL is running, it runs `migrate_to_postgres.py`
3. The push continues after successful sync

## Exporting from PostgreSQL to SQLite

When you shut down Docker containers, PostgreSQL data is automatically exported back to SQLite.

### Using the wrapper script:
```bash
./docker-down.sh
```

This script:
1. Exports all PostgreSQL data to `data/feedback.db`
2. Runs `docker-compose down`
3. Your SQLite file is now up-to-date and ready to commit

### Manual export:
```bash
python3 export_to_sqlite.py
```

## Workflow

```
Docker Up (./docker-up.sh) → Work in PostgreSQL → Docker Down (./docker-down.sh) → Commit → Push
```

## Notes

- The SQLite file `data/feedback.db` is tracked in git (see `.gitignore`)
- PostgreSQL data lives in Docker volumes (not tracked in git)
- Always use `./docker-up.sh` and `./docker-down.sh` to maintain data consistency
- The pre-push hook syncs locally if PostgreSQL is running
