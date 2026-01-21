# Quick Start - Database Sync

## Daily Workflow

### Start working:
```bash
./docker-up.sh
```
This automatically syncs SQLite → PostgreSQL if needed.

### Stop working:
```bash
./docker-down.sh
```
This automatically syncs PostgreSQL → SQLite before shutdown.

### Push to GitHub:
```bash
git add data/feedback.db
git commit -m "Update database"
git push
```
The pre-push hook syncs to PostgreSQL (if running) before pushing.

## Commands Summary

| Command | What it does |
|---------|-------------|
| `./docker-up.sh` | Start containers + smart sync from SQLite |
| `./docker-down.sh` | Export to SQLite + stop containers |
| `python3 scripts/migrate_to_postgres.py` | Manual SQLite → PostgreSQL |
| `python3 scripts/export_to_sqlite.py` | Manual PostgreSQL → SQLite |
| `python3 scripts/check_and_sync.py` | Check if sync needed |

## What Gets Synced?

- Users
- Models  
- Predictions
- Feedback
- Training Samples
- Training Runs

## Notes

- SQLite file `data/feedback.db` is tracked in git
- PostgreSQL data is in Docker volumes (not tracked)
- Always use wrapper scripts (`docker-up.sh` / `docker-down.sh`) to maintain data consistency
