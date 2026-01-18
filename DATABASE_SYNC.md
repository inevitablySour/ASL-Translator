# Database Sync Setup

This project includes automatic SQLite to PostgreSQL synchronization when pushing to GitHub.

## Local Setup (Git Hook)

A pre-push hook automatically syncs `data/feedback.db` to PostgreSQL before each push.

### How it works:
1. When you run `git push`, the hook checks if `feedback.db` has changes
2. If changes are detected and PostgreSQL is running, it runs `migrate_to_postgres.py`
3. The push continues after successful sync

### Prerequisites:
- PostgreSQL container must be running: `docker-compose up -d postgres`
- Python 3 with sqlalchemy and psycopg2 installed

### Manual sync:
```bash
python3 migrate_to_postgres.py
```

## GitHub Actions (Remote)

The `.github/workflows/sync-database.yml` workflow automatically syncs the database when you push changes to `data/feedback.db`.

### Setup:
1. Add your PostgreSQL connection string as a GitHub secret:
   - Go to: Repository Settings → Secrets and variables → Actions
   - Create secret: `POSTGRES_DATABASE_URL`
   - Format: `postgresql://user:password@host:port/database`

2. Your PostgreSQL instance must be accessible from GitHub Actions:
   - Use a cloud PostgreSQL service (AWS RDS, DigitalOcean, etc.)
   - Or use a self-hosted runner with access to your local network

### Trigger:
- Automatically runs on push to `master`/`main` when `data/feedback.db` changes
- View workflow runs in the "Actions" tab on GitHub

## Workflow

```
Local Change → Commit → Push → Pre-push Hook (local sync) → GitHub Actions (remote sync)
```

## Notes

- The SQLite file `data/feedback.db` is tracked in git (see `.gitignore`)
- Both local and remote syncs run independently
- If local PostgreSQL isn't running, the hook will warn but allow the push
- GitHub Actions requires the secret `POSTGRES_DATABASE_URL` to be configured
