# ASL Translator

Real-time American Sign Language (ASL) recognition system with continuous learning capabilities. The system uses computer vision and machine learning to translate ASL hand gestures into text, and includes an automated feedback loop for model improvement.

## Features

- **Real-time ASL Recognition**: Detects and classifies ASL hand gestures (A-H) using a trained Random Forest model
- **Web Interface**: Clean, responsive UI with live camera feed and predictions
- **Continuous Learning**: Collects high-confidence predictions for automatic model retraining
- **Microservices Architecture**: Scalable design with separate API, inference, and training services
- **Model Monitoring**: Comprehensive dashboard for tracking predictions, model performance, and training history
- **Database**: PostgreSQL for production with automatic SQLite backup migration
- **Message Broker**: RabbitMQ for asynchronous inference processing

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Web UI     │────▶│  FastAPI     │────▶│  RabbitMQ   │
│  (Browser)  │◀────│  (API)       │◀────│  (Broker)   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                      │
                           ▼                      ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  PostgreSQL  │     │  Inference  │
                    │  (Database)  │     │  Service    │
                    └──────────────┘     └─────────────┘
                           ▲                      
                           │                      
                    ┌──────────────┐             
                    │  Training    │             
                    │  Service     │             
                    └──────────────┘             
```

### Components

1. **API Service** (`services/api/`)
   - FastAPI web server
   - Serves web UI and handles prediction requests
   - Manages feedback collection and database operations
   - Auto-migrates SQLite data to PostgreSQL on first run

2. **Inference Service** (`services/inference/`)
   - Processes prediction requests from RabbitMQ
   - Uses MediaPipe for hand detection and landmark extraction
   - Loads trained Random Forest classifier for gesture recognition

3. **Training Service** (`services/training/`)
   - Monitors database for new feedback samples
   - Automatically retrains model when 200 samples collected
   - Uses MLflow for experiment tracking

4. **PostgreSQL Database**
   - Stores predictions, feedback, training samples, and model metadata
   - Persistent storage via Docker volume

5. **RabbitMQ Message Broker**
   - Asynchronous communication between API and inference services
   - Ensures scalability and reliability

## Setup & Installation

### Prerequisites

- Docker & Docker Compose
- Git
- 4GB+ RAM recommended

**macOS Users:**
- Docker Desktop for Mac (with sufficient resources allocated)
- Allow camera access for Docker Desktop
- Rosetta 2 for Apple Silicon Macs (for x86 images)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ASL-Translator
   ```

2. **Start all services**
   
   **macOS/Linux:**
   ```bash
   ./docker-up.sh
   # OR
   docker-compose up -d
   ```
   
   **Windows (PowerShell):**
   ```powershell
   # First, ensure Docker Desktop is running
   # Then install required packages for database sync:
   pip install sqlalchemy psycopg2-binary
   
   # Run the startup script
   .\docker-up.ps1
   # OR
   docker-compose up -d
   ```

3. **Access the application**
   - Web UI: http://localhost:8000
   - Dashboard: http://localhost:8000/dashboard
   - RabbitMQ Management: http://localhost:15672 (guest/guest)

4. **Check service status**
   
   **macOS/Linux:**
   ```bash
   docker-compose ps
   docker-compose logs -f api  # View API logs
   ```
   
   **Windows:**
   ```powershell
   docker-compose ps
   docker-compose logs -f api  # View API logs
   ```

**macOS Note:** On first run, macOS may ask for camera permissions. Grant access to your browser and Docker Desktop.

### Database Auto-Migration

On first run, the system automatically:
1. Creates PostgreSQL tables
2. Checks if database is empty
3. If empty, migrates data from `data/feedback.db` (SQLite backup)
4. This ensures anyone cloning the repo gets your existing data

**Current database contains:**
- 662 predictions
- 231 feedback entries
- 1,514 training samples (1,283 original + 231 feedback)
- 1 trained model (99.67% accuracy)

## Development

### Local Setup (without Docker)

1. **Create virtual environment**
   
   **macOS/Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   
   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   # OR if you get execution policy error:
   .venv\Scripts\activate.bat
   ```

2. **Install dependencies**
   ```bash
   pip install -r services/api/requirements.txt
   pip install -r services/inference/requirements.txt
   ```

3. **Collect training data**
   ```bash
   python scripts/collect_static_data.py
   ```
   - Press 'N' to select gesture name
   - Press SPACE to capture samples
   - Collect 50-100 samples per gesture

4. **Train a model**
   ```bash
   python scripts/train_model_cli.py
   ```

### Project Structure

```
ASL-Translator/
├── services/
│   ├── api/              # FastAPI web service
│   │   ├── src/
│   │   │   ├── api.py           # Main API endpoints
│   │   │   ├── database.py      # SQLAlchemy models
│   │   │   ├── feedback_manager.py
│   │   │   └── static/          # Web UI files
│   │   ├── init_db.py           # Auto-migration script
│   │   ├── entrypoint.sh        # Startup script
│   │   └── Dockerfile
│   ├── inference/        # Inference worker service
│   │   ├── src/
│   │   │   ├── inference.py
│   │   │   ├── inference_worker.py
│   │   │   ├── hand_detector.py
│   │   │   └── gesture_classifier.py
│   │   └── Dockerfile
│   └── training/         # Model training service
│       ├── retrain_worker.py
│       └── Dockerfile
├── data/
│   ├── feedback.db       # SQLite backup (tracked in git)
│   └── gestures/         # Training data (images + JSON)
├── models/
│   ├── asl_model_*/      # Trained model versions
│   └── mlruns/           # MLflow experiment tracking
├── src/                  # Shared training code
│   ├── model_trainer.py
│   ├── db_data_loader.py
│   └── hand_detector.py
├── scripts/              # CLI and maintenance scripts
│   ├── collect_static_data.py   # Data collection script
│   ├── train_model_cli.py       # Manual training script
│   ├── migrate_to_postgres.py   # One-time migration script
│   ├── export_to_sqlite.py      # PostgreSQL → SQLite export
│   ├── check_and_sync.py        # Decide if sync is needed
│   └── preprocess_dataset.py    # Image preprocessing for landmarks
└── docker-compose.yaml
```

## Usage

### Web Interface

1. **Start Camera**: Enable webcam access
2. **Select Model**: Choose trained model version
3. **Start Live Detection**: Begin real-time gesture recognition
4. **View Predictions**: See gesture, confidence, and processing time
5. **Provide Feedback**: After stopping, choose to contribute high-confidence samples

### Dashboard

Access the monitoring dashboard at `/dashboard` to view:
- Total predictions and average confidence
- Gesture distribution
- Model versions and accuracy
- Training history
- Feedback collection progress

### API Endpoints

- `POST /predict` - Submit image for gesture prediction
- `POST /feedback` - Submit user feedback
- `GET /api/models` - List available models
- `GET /api/stats` - Production statistics
- `GET /api/training-history` - Training run history
- `GET /dashboard` - Monitoring dashboard

## Model Training

### Automatic Retraining

The system automatically retrains when:
- 200 new feedback samples are collected
- Training service runs every 5 minutes checking for new data

### Manual Training

```bash
# Train from database
python scripts/train_model_cli.py

# View training metrics
mlflow ui --backend-store-uri models/mlruns
```

### Current Model

- **Type**: Random Forest Classifier
- **Accuracy**: 99.67%
- **Gestures**: A, B, C, D, E, F, G, H
- **Features**: 63 hand landmark coordinates (21 points × 3 dimensions)
- **Training Samples**: 1,502

## Database

### PostgreSQL (Production)

**Connection Details:**
- Host: localhost:5432
- Database: asl_translator
- User: asl_user
- Password: asl_password

**Tables:**
- `users` - User accounts (for future auth)
- `models` - Model versions and metadata
- `predictions` - All prediction logs
- `feedback` - User feedback on predictions
- `training_samples` - Training data (original + feedback)
- `training_runs` - Training history

### SQLite Backup

The `data/feedback.db` file serves as:
- Backup of production data
- Initial data for new deployments
- Development database

## Configuration

### Environment Variables

```bash
# API Service
DATABASE_URL=postgresql://asl_user:asl_password@postgres:5432/asl_translator
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672

# Training Service
REQUIRED_FEEDBACK_SAMPLES=200  # Threshold for auto-retraining
```

### Model Configuration

Edit `services/inference/src/config.py`:
```python
classifier_confidence_threshold = 0.6  # Min confidence for predictions
mediapipe_min_detection_confidence = 0.5  # Hand detection threshold
feedback_confidence_threshold = 0.9  # Min confidence to collect feedback
```

## Platform-Specific Notes

### macOS

**Camera Access:**
- System Preferences → Security & Privacy → Camera
- Grant access to your browser (Chrome, Safari, etc.)
- Docker Desktop needs to be running

**Performance:**
- Allocate at least 4GB RAM to Docker Desktop
- Preferences → Resources → Memory

**Apple Silicon (M1/M2):**
- Works via Rosetta 2 emulation
- Performance is excellent for this workload
- Some Python packages may show warnings but work fine

**File Sharing:**
- Ensure Docker has access to the project directory
- Preferences → Resources → File Sharing

### Linux

- Works natively without additional configuration
- Ensure Docker service is running: `systemctl start docker`
- Add user to docker group: `sudo usermod -aG docker $USER`

### Windows

**Prerequisites:**
- Docker Desktop for Windows (with WSL2 backend recommended)
- Python 3.10+ with virtual environment
- PowerShell 5.1 or higher

**Setup Steps:**

1. **Install Docker Desktop**
   - Download from https://www.docker.com/products/docker-desktop
   - Enable WSL2 integration (Settings → Resources → WSL Integration)
   - Ensure Docker Desktop is running before starting services

2. **Set up Python environment**
   ```powershell
   # Navigate to project directory
   cd C:\path\to\ASL-Translator
   
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   
   # Install database sync dependencies
   pip install sqlalchemy psycopg2-binary
   ```
   
   **Note:** If you get a PowerShell execution policy error when activating the venv, use:
   ```powershell
   .venv\Scripts\activate.bat
   ```

3. **Start the application**
   ```powershell
   # Using the PowerShell startup script (recommended)
   .\docker-up.ps1
   
   # OR manually with docker-compose
   docker-compose up -d
   ```

4. **Verify services are running**
   ```powershell
   docker-compose ps
   ```

**Camera Access:**
- Allow camera access in browser settings (Chrome: Settings → Privacy and security → Camera)
- Windows may prompt for camera permissions on first use

**Common Issues:**
- If Docker commands fail with "pipe/dockerDesktopLinuxEngine" error, ensure Docker Desktop is running
- If bash scripts don't work, use the PowerShell equivalents (`.ps1` files)
- For WSL issues, enable Docker integration: Docker Desktop → Settings → Resources → WSL Integration

## CI / CT / CD Pipeline

The project includes an end-to-end CI/CT/CD pipeline based on **GitHub Actions** and a Docker VM.

### Continuous Integration (CI)

- Defined in `.github/workflows/ci.yml`.
- Triggers on:
  - `push` to `master`.
  - `pull_request` targeting `master`.
- `build-and-test` job runs on `ubuntu-latest` and:
  - Checks out the repo (`actions/checkout@v4`).
  - Sets up Python 3.11 (`actions/setup-python@v5`).
  - Installs dependencies from `requirements.txt`.
  - Runs `python -m compileall .` to catch syntax/import issues.
  - If `tests/` exists, installs `pytest`, sets `PYTHONPATH` to the project root and runs `pytest tests/ -v`.
  - Validates Docker config with `docker compose -f docker-compose.yaml config`.
  - Builds all Docker services with `docker compose -f docker-compose.yaml build`.

### Continuous Testing (CT)

- Tests live under the `tests/` directory and are executed automatically by the CI workflow.
- Current coverage focuses on:
  - Core training logic and data loading in `src/` and `services/training/src/`.
  - Inference behavior in `services/inference/src/`.
  - API behavior in `services/api/src/` via FastAPI test client.
- Any new tests added under `tests/` will run on every push/PR to `master`.

### Continuous Deployment (CD)

- Implemented as a second job `deploy` in `.github/workflows/ci.yml`.
- `deploy`:
  - Has `needs: build-and-test`, so it only runs if CI/CT succeeded.
  - Runs only for the `master` branch (`if: github.ref == 'refs/heads/master'`).
  - Uses `appleboy/ssh-action@v1.1.0` to SSH into the Docker VM using GitHub Secrets:
    - `VM_HOST`, `VM_USER`, `VM_SSH_KEY`, `VM_SSH_PORT`.
  - On the VM it executes:
    - `cd /opt/asl-translator/ASL-Translator` (repository location).
    - `git pull` to fetch the latest code.
    - `docker compose down` to stop existing containers.
    - `docker compose pull || true` to pull any updated images (optional but harmless).
    - `docker compose up -d --build` to rebuild and restart `api`, `inference`, `training`, `postgres`, and `rabbitmq`.
- The VM itself is responsible for production concerns such as:
  - Running Docker Compose.
  - Terminating HTTPS and proxying to the API via Nginx (required for browser camera access).

For more background on the design decisions behind this pipeline, see `.github/cictcd.md`.

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs api
docker-compose logs inference

# Restart services
docker-compose restart
```

### Database migration issues
```bash
# Manual migration
python scripts/migrate_to_postgres.py

# Reset PostgreSQL (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Model not loading
```bash
# Check model symlinks
ls -la models/

# Create symlinks to latest model
cd models
ln -sf asl_model_*/asl_classifier.pkl asl_classifier.pkl
ln -sf asl_model_*/scaler.pkl scaler.pkl
ln -sf asl_model_*/label_encoder.pkl label_encoder.pkl
```

## Performance

- **Inference Time**: ~137ms average
- **Throughput**: ~7 predictions/second
- **Model Size**: ~2.1 MB
- **Accuracy**: 99.67% on validation set

## Future Enhancements
- [ ] Temporal gesture recognition

- [ ] Multi-language support
- [ ] Mobile app
- [ ] Cloud deployment

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- MediaPipe for hand detection
- FastAPI for the web framework
- MLflow for experiment tracking
- RabbitMQ for message brokering

## Resources

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RabbitMQ Best Practices](https://codemia.io/knowledge-hub/path/should_i_close_the_channelconnection_after_every_publish)
