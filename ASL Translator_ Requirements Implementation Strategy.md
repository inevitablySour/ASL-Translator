# ASL Translator: Requirements Implementation Strategy
## Project Context
Building a production-grade ML system for ASL gesture recognition that meets MA03 project requirements. The system will translate American Sign Language hand gestures to text using computer vision and machine learning, deployed as a modular service-based architecture.

## Mandatory Requirements (R-01 to R-06)
### R-01: Service-Based Architecture
**Requirement:** All components must be independent, standalone services that communicate over a network.
**Implementation Strategy:**
Decompose the monolithic application into 4-5 independent microservices:
1. **Frontend Service (Web UI)**
    * Serve static HTML/CSS/JS via nginx or simple HTTP server
    * Handle webcam access and image capture
    * Make HTTP requests to API Gateway
    * No direct access to other services
2. **API Gateway Service**
    * FastAPI service acting as entry point
    * Route requests to appropriate backend services
    * Handle request validation and authentication
    * Aggregate responses if needed
    * Expose endpoints: `/predict`, `/health`, `/languages`
3. **Model Inference Service**
    * FastAPI service dedicated to ML predictions
    * Load and serve the gesture classification model
    * Process hand landmarks and return predictions
    * Stateless design for horizontal scaling
    * Internal API not exposed externally
4. **Database Service**
    * PostgreSQL container
    * Store prediction history, user feedback, model metrics
    * Accessed only by API Gateway and Monitoring services
5. **MLflow Service** 
    * Model registry and experiment tracking
    * Version control for models
**Technology Choices:**
* FastAPI for Python-based services (consistent with existing code)
* nginx for frontend serving (lightweight, production-ready)
* REST APIs for inter-service communication (simple, well-understood)
* Docker networks for service isolation and communication
**Acceptance Criteria:**
* Each service has its own directory with dedicated Dockerfile
* Services communicate via HTTP/REST over Docker network
* Services can be deployed independently
* Stopping one non-critical service doesn't crash the system
* docker-compose orchestrates all services
### R-02: Docker Containerization
**Requirement:** All components run in Docker containers with well-defined APIs.
**Implementation Strategy:**
1. **Create Service-Specific Dockerfiles**
    * `services/frontend/Dockerfile`: nginx-based image for static files
    * `services/api_gateway/Dockerfile`: Python 3.12 with FastAPI
    * `services/model_service/Dockerfile`: Python 3.12 with ML dependencies (OpenCV, MediaPipe, scikit-learn)
    * Use multi-stage builds for smaller image sizes
    * Pin all dependency versions for reproducibility
2. **Docker Compose Configuration**
    * Define all services in `docker-compose.yml`
    * Create custom bridge network for inter-service communication
    * Configure health checks for each service
    * Set resource limits (CPU, memory)
    * Define restart policies
    * Use environment variables for configuration
3. **API Contracts**
    * OpenAPI/Swagger specs for each service
    * Document request/response formats
    * Version APIs (e.g., `/api/v1/predict`)
**Technology Choices:**
* Docker Compose v3.8+ for orchestration
* Alpine-based images where possible (smaller footprint)
* REST APIs with JSON payloads (widely supported)
* Environment variables for configuration (12-factor app principle)
**Acceptance Criteria:**
* All services run in separate containers
* System starts with single command: `docker-compose up`
* Services can reach each other via container names
* Health checks verify service availability
* No hardcoded connection strings (use environment variables)
### R-03: Database Implementation
**Requirement:** Store and retrieve data (user inputs, model outputs, user feedback).
**Implementation Strategy:**
1. **Database Choice: PostgreSQL**
    * Robust, open-source relational database
    * Strong data integrity with ACID compliance
    * Good performance for structured data
    * Excellent Docker support
2. **Schema Design**
    * `predictions` table: Store each prediction with timestamp, input image reference, gesture detected, confidence score, language, processing time
    * `user_feedback` table: Store user corrections/ratings on predictions (for model improvement)
    * `model_versions` table: Track deployed model versions with metadata
    * `system_metrics` table: Store aggregated metrics over time
3. **Database Service**
    * Use official PostgreSQL Docker image
    * Create initialization scripts in `services/database/init/`
    * Set up indexes for common queries
    * Configure persistent volume for data retention
4. **Database Access Layer**
    * Use SQLAlchemy ORM in API Gateway service
    * Create database models matching schema
    * Implement repository pattern for data access
    * Connection pooling for performance
**Technology Choices:**
* PostgreSQL 15+ (mature, well-documented)
* SQLAlchemy 2.0+ (Python ORM)
* Alembic for database migrations (future extensibility)
**Acceptance Criteria:**
* PostgreSQL container runs and is accessible
* Database schema is created automatically on first run
* Each prediction is logged to database
* API can query prediction history
* User feedback can be submitted and stored
* Database persists data across container restarts
### R-04: Model Serving Endpoint
**Requirement:** ML model hosted as a service that receives requests and returns predictions.
**Implementation Strategy:**
1. **Model Service Design**
    * Dedicated FastAPI service at `services/model_service/`
    * Load model on startup (from file or MLflow registry)
    * Expose internal endpoint: `POST /infer`
    * Accept hand landmark features as input
    * Return gesture label and confidence score
2. **Model Loading**
    * Load trained model from `models/` directory or MLflow registry
    * Support multiple model types (scikit-learn, PyTorch, TensorFlow)
    * Cache model in memory for fast inference
    * Implement model versioning support
3. **Preprocessing Pipeline**
    * MediaPipe hand detection runs in model service
    * Feature extraction from hand landmarks
    * Normalization/scaling applied
    * Error handling for invalid inputs
4. **Response Format**
```json
{
  "gesture": "A",
  "confidence": 0.94,
  "processing_time_ms": 45
}
```
**Technology Choices:**
* FastAPI for model serving (async support, automatic docs)
* Use existing MediaPipe + scikit-learn model
* Option to swap in TensorFlow/PyTorch models later
* Pydantic models for request/response validation
**Acceptance Criteria:**
* Model service starts and loads model successfully
* `POST /infer` endpoint accepts image or features
* Endpoint returns prediction with confidence score
* Invalid inputs return clear error messages
* Average inference time < 200ms
* Service is accessible only from API Gateway (not publicly exposed)
### R-05: User Interface
**Requirement:** End-user can access system to provide input and receive predictions from a typical user machine.
**Implementation Strategy:**
1. **Frontend Service**
    * Serve existing web UI (HTML/CSS/JS) via nginx
    * Lightweight container, separate from backend
    * Static file serving only
2. **UI Functionality**
    * Webcam access for real-time video capture
    * "Capture & Translate" button to send frame to backend
    * Language selection dropdown (English/Dutch)
    * Display prediction result with confidence score
    * Show prediction history
    * Feedback mechanism (thumbs up/down on predictions)
3. **Communication Flow**
    * UI → API Gateway → Model Service → Database
    * Use fetch API for HTTP requests
    * Handle loading states and errors gracefully
    * Display user-friendly error messages
4. **Responsive Design**
    * Mobile-friendly layout
    * Works on desktop and tablet browsers
    * Progressive enhancement (works without JavaScript for basic features)
**Technology Choices:**
* nginx for static file serving (production-grade)
* Vanilla JavaScript (no framework overhead, use existing code)
* HTML5 WebRTC for webcam access
* CSS Grid/Flexbox for responsive layout
**Acceptance Criteria:**
* UI is accessible from browser at `http://localhost:8080`
* Webcam permission request works on Chrome, Firefox, Edge
* User can capture image and receive prediction
* Prediction displays gesture, translation, and confidence
* Language can be switched between English and Dutch
* UI works on mobile browser (responsive)
* User feedback can be submitted
### R-06: Basic Security Measures
**Requirement:** Protect against common vulnerabilities and handle sensitive information appropriately.
**Implementation Strategy:**
1. **API Security**
    * API Key authentication for service-to-service communication
    * Rate limiting on public endpoints (e.g., 100 requests/minute per IP)
    * Input validation using Pydantic models
    * CORS policy restricting allowed origins
    * HTTPS support (TLS certificates)
2. **Input Validation & Sanitization**
    * Validate image format and size (max 5MB)
    * Sanitize all user inputs
    * Prevent SQL injection via parameterized queries (SQLAlchemy handles this)
    * Validate file uploads (only accept images)
3. **Secrets Management**
    * Store database credentials in environment variables
    * Use Docker secrets for sensitive data
    * Never commit secrets to Git (`.gitignore` configuration)
    * Separate `.env` files for dev/prod
4. **Security Headers**
    * Add security headers via middleware:
        * `X-Content-Type-Options: nosniff`
        * `X-Frame-Options: DENY`
        * `Content-Security-Policy`
        * `Strict-Transport-Security` (HTTPS only)
5. **Database Security**
    * Database user with minimal required privileges
    * Database not exposed on host network (internal only)
    * Regular backups
6. **Container Security**
    * Run containers as non-root user
    * Read-only root filesystem where possible
    * Minimal base images (reduce attack surface)
    * Regular security updates for base images
**Technology Choices:**
* `slowapi` for rate limiting in FastAPI
* Pydantic for input validation
* Docker secrets for credential management
* Let's Encrypt for SSL certificates (production)
**Acceptance Criteria:**
* API Gateway requires authentication for prediction endpoint
* Rate limiting prevents abuse (returns 429 after threshold)
* Invalid inputs are rejected with clear error messages
* Database credentials are not hardcoded
* Security headers present in HTTP responses
* Model service is not accessible from outside Docker network
* System passes basic OWASP Top 10 vulnerability scan
## Optional Requirements (OR-01, OR-02, OR-03)
### OR-01: CI/CT/CD Pipeline
**Requirement:** Automate building, testing, and deployment of services.
**Implementation Strategy:**
1. **Platform Choice: GitHub Actions**
    * Free for public repositories
    * Native GitHub integration
    * Good Docker support
    * Extensive marketplace of actions
2. **Pipeline Stages**
   **Continuous Integration:**
* Trigger on push/pull request to main branch
* Checkout code
* Set up Python environment
* Install dependencies
* Run linting (black, flake8, mypy)
* Run unit tests with pytest
* Generate test coverage report
* Build Docker images
* Push images to registry (Docker Hub or GHCR)
   **Continuous Testing:**
* Integration tests between services
* API contract tests
* Load testing (optional, simple baseline)
* Security scanning (Trivy for container vulnerabilities)
   **Continuous Deployment:**
* Deploy to staging environment (Docker Compose on VM)
* Run smoke tests
* Manual approval gate for production
* Deploy to production environment
* Rollback capability
3. **Workflow Files**
    * `.github/workflows/ci.yml`: Run on every push
    * `.github/workflows/cd.yml`: Run on tag/release
    * `.github/workflows/test.yml`: Integration tests
4. **Testing Strategy**
    * Unit tests for each service component
    * Integration tests using pytest and requests
    * API contract tests
    * Minimum 70% code coverage requirement
**Technology Choices:**
* GitHub Actions (free, integrated)
* pytest for testing framework
* Docker Hub or GHCR for image registry
* pytest-cov for coverage reporting
* black, flake8 for code quality
**Acceptance Criteria:**
* Pipeline runs automatically on git push
* All tests must pass before merge
* Docker images built and tagged with commit SHA
* Images pushed to container registry
* Deployment triggered on tag creation
* Failed builds block merges to main
* Pipeline completes in < 10 minutes
### OR-02: Monitoring Dashboard
**Requirement:** Monitor model health and statistical properties of production data.
**Implementation Strategy:**
1. **Monitoring Stack**
    * **Prometheus**: Metrics collection and storage
    * **Grafana**: Visualization and dashboards
    * **Alert Manager**: Alert notifications (optional)
2. **Metrics to Track**
   **Model Performance Metrics:**
* Prediction count (total, per hour)
* Average confidence scores
* Distribution of predicted gestures
* Prediction latency (p50, p95, p99)
* Error rate (failed predictions)
* Model version in use
   **Data Quality Metrics:**
* Input image dimensions
* Hand detection success rate
* Confidence score distribution
* Low confidence predictions (< 0.6)
* Data drift detection (feature distribution changes)
   **System Metrics:**
* Request rate (requests per second)
* Response times per service
* CPU/Memory usage per container
* Database connection pool status
* Error rates and types
3. **Implementation**
    * Add Prometheus service to docker-compose
    * Instrument code with prometheus-client metrics
    * Configure Prometheus to scrape all services
    * Add Grafana service to docker-compose
    * Create custom Grafana dashboards
    * Set up data source connection in Grafana
4. **Dashboard Panels**
    * Real-time prediction rate
    * Model confidence distribution histogram
    * Prediction latency over time
    * Gesture distribution pie chart
    * Error rate graph
    * System resource utilization
    * Data quality alerts panel
5. **Alerting Rules** (optional but recommended)
    * Alert if error rate > 5%
    * Alert if average confidence < 0.7 for extended period
    * Alert if prediction latency > 500ms
    * Alert if service health check fails
**Technology Choices:**
* Prometheus 2.x (industry standard)
* Grafana 10.x (powerful visualization)
* prometheus-client Python library
* AlertManager for notifications (optional)
**Acceptance Criteria:**
* Prometheus container collects metrics from all services
* Grafana dashboard accessible at `http://localhost:3000`
* Dashboard displays real-time prediction metrics
* Model performance metrics visible (confidence, latency)
* Data distribution charts update in real-time
* Historical data retained for 30 days minimum
* Dashboard shows model health status
### OR-03: Data & Model Version Control
**Requirement:** Version all training datasets and model artifacts for reproducibility and traceability.
**Implementation Strategy:**
1. **Tool Choice: DVC (Data Version Control)**
    * Git-like versioning for data and models
    * Lightweight (tracks metadata in Git, stores data separately)
    * Integrates well with MLflow
    * Supports multiple storage backends (S3, local, GCS)
2. **Version Control Structure**
   **Dataset Versioning:**
* Track raw training images in `data/raw/`
* Track processed features in `data/processed/`
* Track train/val/test splits
* Each dataset version tagged (e.g., `v1.0`, `v1.1`)
* Metadata includes: collection date, size, class distribution
   **Model Versioning:**
* Track trained model artifacts (`model.pkl`, `scaler.pkl`)
* Track model configuration files
* Track training scripts
* Each model version linked to dataset version
* Metadata includes: accuracy, training date, hyperparameters
3. **Implementation Steps**
    * Initialize DVC in repository: `dvc init`
    * Configure remote storage (local or cloud)
    * Track data directories: `dvc add data/`
    * Track model directories: `dvc add models/`
    * Commit `.dvc` files to Git
    * Create pipelines with `dvc.yaml` for reproducibility
4. **Integration with MLflow**
    * MLflow tracks experiments and model registry
    * DVC tracks underlying data and artifacts
    * Combined approach: DVC for data, MLflow for model lifecycle
    * Link model versions between systems via tags/metadata
5. **Reproducibility Workflow**
```warp-runnable-command
git checkout v1.0
dvc pull
python train_model.py
# Reproduces exact model from dataset v1.0
```
6. **Storage Backend**
    * Development: Local storage in `.dvc/cache`
    * Production: AWS S3, Google Cloud Storage, or Azure Blob
    * For project: Use local storage or shared network drive
**Technology Choices:**
* DVC 3.x (mature, well-documented)
* Git for code and D
