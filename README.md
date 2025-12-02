# 🤟 ASL Translator

An AI-powered application that uses computer vision to recognize American Sign Language (ASL) hand gestures and translates them to English or Dutch text in real-time.

## Features

- **Real-time Hand Detection**: Uses MediaPipe Hands for accurate hand landmark detection
- **Gesture Classification**: ML-based classification of ASL alphabet gestures
- **Multi-language Support**: Translates to English or Dutch
- **AI Ops Integration**: MLflow for experiment tracking and model versioning
- **REST API**: FastAPI-based backend with comprehensive endpoints
- **Web Interface**: Interactive web UI with webcam support
- **Monitoring**: Prometheus metrics for production monitoring
- **Containerized**: Docker support for easy deployment

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Webcam UI)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│   (REST API)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│MediaPipe│ │Classifier│
│  Hands  │ │  Model   │
└────────┘ └──────────┘
         │
         ▼
    ┌──────────┐
    │Translator│
    └──────────┘
         │
         ▼
    ┌──────────┐
    │  MLflow  │
    │(Tracking)│
    └──────────┘
```

## Technology Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: NumPy (extendable to TensorFlow/PyTorch)
- **AI Ops**: MLflow, Prometheus
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Docker, Docker Compose

## Prerequisites

- Python 3.10 or higher
- Webcam (for real-time gesture recognition)
- Git (for version control)
- Docker (optional, for containerized deployment)

## Installation

### Option 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ASL-Translator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your preferred settings
   ```

5. **Run the application**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

6. **Access the application**
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Option 2: Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Web UI: http://localhost:8000
   - MLflow UI: http://localhost:5000

3. **Stop the application**
   ```bash
   docker-compose down
   ```

## Usage

### Web Interface

1. Open http://localhost:8000 in your web browser
2. Click "Start Camera" to enable your webcam
3. Position your hand in front of the camera
4. Form an ASL letter gesture
5. Click "Capture & Translate" to recognize the gesture
6. View the translation in your selected language

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Model Information
```bash
GET /info
```

#### Predict Gesture (Base64)
```bash
POST /predict
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "language": "english"
}
```

#### Supported Languages
```bash
GET /languages
```

#### Prometheus Metrics
```bash
GET /metrics
```

## Project Structure

```
ASL-Translator/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── hand_detector.py        # MediaPipe hand detection
│   ├── gesture_classifier.py   # Gesture classification
│   ├── translator.py           # Translation service
│   └── mlflow_manager.py       # MLflow integration
├── static/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── app.js
│   └── index.html
├── models/                     # Trained models (gitignored)
├── data/                       # Training data (gitignored)
├── tests/                      # Unit tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## AI Ops Features

### MLflow Integration

The application integrates MLflow for:
- Experiment tracking
- Model versioning
- Performance monitoring
- Model registry

Start MLflow UI:
```bash
mlflow ui --port 5000
```

Access at http://localhost:5000

### Prometheus Metrics

Available metrics:
- `asl_predictions_total`: Total number of predictions
- `asl_prediction_latency_seconds`: Prediction latency histogram

Access metrics at http://localhost:8000/metrics

## Development

### Adding a New Gesture

1. Collect training data for the gesture
2. Train/update the classifier model
3. Update `gesture_classifier.py` with the new gesture
4. Add translation mappings in `translator.py`

### Training a Custom Model

The current implementation uses a heuristic-based classifier. To train a custom ML model:

1. **Collect Data**: Use the webcam to capture hand gestures
2. **Preprocess**: Extract MediaPipe landmarks
3. **Train Model**: Use scikit-learn, TensorFlow, or PyTorch
4. **Save Model**: Store in `models/` directory
5. **Update Classifier**: Implement model loading in `gesture_classifier.py`

Example:
```python
# In gesture_classifier.py
def __init__(self):
    self.model = joblib.load('models/asl_classifier.pkl')
```

### Running Tests

```bash
pytest tests/
```

## Configuration

Key configuration options in `.env`:

```env
# Application
APP_NAME=ASL Translator
APP_VERSION=1.0.0
DEBUG=False

# Server
HOST=0.0.0.0
PORT=8000

# Model
MODEL_CONFIDENCE_THRESHOLD=0.6
MEDIAPIPE_MIN_DETECTION_CONFIDENCE=0.5

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=asl-gesture-recognition

# Language
DEFAULT_LANGUAGE=english
```

## Troubleshooting

### Camera not working
- Ensure browser has camera permissions
- Check if another application is using the camera
- Try refreshing the page

### Import errors
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Low prediction accuracy
- Ensure good lighting conditions
- Position hand clearly in frame
- Consider training a custom model with more data

## Future Enhancements

- [ ] Train a production-ready ML model
- [ ] Add support for ASL phrases and sentences
- [ ] Implement continuous gesture recognition (no capture button)
- [ ] Add support for more languages
- [ ] Mobile app development
- [ ] Dataset collection tool
- [ ] Model retraining pipeline
- [ ] A/B testing framework

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **MediaPipe**: Hand landmark detection
- **FastAPI**: Modern Python web framework
- **MLflow**: ML lifecycle management
- **OpenCV**: Computer vision library

## Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ for accessible communication
