# ASL Translator - User Feedback & Continuous Learning System

## Overview
This system collects user feedback on ASL translations to continuously improve the model. When predictions have high confidence (≥90%), users are prompted to share their hand landmark data (not photos) to help train better models.

## Key Features

### 1. **Privacy-First Data Collection**
- Only hand landmark coordinates (63 float values) are stored
- No photos or video frames are saved
- Clear privacy notice shown to users
- Users must explicitly consent to data sharing

### 2. **Automatic Retraining**
- Triggers when 200 feedback samples are collected
- Exports feedback to JSON format
- Trains new versioned model
- Tracks all model versions in database

### 3. **Model Versioning**
- Each model is saved with timestamp
- All versions tracked in database
- Metadata includes training date, accuracy, sample count
- Models can be activated/deactivated

## Architecture

### Components

1. **Frontend (JavaScript)**
   - `services/api/src/static/js/app.js` - Feedback UI logic
   - `services/api/src/static/index.html` - Feedback prompt UI
   - `services/api/src/static/css/feedback.css` - Feedback styles

2. **Backend (Python)**
   - `services/api/src/database.py` - SQLAlchemy models
   - `services/api/src/feedback_manager.py` - Feedback storage & export
   - `services/api/src/api.py` - REST API endpoints

3. **Inference**
   - `services/inference/src/inference_worker.py` - Returns landmarks for high-confidence predictions
   - `services/inference/src/inference.py` - Passes landmarks through message queue

4. **Training**
   - `services/training/retrain_worker.py` - Monitors DB and triggers retraining
   - `train_model_cli.py` - Model training script

### Database Schema

**Predictions Table**
- Stores all predictions with job_id, gesture, confidence, timestamps
- Includes landmarks (JSON) for high-confidence predictions

**Feedback Table**
- Links to predictions via job_id
- Tracks accepted/rejected status
- Flags if data has been used for training

**Models Table**
- Tracks all model versions
- Stores metadata, file paths, accuracy metrics
- Indicates which model is currently active

**TrainingRuns Table**
- Records each retraining session
- Links to resulting model
- Tracks sample counts and status

## Usage

### Starting the Feedback System

1. **Start the API service** (includes feedback endpoints):
   ```bash
   cd services/api
   uvicorn src.api:app --reload
   ```

2. **Start the retraining worker** (monitors for threshold):
   ```bash
   python services/training/retrain_worker.py --threshold 200 --interval 300
   ```

### API Endpoints

**POST /predict**
- Returns prediction with landmarks if confidence ≥ 0.9
- Automatically stores prediction in database

**POST /feedback**
- Submit user feedback
- Request body: `{"job_id": "...", "accepted": true/false}`
- Returns: `{"success": true, "message": "...", "feedback_id": "..."}`

**GET /feedback/stats**
- Get feedback statistics
- Returns counts, progress to threshold, etc.

### User Flow

1. User makes ASL gesture
2. System predicts with high confidence (≥90%)
3. Feedback prompt appears with privacy notice
4. User clicks "Yes, use my data" or "No thanks"
5. If accepted, landmarks stored in database
6. When 200 samples collected, retraining triggers automatically
7. New model version created and saved

## Configuration

Edit `services/inference/src/config.py`:

```python
# Minimum confidence to request feedback (0.0-1.0)
feedback_confidence_threshold: float = 0.9

# Number of feedback samples before retraining
retraining_sample_threshold: int = 200

# Database location
database_url: str = "sqlite:///data/feedback.db"
```

## Monitoring

### Check Feedback Stats
```bash
curl http://localhost:8000/feedback/stats
```

Returns:
```json
{
  "total_predictions": 1500,
  "total_feedback": 180,
  "unused_feedback": 180,
  "used_feedback": 0,
  "accepted_feedback": 170,
  "rejected_feedback": 10,
  "threshold": 200,
  "progress_to_threshold": "180/200"
}
```

### View Training Runs
Query the database:
```python
from services.api.src.database import get_session, TrainingRun

session = get_session()
runs = session.query(TrainingRun).order_by(TrainingRun.started_at.desc()).all()
for run in runs:
    print(f"{run.id}: {run.status} - {run.feedback_samples} samples")
```

### View Model Versions
```python
from services.api.src.database import get_session, Model

session = get_session()
models = session.query(Model).order_by(Model.created_at.desc()).all()
for model in models:
    print(f"{model.version}: active={model.is_active}, accuracy={model.accuracy}")
```

## Data Export Format

Feedback data is exported to: `data/gestures/feedback/export_{timestamp}/`

Structure:
```
export_20260118_132045/
├── A/
│   ├── A_feedback_20260118_132045_0000.json
│   ├── A_feedback_20260118_132045_0001.json
│   └── ...
├── B/
│   ├── B_feedback_20260118_132045_0000.json
│   └── ...
└── ...
```

JSON format:
```json
{
  "gesture": "A",
  "landmarks": [0.123, 0.456, 0.789, ...],  // 63 values (21 landmarks × 3 coords)
  "timestamp": "2026-01-18T13:20:45.123456",
  "confidence": 0.95,
  "feedback_id": "uuid-here",
  "source": "user_feedback"
}
```

## Manual Retraining

To manually trigger retraining without waiting for threshold:

```python
from services.api.src.feedback_manager import FeedbackManager

manager = FeedbackManager(retraining_threshold=200)

# Export feedback
result = manager.export_feedback_for_training()
print(f"Exported {result['count']} samples to {result['export_dir']}")

# Train manually
# python train_model_cli.py --model-name manual_retrain_20260118 --data-dir data/gestures

# Mark as used
manager.mark_feedback_as_used()
```

## Security Considerations

1. **No PII stored** - Only anonymous landmark coordinates
2. **Consent required** - Users must explicitly agree
3. **Database backups** - Regularly backup `data/feedback.db`
4. **Model review** - New models are not auto-activated; review performance first

## Future Enhancements

- [ ] Add user authentication for tracking individual contributors
- [ ] Implement model A/B testing
- [ ] Add confidence threshold adjustment based on feedback
- [ ] Support correcting wrong predictions (currently only accept/reject)
- [ ] Add web dashboard for monitoring feedback stats
- [ ] Implement scheduled retraining (daily/weekly) in addition to threshold-based

## Troubleshooting

**Feedback not being stored?**
- Check database initialization: `ls data/feedback.db`
- Check API logs for database errors
- Verify SQLAlchemy is installed: `pip list | grep -i sqlalchemy`

**Retraining not triggering?**
- Check retraining worker is running
- Verify threshold reached: `curl localhost:8000/feedback/stats`
- Check worker logs for errors

**Landmarks not appearing in predictions?**
- Verify confidence is ≥ 0.9
- Check inference_worker.py includes landmarks in return
- Verify consumer passes landmarks through message queue

## Support

For issues or questions, check:
1. API logs: Look for `[API]` and `[FeedbackManager]` messages
2. Retraining worker logs: Look for training status messages
3. Database state: Query tables directly to verify data storage
