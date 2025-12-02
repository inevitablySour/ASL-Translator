# ASL Translator - Model Training Guide

This guide walks you through collecting training data and training your own ASL gesture recognition model.

## Overview

The training pipeline consists of two main steps:
1. **Data Collection**: Capture hand gesture images with labels
2. **Model Training**: Train a machine learning model on the collected data

## Prerequisites

Ensure you have installed the dependencies with scikit-learn:
```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

## Step 1: Collect Training Data

### Using the Data Collector

Run the data collection tool:
```powershell
.\collect_data.bat
```

Or manually:
```powershell
.venv\Scripts\python.exe -m src.data_collector
```

### Data Collection Controls

- **SPACE**: Capture current hand gesture as a sample
- **N**: Move to next gesture letter
- **Q**: Quit and save

### Best Practices

1. **Consistent Lighting**: Use good, consistent lighting
2. **Multiple Angles**: Vary hand position slightly for each sample
3. **Clean Background**: Use a plain background if possible
4. **Target 50-100 samples** per gesture for decent accuracy
5. **Quality over Quantity**: Ensure hand is clearly visible

### Collection Tips

- Start with a few letters (A, B, C, V) to test the pipeline
- Hold gesture steady when capturing
- Make sure hand landmarks are visible (green dots)
- Collection saves to `data/gestures/[LETTER]/`

### Check Collected Data

View what you've collected:
```powershell
.venv\Scripts\python.exe -c "from src.data_collector import DataCollector; c = DataCollector(); print(c.get_dataset_info())"
```

## Step 2: Train the Model

### Run Training

After collecting data, train the model:
```powershell
.\train_model.bat
```

Or manually:
```powershell
.venv\Scripts\python.exe -m src.model_trainer
```

### Training Output

The training script will:
1. Load all collected gesture data
2. Split into training (80%) and test (20%) sets
3. Train a Random Forest classifier
4. Evaluate model accuracy
5. Save trained model to `models/` directory
6. Log metrics to MLflow

### Model Files

Training creates these files in `models/`:
- `asl_classifier.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature normalization scaler
- `label_encoder.pkl` - Label encoding for gestures

## Step 3: Use Your Trained Model

### Automatic Loading

The application automatically loads trained models on startup. Simply run:
```powershell
.\run.bat
```

The gesture classifier will automatically use your trained model if available.

### Verify Model Loading

Check the console output when starting the app. You should see:
```
Model loaded successfully from models/asl_classifier.pkl
```

## Model Performance

### Expected Accuracy

With good training data:
- **50-100 samples per gesture**: 70-85% accuracy
- **100-200 samples per gesture**: 85-95% accuracy
- **200+ samples per gesture**: 90-98% accuracy

### Improving Accuracy

If accuracy is low:

1. **Collect More Data**: More samples = better accuracy
2. **Consistent Gestures**: Make sure gestures are performed correctly
3. **Better Lighting**: Improve camera lighting
4. **Clean Data**: Remove bad samples from `data/gestures/`
5. **Retrain**: Delete old model and retrain with new data

### Retraining

To retrain with more data:
1. Collect additional samples using data collector
2. Run training script again
3. Restart the application

Old models are automatically replaced.

## MLflow Tracking

### View Training Runs

Start MLflow UI:
```powershell
mlflow ui --port 5000
```

Access at: http://localhost:5000

### MLflow Features

- Compare training runs
- View accuracy metrics
- Track hyperparameters
- Model versioning

## Advanced: Custom Training

### Modify Hyperparameters

Edit `src/model_trainer.py` and adjust:

```python
hyperparameters = {
    "n_estimators": 200,      # More trees = better accuracy (slower)
    "max_depth": 20,          # Tree depth
    "min_samples_split": 5,   # Minimum samples to split
    "min_samples_leaf": 2,    # Minimum samples per leaf
    "random_state": 42
}
```

### Different Model Types

To use a different model (SVM, XGBoost, etc.):
1. Install the library
2. Modify `ModelTrainer` class in `src/model_trainer.py`
3. Add model type to initialization

## Troubleshooting

### "No training data found"

**Problem**: No samples collected yet

**Solution**: Run data collector first to gather training data

### Low Accuracy (<60%)

**Problem**: Insufficient or poor quality training data

**Solutions**:
- Collect more samples (aim for 100+ per gesture)
- Ensure consistent hand positioning
- Check lighting conditions
- Remove outlier samples

### Model Not Loading

**Problem**: Model file not found

**Solution**: Check that `models/asl_classifier.pkl` exists. If not, train a model first.

### Camera Not Working

**Problem**: Webcam access denied

**Solutions**:
- Check camera permissions
- Close other apps using camera
- Try different browser/app

## Dataset Management

### View Dataset Stats

```powershell
.venv\Scripts\python.exe -c "from src.data_collector import DataCollector; import json; c = DataCollector(); print(json.dumps(c.get_dataset_info(), indent=2))"
```

### Clear Dataset

To start fresh:
```powershell
rm -r data/gestures/*
```

### Backup Dataset

```powershell
cp -r data/gestures data/gestures_backup_$(date +%Y%m%d)
```

## Next Steps

After training:
1. Test your model with the web interface
2. Collect more data for gestures with low accuracy
3. Implement additional gestures or phrases
4. Deploy to production

## Example Workflow

Complete workflow for training:

```powershell
# 1. Collect data (collect 50-100 samples per gesture)
.\collect_data.bat

# 2. Train model
.\train_model.bat

# 3. Run application with trained model
.\run.bat

# 4. Test in browser at http://localhost:8000
```

## Resources

- ASL Alphabet Reference: [ASL University](https://www.lifeprint.com/asl101/pages-layout/numbersasl.htm)
- MediaPipe Hands: [Google MediaPipe](https://mediapipe.dev/)
- Scikit-learn: [Documentation](https://scikit-learn.org/)

---

Happy Training! 🚀
