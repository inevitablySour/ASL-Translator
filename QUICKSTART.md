# Quick Start Guide

Get your ASL Translator up and running in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.10+ installed
- ✅ A working webcam
- ✅ Internet connection (for downloading dependencies)

## Step-by-Step Setup

### 1. Navigate to Project Directory
```powershell
cd C:\Users\Joel\Documents\Development\Zuyd\Python\ASL-Translator
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

This will take a few minutes as it downloads all required packages including:
- FastAPI & Uvicorn (web framework)
- MediaPipe (hand detection)
- OpenCV (computer vision)
- MLflow (AI ops)

### 4. Run the Application

**Option A: Using the run script (Easiest)**
```powershell
.\run.bat
```

**Option B: Manual command**
```powershell
.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

You should see output like:
```
Starting ASL Translator v1.0.0
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5. Open Your Browser
Navigate to: **http://localhost:8000**

### 6. Test the Application

1. Click **"Start Camera"** - allow camera permissions when prompted
2. Position your hand in the camera view
3. Make an ASL gesture (try a fist for letter "A")
4. Click **"Capture & Translate"**
5. View your results!

## Testing the API

### Check Health
```powershell
curl http://localhost:8000/health
```

### Get Model Info
```powershell
curl http://localhost:8000/info
```

### View API Documentation
Open: **http://localhost:8000/docs**

## Troubleshooting

### Camera Permission Denied
- Check browser settings
- Ensure no other app is using the camera
- Try a different browser

### Module Import Errors
```powershell
# Deactivate and reactivate virtual environment
deactivate
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use
Change the port:
```powershell
python -m uvicorn src.main:app --port 8001
```

## Next Steps

### 1. Configure Settings
Edit `.env` file (copy from `.env.example`):
```powershell
copy .env.example .env
notepad .env
```

### 2. Try Different Languages
In the web interface, select "Dutch" from the language dropdown

### 3. Explore MLflow
Start MLflow UI in a new terminal:
```powershell
mlflow ui --port 5000
```
Then visit: **http://localhost:5000**

### 4. Train a Custom Model
See README.md "Training a Custom Model" section for details

## Development Mode

For development with auto-reload:
```powershell
python -m uvicorn src.main:app --reload --port 8000
```

## Docker Alternative

If you prefer Docker:
```powershell
docker-compose up -d
```

Access at: **http://localhost:8000**

## Stop the Application

Press `CTRL+C` in the terminal

To deactivate virtual environment:
```powershell
deactivate
```

## Need Help?

- 📖 Read the full README.md
- 🐛 Check GitHub Issues
- 💬 Contact support

---

Happy translating! 🤟
