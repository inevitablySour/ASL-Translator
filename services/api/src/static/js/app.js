// Global variables
let stream = null;
let isCapturing = false;
let isLiveDetection = false;
let predictionInterval = null;
const PREDICTION_RATE = 500; // Predict every 500ms

// DOM elements (initialized after DOM loads)
let webcam, canvas, startBtn, stopBtn, toggleLiveBtn, languageSelect, resultsDiv, predictionDiv;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    webcam = document.getElementById('webcam');
    canvas = document.getElementById('canvas');
    startBtn = document.getElementById('startBtn');
    stopBtn = document.getElementById('stopBtn');
    toggleLiveBtn = document.getElementById('toggleLiveBtn');
    languageSelect = document.getElementById('language');
    resultsDiv = document.getElementById('results');
    predictionDiv = document.getElementById('prediction');
    
    // Event listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    toggleLiveBtn.addEventListener('click', toggleLiveDetection);
    
    // Check API health
    checkHealth();
    console.log('ASL Translator loaded');
});

// Start camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480 
            } 
        });
        
        webcam.srcObject = stream;
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        toggleLiveBtn.disabled = false;
        
        showMessage('Camera started! Click "Start Live Detection" to begin.', 'success');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showMessage('Error accessing camera. Please ensure camera permissions are granted.', 'error');
    }
}

// Stop camera
function stopCamera() {
    // Stop live detection if running
    if (isLiveDetection) {
        stopLiveDetection();
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        stream = null;
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        toggleLiveBtn.disabled = true;
        
        showMessage('Camera stopped.', 'info');
    }
}

// Toggle live detection
function toggleLiveDetection() {
    if (isLiveDetection) {
        stopLiveDetection();
    } else {
        startLiveDetection();
    }
}

// Start live detection
function startLiveDetection() {
    isLiveDetection = true;
    toggleLiveBtn.textContent = 'Stop Live Detection';
    toggleLiveBtn.classList.remove('btn-success');
    toggleLiveBtn.classList.add('btn-warning');
    
    showMessage('Live detection active...', 'success');
    
    // Start continuous prediction
    predictionInterval = setInterval(captureAndPredict, PREDICTION_RATE);
}

// Stop live detection
function stopLiveDetection() {
    isLiveDetection = false;
    toggleLiveBtn.textContent = 'Start Live Detection';
    toggleLiveBtn.classList.remove('btn-warning');
    toggleLiveBtn.classList.add('btn-success');
    
    if (predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
    }
    
    showMessage('Live detection stopped.', 'info');
}

// Capture frame and send for prediction
async function captureAndPredict() {
    if (isCapturing) return;
    
    isCapturing = true;
    
    try {
        // Capture frame from video
        const context = canvas.getContext('2d');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Get selected language
        const language = languageSelect.value;
        
        // Send to API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                language: language
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error during prediction:', error);
        // Only show error if not in live detection mode to avoid spam
        if (!isLiveDetection) {
            showMessage(`Error: ${error.message}`, 'error');
        }
    } finally {
        isCapturing = false;
    }
}

// Display prediction results
function displayResults(result) {
    // Hide placeholder
    const placeholder = resultsDiv.querySelector('.placeholder');
    if (placeholder) {
        placeholder.style.display = 'none';
    }
    
    // Show prediction details
    predictionDiv.style.display = 'block';
    
    // Update values
    document.getElementById('gesture').textContent = result.gesture;
    document.getElementById('translation').textContent = result.translation;
    
    // Color code confidence
    const confidenceEl = document.getElementById('confidence');
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidenceEl.textContent = `${confidencePercent}%`;
    
    if (result.confidence > 0.7) {
        confidenceEl.style.color = '#28a745';
    } else if (result.confidence > 0.4) {
        confidenceEl.style.color = '#ffc107';
    } else {
        confidenceEl.style.color = '#dc3545';
    }
    
    document.getElementById('processingTime').textContent = `${result.processing_time_ms.toFixed(2)} ms`;
    
    // Only show success message if not in live mode
    if (!isLiveDetection) {
        showMessage('Prediction successful!', 'success');
    }
}

// Show message to user
function showMessage(message, type = 'info') {
    const placeholder = resultsDiv.querySelector('.placeholder');
    if (placeholder) {
        placeholder.textContent = message;
        placeholder.className = `placeholder ${type}`;
        placeholder.style.display = 'block';
    }
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('API Status:', data);
    } catch (error) {
        console.error('API health check failed:', error);
        showMessage('Warning: Cannot connect to API server', 'error');
    }
}
