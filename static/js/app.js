// Global variables
let stream = null;
let isCapturing = false;
let isLiveDetection = false;
let predictionInterval = null;
const PREDICTION_RATE = 200; // Predict every 200ms (5 FPS)
const TEMPORAL_BUFFER_SIZE = 30; // Number of frames for temporal detection

// Temporal detection state
let detectionMode = 'static'; // 'static' or 'temporal'
let frameBuffer = []; // Buffer for temporal detection
let gestureHistory = [];
let lastPredictedGesture = null;
let predictionCooldown = false;
const COOLDOWN_DURATION = 2000; // 2 seconds between same gesture predictions

// DOM elements (initialized after DOM loads)
let webcam, canvas, startBtn, stopBtn, toggleLiveBtn, languageSelect, resultsDiv, predictionDiv;
let modeSelect, modeInfo, historyDiv, clearHistoryBtn;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    webcam = document.getElementById('webcam');
    canvas = document.getElementById('canvas');
    startBtn = document.getElementById('startBtn');
    stopBtn = document.getElementById('stopBtn');
    toggleLiveBtn = document.getElementById('toggleLiveBtn');
    languageSelect = document.getElementById('language');
    modeSelect = document.getElementById('detectionMode');
    modeInfo = document.getElementById('modeInfo');
    resultsDiv = document.getElementById('results');
    predictionDiv = document.getElementById('prediction');
    historyDiv = document.getElementById('gestureHistory');
    clearHistoryBtn = document.getElementById('clearHistory');
    
    // Event listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    toggleLiveBtn.addEventListener('click', toggleLiveDetection);
    modeSelect.addEventListener('change', onModeChange);
    clearHistoryBtn.addEventListener('click', clearGestureHistory);
    
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
    
    // Clear buffer when starting
    frameBuffer = [];
    
    if (detectionMode === 'temporal') {
        showMessage('Live detection active... Building frame buffer (0/30)', 'info');
    } else {
        showMessage('Live detection active...', 'success');
    }
    
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
        
        let result;
        
        if (detectionMode === 'temporal') {
            // Add frame to buffer
            frameBuffer.push(imageData);
            
            // Keep buffer at target size
            if (frameBuffer.length > TEMPORAL_BUFFER_SIZE) {
                frameBuffer.shift();
            }
            
            // Only predict when buffer is full
            if (frameBuffer.length >= TEMPORAL_BUFFER_SIZE) {
                result = await predictTemporal(language);
            } else {
                // Show buffer filling progress
                if (frameBuffer.length === 1) {
                    showMessage(`Temporal mode: Buffering frames (${frameBuffer.length}/${TEMPORAL_BUFFER_SIZE})...`, 'info');
                }
                isCapturing = false;
                return; // Wait for more frames
            }
        } else {
            // Static mode - single frame prediction
            result = await predictStatic(imageData, language);
        }
        
        // Display results
        if (result) {
            // Check cooldown for temporal mode
            if (detectionMode === 'temporal') {
                // If same gesture as last and in cooldown, skip
                if (predictionCooldown && result.gesture === lastPredictedGesture) {
                    isCapturing = false;
                    return;
                }
                
                // New gesture or cooldown expired
                if (result.gesture !== lastPredictedGesture || !predictionCooldown) {
                    displayResults(result);
                    addToHistory(result);
                    lastPredictedGesture = result.gesture;
                    
                    // Start cooldown
                    predictionCooldown = true;
                    setTimeout(() => {
                        predictionCooldown = false;
                    }, COOLDOWN_DURATION);
                }
            } else {
                // Static mode - no cooldown
                displayResults(result);
                addToHistory(result);
            }
        }
        
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

// Static prediction (single frame)
async function predictStatic(imageData, language) {
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
    result.gesture_type = 'static';
    return result;
}

// Temporal prediction (sequence of frames)
async function predictTemporal(language) {
    const response = await fetch('/predict/temporal', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            frames: frameBuffer,
            language: language
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Temporal prediction failed');
    }
    
    return await response.json();
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
    
    // Display gesture type if available
    const gestureTypeEl = document.getElementById('gestureType');
    if (result.gesture_type) {
        gestureTypeEl.textContent = result.gesture_type.charAt(0).toUpperCase() + result.gesture_type.slice(1);
        gestureTypeEl.style.color = result.gesture_type === 'dynamic' ? '#007bff' : '#6c757d';
    }
    
    // Only show success message if not in live mode
    if (!isLiveDetection) {
        showMessage('Prediction successful!', 'success');
    }
}

// Handle mode change
function onModeChange() {
    detectionMode = modeSelect.value;
    frameBuffer = []; // Clear buffer when switching modes
    
    // Update info text
    if (detectionMode === 'temporal') {
        modeInfo.textContent = 'Best for dynamic signs like J, Z (requires motion)';
    } else {
        modeInfo.textContent = 'Best for static letters like A, B, C';
    }
    
    console.log('Detection mode changed to:', detectionMode);
}

// Add gesture to history
function addToHistory(result) {
    const timestamp = new Date().toLocaleTimeString();
    gestureHistory.push({
        gesture: result.gesture,
        translation: result.translation,
        confidence: result.confidence,
        timestamp: timestamp
    });
    
    // Keep last 20 gestures
    if (gestureHistory.length > 20) {
        gestureHistory.shift();
    }
    
    updateHistoryDisplay();
}

// Update history display
function updateHistoryDisplay() {
    const placeholder = historyDiv.querySelector('.placeholder');
    if (placeholder && gestureHistory.length > 0) {
        placeholder.style.display = 'none';
    }
    
    // Build gesture sequence string
    const gestureSequence = gestureHistory
        .map(item => item.gesture)
        .filter(g => g !== 'UNKNOWN')
        .join('');
    
    historyDiv.innerHTML = `
        <div class="history-content">
            <div class="gesture-sequence">
                <strong>Gesture Sequence:</strong> ${gestureSequence || '(none)'}
            </div>
            <div class="history-items">
                ${gestureHistory.slice().reverse().slice(0, 10).map(item => `
                    <div class="history-item">
                        <span class="history-gesture">${item.gesture}</span>
                        <span class="history-translation">${item.translation}</span>
                        <span class="history-confidence">${(item.confidence * 100).toFixed(0)}%</span>
                        <span class="history-time">${item.timestamp}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// Clear gesture history
function clearGestureHistory() {
    gestureHistory = [];
    historyDiv.innerHTML = '<p class="placeholder">Start detection to build gesture history...</p>';
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
