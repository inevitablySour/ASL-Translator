// Global variables
let stream = null;
let isCapturing = false;
let isLiveDetection = false;
let predictionInterval = null;
const PREDICTION_RATE = 500; // Predict every 500ms

// Feedback collection during session
let feedbackCandidates = []; // Store high-confidence predictions during live detection

// DOM elements (initialized after DOM loads)
let webcam, canvas, startBtn, stopBtn, toggleLiveBtn, modelSelect, resultsDiv, predictionDiv;
let feedbackPrompt, feedbackYesBtn, feedbackNoBtn, feedbackSummary;
let currentJobId = null;
let currentLandmarks = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    webcam = document.getElementById('webcam');
    canvas = document.getElementById('canvas');
    startBtn = document.getElementById('startBtn');
    stopBtn = document.getElementById('stopBtn');
    toggleLiveBtn = document.getElementById('toggleLiveBtn');
    modelSelect = document.getElementById('model');
    resultsDiv = document.getElementById('results');
    predictionDiv = document.getElementById('prediction');
    feedbackPrompt = document.getElementById('feedbackPrompt');
    feedbackSummary = document.getElementById('feedbackSummary');
    feedbackYesBtn = document.getElementById('feedbackYes');
    feedbackNoBtn = document.getElementById('feedbackNo');
    
    // Event listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    toggleLiveBtn.addEventListener('click', toggleLiveDetection);
    feedbackYesBtn.addEventListener('click', () => submitFeedback(true));
    feedbackNoBtn.addEventListener('click', () => submitFeedback(false));
    
    // Load available models
    loadModels();
});

// Start camera when the button gets pressed.
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

// Toggle live detection (checks if it is already detecting or not)
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
    
    // Clear previous session data
    feedbackCandidates = [];
    
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
    
    // Show feedback prompt if we have high-confidence predictions
    if (feedbackCandidates.length > 0) {
        showFeedbackSummary();
    }
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
        
        // Convert to base64 so that it can be used by mediapipe and get send through the broker.
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Get selected model
        const selectedModel = modelSelect.value;
        
        // Send the converted image to the API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                model: selectedModel
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Store job ID and landmarks for feedback
        currentJobId = result.job_id;
        currentLandmarks = result.landmarks;
        
        // Collect high-confidence predictions during live detection
        if (isLiveDetection && result.landmarks && result.confidence >= 0.9 && result.gesture !== 'NO_HAND') {
            feedbackCandidates.push({
                job_id: result.job_id,
                gesture: result.gesture,
                translation: result.translation,
                confidence: result.confidence,
                landmarks: result.landmarks
            });
        }
        
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
    
    // Don't show feedback prompt during live detection - we'll show it after stopping
    feedbackPrompt.style.display = 'none';
    
    // Only show success message if not in live mode
    if (!isLiveDetection) {
        showMessage('Prediction successful!', 'success');
    }
}

// Show feedback summary after stopping live detection
function showFeedbackSummary() {
    // Group by gesture and count
    const gestureGroups = {};
    feedbackCandidates.forEach(candidate => {
        if (!gestureGroups[candidate.gesture]) {
            gestureGroups[candidate.gesture] = [];
        }
        gestureGroups[candidate.gesture].push(candidate);
    });
    
    // Build summary HTML
    let summaryHTML = '<h4>Training Data Collected</h4>';
    summaryHTML += '<p class="feedback-description">We collected high-confidence predictions during your session:</p>';
    summaryHTML += '<ul class="gesture-summary">';
    
    for (const [gesture, predictions] of Object.entries(gestureGroups)) {
        const avgConfidence = (predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length * 100).toFixed(1);
        summaryHTML += `<li><strong>${gesture}</strong>: ${predictions.length} samples (avg. ${avgConfidence}% confidence)</li>`;
    }
    
    summaryHTML += '</ul>';
    summaryHTML += `<p class="total-samples">Total: <strong>${feedbackCandidates.length} samples</strong></p>`;
    
    feedbackSummary.innerHTML = summaryHTML;
    feedbackPrompt.style.display = 'block';
}

// Submit feedback for all collected samples
async function submitFeedback(accepted) {
    if (feedbackCandidates.length === 0) {
        console.error('No feedback candidates available');
        return;
    }
    
    try {
        if (accepted) {
            // Submit all collected samples
            let successCount = 0;
            for (const candidate of feedbackCandidates) {
                const response = await fetch('/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        job_id: candidate.job_id,
                        accepted: true
                    })
                });
                
                if (response.ok) {
                    successCount++;
                }
            }
            
            showMessage(`Thank you! ${successCount} samples added to training data.`, 'success');
        } else {
            showMessage('No data was saved. Thank you!', 'info');
        }
        
        // Hide feedback prompt and clear candidates
        feedbackPrompt.style.display = 'none';
        feedbackCandidates = [];
        
    } catch (error) {
        console.error('Error submitting feedback:', error);
        showMessage('Failed to submit feedback', 'error');
    }
}

// Load available models from API
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        modelSelect.innerHTML = '';
        
        if (data.models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            modelSelect.appendChild(option);
            modelSelect.disabled = true;
            return;
        }
        
        // Add models to select
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.version;
            const accuracy = model.accuracy ? `(${(model.accuracy * 100).toFixed(1)}%)` : '';
            option.textContent = `${model.version} ${accuracy}`;
            
            // Select active model by default
            if (model.is_active) {
                option.selected = true;
            }
            
            modelSelect.appendChild(option);
        });
        
        // If no active model, select the first one
        if (!data.models.some(m => m.is_active) && modelSelect.options.length > 0) {
            modelSelect.selectedIndex = 0;
        }
        
    } catch (error) {
        console.error('Error loading models:', error);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        modelSelect.disabled = true;
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
