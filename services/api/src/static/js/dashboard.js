// Dashboard Data Management
let gestureChart = null;
let confidenceChart = null;

// Load all dashboard data
async function loadDashboard() {
    try {
        await Promise.all([
            loadServicesHealth(),
            loadProductionStats(),
            loadModels(),
            loadTrainingHistory(),
            loadFeedbackStats()
        ]);
        
        updateLastUpdatedTime();
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showError('Failed to load dashboard data. Please refresh.');
    }
}

// Load services health status
async function loadServicesHealth() {
    try {
        const response = await fetch('/api/services/health');
        const data = await response.json();
        
        // Update each service status
        const services = data.services;
        for (const [serviceName, serviceData] of Object.entries(services)) {
            const statusElement = document.getElementById(`health-${serviceName}`);
            const healthItem = statusElement.closest('.health-item');
            const icon = healthItem.querySelector('.health-icon');
            
            if (statusElement) {
                statusElement.textContent = serviceData.message || serviceData.status;
                
                // Remove old status classes
                healthItem.classList.remove('status-healthy', 'status-unhealthy', 'status-degraded', 'status-unknown');
                
                // Add new status class
                healthItem.classList.add(`status-${serviceData.status}`);
            }
        }
    } catch (error) {
        console.error('Error loading services health:', error);
        // Mark all as unknown if fetch fails
        const healthStatuses = document.querySelectorAll('.health-status');
        healthStatuses.forEach(el => {
            el.textContent = 'Check failed';
            el.closest('.health-item').classList.add('status-unknown');
        });
    }
}

// Load production statistics
async function loadProductionStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        // Update metrics
        document.getElementById('totalPredictions').textContent = stats.total_predictions.toLocaleString();
        document.getElementById('avgLatency').textContent = stats.avg_processing_time_ms.toFixed(1);
        document.getElementById('avgConfidence').textContent = (stats.avg_confidence * 100).toFixed(1) + '%';
        document.getElementById('predictions24h').textContent = stats.predictions_24h.toLocaleString();
        
        // Update gesture distribution chart
        updateGestureChart(stats.gesture_distribution);
        
        // Update confidence distribution chart
        updateConfidenceChart(stats.confidence_distribution);
        
    } catch (error) {
        console.error('Error loading production stats:', error);
    }
}

// Update gesture distribution chart
function updateGestureChart(distribution) {
    const ctx = document.getElementById('gestureChart');
    
    if (gestureChart) {
        gestureChart.destroy();
    }
    
    const gestures = Object.keys(distribution).sort();
    const counts = gestures.map(g => distribution[g]);
    
    gestureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: gestures,
            datasets: [{
                label: 'Predictions',
                data: counts,
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Update confidence distribution chart
function updateConfidenceChart(distribution) {
    const ctx = document.getElementById('confidenceChart');
    
    if (confidenceChart) {
        confidenceChart.destroy();
    }
    
    confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High (≥90%)', 'Medium (70-90%)', 'Low (<70%)'],
            datasets: [{
                data: [
                    distribution.high,
                    distribution.medium,
                    distribution.low
                ],
                backgroundColor: [
                    'rgba(39, 174, 96, 0.8)',
                    'rgba(243, 156, 18, 0.8)',
                    'rgba(231, 76, 60, 0.8)'
                ],
                borderColor: [
                    'rgba(39, 174, 96, 1)',
                    'rgba(243, 156, 18, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Load models list
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const tbody = document.querySelector('#modelsTable tbody');
        tbody.innerHTML = '';
        
        if (data.models.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="loading">No models found</td></tr>';
            return;
        }
        
        data.models.forEach(model => {
            const row = tbody.insertRow();
            
            // Version
            row.insertCell().textContent = model.version;
            
            // Name
            row.insertCell().textContent = model.name;
            
            // Accuracy
            const accuracyCell = row.insertCell();
            if (model.accuracy) {
                accuracyCell.textContent = (model.accuracy * 100).toFixed(2) + '%';
            } else {
                accuracyCell.textContent = 'N/A';
            }
            
            // Status
            const statusCell = row.insertCell();
            const statusSpan = document.createElement('span');
            statusSpan.className = model.is_active ? 'status-active' : 'status-inactive';
            statusSpan.textContent = model.is_active ? 'Active' : 'Inactive';
            statusCell.appendChild(statusSpan);
            
            // Samples (from metadata)
            const samplesCell = row.insertCell();
            if (model.metadata && model.metadata.total_samples) {
                samplesCell.textContent = model.metadata.total_samples.toLocaleString();
            } else {
                samplesCell.textContent = 'N/A';
            }
            
            // Created
            const createdCell = row.insertCell();
            if (model.created_at) {
                const date = new Date(model.created_at);
                createdCell.textContent = date.toLocaleString();
            } else {
                createdCell.textContent = 'N/A';
            }
            
            // Actions
            const actionsCell = row.insertCell();
            if (!model.is_active) {
                const activateBtn = document.createElement('button');
                activateBtn.textContent = 'Activate';
                activateBtn.className = 'activate-btn';
                activateBtn.onclick = () => activateModel(model.id, model.version);
                actionsCell.appendChild(activateBtn);
            } else {
                actionsCell.textContent = '-';
            }
        });
        
    } catch (error) {
        console.error('Error loading models:', error);
        const tbody = document.querySelector('#modelsTable tbody');
        tbody.innerHTML = '<tr><td colspan="7" class="loading">Error loading models</td></tr>';
    }
}

// Load training history
async function loadTrainingHistory() {
    try {
        const response = await fetch('/api/training-history');
        const data = await response.json();
        
        const tbody = document.querySelector('#trainingTable tbody');
        tbody.innerHTML = '';
        
        if (data.training_runs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading">No training runs found</td></tr>';
            return;
        }
        
        data.training_runs.forEach(run => {
            const row = tbody.insertRow();
            
            // Model Version
            row.insertCell().textContent = run.model_version || 'N/A';
            
            // Total Samples
            row.insertCell().textContent = run.samples_used ? run.samples_used.toLocaleString() : '0';
            
            // Feedback Samples
            row.insertCell().textContent = run.feedback_samples ? run.feedback_samples.toLocaleString() : '0';
            
            // Accuracy
            const accuracyCell = row.insertCell();
            if (run.accuracy) {
                accuracyCell.textContent = (run.accuracy * 100).toFixed(2) + '%';
            } else {
                accuracyCell.textContent = 'N/A';
            }
            
            // Status
            const statusCell = row.insertCell();
            const statusSpan = document.createElement('span');
            statusSpan.className = 'status-' + run.status;
            statusSpan.textContent = run.status.charAt(0).toUpperCase() + run.status.slice(1);
            statusCell.appendChild(statusSpan);
            
            // Started
            const startedCell = row.insertCell();
            if (run.started_at) {
                const date = new Date(run.started_at);
                startedCell.textContent = date.toLocaleString();
            } else {
                startedCell.textContent = 'N/A';
            }
            
            // Completed
            const completedCell = row.insertCell();
            if (run.completed_at) {
                const date = new Date(run.completed_at);
                completedCell.textContent = date.toLocaleString();
            } else if (run.status === 'running') {
                completedCell.textContent = 'In Progress...';
            } else {
                completedCell.textContent = 'N/A';
            }
        });
        
    } catch (error) {
        console.error('Error loading training history:', error);
        const tbody = document.querySelector('#trainingTable tbody');
        tbody.innerHTML = '<tr><td colspan="7" class="loading">Error loading training history</td></tr>';
    }
}

// Load feedback statistics
async function loadFeedbackStats() {
    try {
        const response = await fetch('/feedback/stats');
        const stats = await response.json();
        
        document.getElementById('totalFeedback').textContent = stats.total_feedback || 0;
        document.getElementById('unusedFeedback').textContent = stats.unused_feedback || 0;
        
        // Update progress bar
        const progress = Math.min((stats.unused_feedback / stats.threshold) * 100, 100);
        const progressFill = document.getElementById('progressFill');
        progressFill.style.width = progress + '%';
        progressFill.textContent = progress.toFixed(0) + '%';
        
    } catch (error) {
        console.error('Error loading feedback stats:', error);
    }
}

// Activate a model
async function activateModel(modelId, modelVersion) {
    if (!confirm(`Are you sure you want to activate model "${modelVersion}"?\n\nThis will restart the inference service to load the new model.`)) {
        return;
    }
    
    try {
        // Show loading message
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = 'Activating...';
        btn.disabled = true;
        
        const response = await fetch(`/api/models/${modelId}/activate`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            alert(`Model "${modelVersion}" activated successfully!\n\nThe inference service has been restarted and is now using the new model.`);
            // Reload models table
            await loadModels();
        } else {
            alert(`Failed to activate model: ${result.detail || 'Unknown error'}`);
            btn.textContent = originalText;
            btn.disabled = false;
        }
    } catch (error) {
        console.error('Error activating model:', error);
        alert('Failed to activate model. Please check the console for details.');
    }
}

// Update last updated time
function updateLastUpdatedTime() {
    const now = new Date();
    document.getElementById('lastUpdated').textContent = now.toLocaleTimeString();
}

// Show error message
function showError(message) {
    // You could implement a toast notification here
    console.error(message);
}

// Refresh button handler
document.getElementById('refreshBtn').addEventListener('click', () => {
    loadDashboard();
});

// Auto-refresh every 30 seconds
setInterval(loadDashboard, 30000);

// Check health status more frequently (every 10 seconds)
setInterval(loadServicesHealth, 10000);

// Initial load
loadDashboard();
