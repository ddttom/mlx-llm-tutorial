// MLX LLM Demo - Frontend Script
// This script handles the UI interactions and API calls for the MLX LLM demo

// Constants
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
    generate: '/generate',
    models: '/models',
    attention: '/attention'
};

// DOM Elements
const modelCards = document.querySelectorAll('.model-card');
const promptInput = document.getElementById('prompt');
const generateBtn = document.getElementById('generate-btn');
const clearBtn = document.getElementById('clear-btn');
const outputArea = document.getElementById('output');
const loadingIndicator = document.getElementById('loading');

// Generation Parameters
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const maxLengthSlider = document.getElementById('max-length');
const maxLengthValue = document.getElementById('max-length-value');
const topPSlider = document.getElementById('top-p');
const topPValue = document.getElementById('top-p-value');

// Visualization Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const attentionHeatmap = document.getElementById('attention-heatmap');
const tokenChart = document.getElementById('token-chart');

// State
let selectedModel = 'simple-llm';
let attentionData = null;
let tokenProbabilities = null;
let attentionChart = null;
let probabilityChart = null;

// Initialize the application
function init() {
    // Set up event listeners
    setupEventListeners();
    
    // Update UI with initial values
    updateSliderValues();
    
    // Check if API is available
    checkApiStatus();
}

// Set up event listeners for UI interactions
function setupEventListeners() {
    // Model selection
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            modelCards.forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            selectedModel = card.dataset.model;
        });
    });
    
    // Generation controls
    generateBtn.addEventListener('click', generateText);
    clearBtn.addEventListener('click', clearOutput);
    
    // Parameter sliders
    temperatureSlider.addEventListener('input', updateSliderValues);
    maxLengthSlider.addEventListener('input', updateSliderValues);
    topPSlider.addEventListener('input', updateSliderValues);
    
    // Tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            button.classList.add('active');
            document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
            
            // Redraw charts if needed
            if (button.dataset.tab === 'attention' && attentionData) {
                drawAttentionHeatmap(attentionData);
            } else if (button.dataset.tab === 'tokens' && tokenProbabilities) {
                drawTokenProbabilities(tokenProbabilities);
            }
        });
    });
}

// Update the displayed values for sliders
function updateSliderValues() {
    temperatureValue.textContent = temperatureSlider.value;
    maxLengthValue.textContent = maxLengthSlider.value;
    topPValue.textContent = topPSlider.value;
}

// Check if the API is available
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (response.ok) {
            console.log('API is available');
        } else {
            showError('API server is not responding correctly');
        }
    } catch (error) {
        showError('Cannot connect to API server. Make sure the server is running.');
        console.error('API connection error:', error);
    }
}

// Generate text based on the prompt and selected parameters
async function generateText() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }
    
    // Show loading indicator
    loadingIndicator.classList.remove('hidden');
    generateBtn.disabled = true;
    
    try {
        const params = {
            prompt: prompt,
            model: selectedModel,
            temperature: parseFloat(temperatureSlider.value),
            max_length: parseInt(maxLengthSlider.value),
            top_p: parseFloat(topPSlider.value)
        };
        
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.generate}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update output area with generated text
        outputArea.textContent = data.text;
        
        // Store visualization data if available
        if (data.attention_weights) {
            attentionData = data.attention_weights;
            drawAttentionHeatmap(attentionData);
        }
        
        if (data.token_probabilities) {
            tokenProbabilities = data.token_probabilities;
            drawTokenProbabilities(tokenProbabilities);
        }
    } catch (error) {
        showError(`Error generating text: ${error.message}`);
        console.error('Generation error:', error);
    } finally {
        // Hide loading indicator
        loadingIndicator.classList.add('hidden');
        generateBtn.disabled = false;
    }
}

// Clear the output area and visualization data
function clearOutput() {
    promptInput.value = '';
    outputArea.textContent = '';
    
    // Clear visualization data
    attentionData = null;
    tokenProbabilities = null;
    
    // Reset visualizations
    if (attentionChart) {
        attentionChart.destroy();
        attentionChart = null;
    }
    
    if (probabilityChart) {
        probabilityChart.destroy();
        probabilityChart = null;
    }
    
    // Show placeholder text
    document.querySelectorAll('.placeholder-text').forEach(el => {
        el.style.display = 'block';
    });
}

// Show error message in the output area
function showError(message) {
    outputArea.innerHTML = `<div class="error-message">${message}</div>`;
}

// Draw attention heatmap visualization
function drawAttentionHeatmap(attentionData) {
    // Hide placeholder text
    document.querySelector('#attention-tab .placeholder-text').style.display = 'none';
    
    // If we already have a chart, destroy it
    if (attentionChart) {
        attentionChart.destroy();
    }
    
    // Prepare data for heatmap
    const labels = attentionData.tokens;
    const datasets = [];
    
    // Create a dataset for each attention head
    for (let head = 0; head < attentionData.weights[0].length; head++) {
        const data = [];
        for (let i = 0; i < attentionData.weights.length; i++) {
            const row = [];
            for (let j = 0; j < attentionData.weights[i][head].length; j++) {
                row.push(attentionData.weights[i][head][j]);
            }
            data.push(row);
        }
        
        datasets.push({
            label: `Head ${head + 1}`,
            data: data,
            hidden: head > 0 // Only show first head by default
        });
    }
    
    // Create chart
    const ctx = document.createElement('canvas');
    attentionHeatmap.innerHTML = '';
    attentionHeatmap.appendChild(ctx);
    
    attentionChart = new Chart(ctx, {
        type: 'heatmap',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Attention Weights Visualization'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const i = context.dataIndex;
                            const j = context.dataset.data.indexOf(context.raw);
                            return `${labels[i]} → ${labels[j]}: ${context.raw.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Key Tokens'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Query Tokens'
                    }
                }
            }
        }
    });
}

// Draw token probabilities visualization
function drawTokenProbabilities(probData) {
    // Hide placeholder text
    document.querySelector('#tokens-tab .placeholder-text').style.display = 'none';
    
    // If we already have a chart, destroy it
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    // Prepare data for chart
    const labels = probData.tokens;
    const datasets = [{
        label: 'Token Probabilities',
        data: probData.probabilities,
        backgroundColor: 'rgba(0, 122, 255, 0.5)',
        borderColor: 'rgb(0, 122, 255)',
        borderWidth: 1
    }];
    
    // Create chart
    const ctx = document.createElement('canvas');
    tokenChart.innerHTML = '';
    tokenChart.appendChild(ctx);
    
    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Token Probabilities'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Probability'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Tokens'
                    }
                }
            }
        }
    });
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);