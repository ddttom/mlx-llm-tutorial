# MLX LLM Web Interface

This directory contains a web interface for interacting with LLMs built using Apple's MLX framework. The interface allows you to generate text, visualize attention patterns, and explore token probabilities.

## Overview

The web interface consists of:

- A frontend built with HTML, CSS, and JavaScript
- A backend server built with Flask that interfaces with MLX models
- Visualization tools for model interpretability

## Prerequisites

Before running the web interface, make sure you have:

1. Trained at least one MLX model (e.g., by following the [Simple LLM tutorial](../../tutorials/simple-llm.md))
2. Python 3.8 or later installed
3. Flask and Flask-CORS installed

## Installation

1. Install the required Python packages:

```bash
pip install flask flask-cors
```

2. Make sure your MLX models are accessible. The server expects to find:
   - `simple_llm_model.npz` and `tokenizer.json` in the parent directory
   - Fine-tuned models in a `fine_tuned_model` directory

## Running the Web Interface

1. Start the server:

```bash
# From the web_interface directory
python server.py

# Or specify custom host, port, and model directory
python server.py --host 0.0.0.0 --port 8000 --model-dir /path/to/models
```

2. Open a web browser and navigate to:

```
http://localhost:8000
```

## Using the Interface

### Model Selection

Choose between available models:

- **Simple LLM**: A small character-level language model trained on Shakespeare
- **Fine-tuned LLM**: A pre-trained model fine-tuned on custom data (if available)

### Text Generation

1. Enter a prompt in the text area
2. Adjust generation parameters:
   - **Temperature**: Controls randomness (higher = more random)
   - **Max Length**: Maximum number of tokens to generate
   - **Top P**: Controls diversity via nucleus sampling
3. Click "Generate" to create text based on your prompt
4. Use "Clear" to reset the interface

### Visualizations

The interface provides two visualization tabs:

1. **Attention Visualization**: Displays attention weights as a heatmap, showing which tokens the model focuses on when generating each output token
2. **Token Probabilities**: Shows the probability distribution over tokens at each generation step

## Customization

### Adding New Models

To add a new model to the interface:

1. Modify the `load_models` function in `server.py` to load your model
2. Add a new model card in `index.html`
3. Update the model handling logic in `generate` endpoint in `server.py`

### Extending Visualizations

To add new visualizations:

1. Add a new tab in the HTML interface
2. Create a new visualization function in `script.js`
3. Add an endpoint in `server.py` to provide the necessary data

## Troubleshooting

### Common Issues

1. **Server won't start**:
   - Check that you have Flask and Flask-CORS installed
   - Ensure the port isn't already in use

2. **Models not loading**:
   - Verify that model files exist in the expected locations
   - Check server logs for specific error messages

3. **Generation not working**:
   - Ensure the API endpoint is correct in `script.js`
   - Check browser console for JavaScript errors
   - Look at server logs for Python errors

### Debugging

Run the server in debug mode for more detailed logs:

```bash
python server.py --debug
```

## Architecture

The web interface follows a client-server architecture:

- **Client**: HTML/CSS/JS frontend that handles user interactions and visualizations
- **Server**: Flask API that loads MLX models and processes generation requests

Communication between client and server happens via JSON-based REST API calls.

## License

This web interface is part of the MLX LLM Tutorial project and is licensed under the MIT License.