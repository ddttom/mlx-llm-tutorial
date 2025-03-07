#!/usr/bin/env python3
"""
MLX LLM Demo - API Server

This server provides API endpoints for the MLX LLM Demo web interface.
It loads MLX models and handles text generation requests.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Union, Any

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add parent directory to path to import the model code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model implementations
try:
    from simple_llm import SimpleTransformer, generate_text as simple_generate
except ImportError:
    print("Warning: Could not import simple_llm module")

try:
    from finetune_llm import TransformerLMHeadModel, generate_text as ft_generate
except ImportError:
    print("Warning: Could not import finetune_llm module")

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Global variables
models = {}
tokenizers = {}

def load_models(model_dir: str) -> None:
    """
    Load available MLX models.
    
    Args:
        model_dir: Directory containing model files
    """
    try:
        # Load simple LLM model if available
        simple_model_path = os.path.join(model_dir, "simple_llm_model.npz")
        simple_tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        if os.path.exists(simple_model_path) and os.path.exists(simple_tokenizer_path):
            print(f"Loading simple LLM model from {simple_model_path}")
            
            # Load tokenizer
            with open(simple_tokenizer_path, "r") as f:
                tokenizer_data = json.load(f)
            
            char_to_idx = tokenizer_data["char_to_idx"]
            idx_to_char = {int(k): v for k, v in tokenizer_data["idx_to_char"].items()}
            vocab_size = len(char_to_idx)
            
            # Create model
            import mlx.core as mx
            
            model = SimpleTransformer(
                vocab_size=vocab_size,
                d_model=128,
                num_heads=4,
                num_layers=4,
                d_ff=256,
                max_seq_len=63,
                dropout=0.0  # No dropout during inference
            )
            
            # Load parameters
            params = mx.load(simple_model_path)
            model.update(params)
            
            # Store model and tokenizer
            models["simple-llm"] = model
            tokenizers["simple-llm"] = (char_to_idx, idx_to_char)
            
            print("Simple LLM model loaded successfully")
        else:
            print(f"Simple LLM model files not found at {model_dir}")
        
        # Load fine-tuned model if available
        ft_model_path = os.path.join(model_dir, "fine_tuned_model/model.npz")
        
        if os.path.exists(ft_model_path):
            print(f"Fine-tuned model found at {ft_model_path}, but loading is not implemented")
            # Note: Loading the fine-tuned model would require additional code
            # that depends on the specific model architecture and tokenizer
    
    except Exception as e:
        print(f"Error loading models: {e}")

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    """Serve static files."""
    return send_from_directory('.', path)

@app.route('/status', methods=['GET'])
def status():
    """Check if the server is running."""
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys())
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Get information about available models."""
    model_info = {}
    
    for model_name in models:
        if model_name == "simple-llm":
            model_info[model_name] = {
                "name": "Simple LLM",
                "description": "A small character-level language model trained on Shakespeare",
                "parameters": "~500K",
                "context_length": 63,
                "tokenizer_type": "character"
            }
        elif model_name == "fine-tuned-llm":
            model_info[model_name] = {
                "name": "Fine-tuned LLM",
                "description": "A pre-trained model fine-tuned on custom data",
                "parameters": "~7M",
                "context_length": 512,
                "tokenizer_type": "subword"
            }
    
    return jsonify(model_info)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text based on a prompt."""
    try:
        data = request.json
        
        # Extract parameters
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'simple-llm')
        temperature = data.get('temperature', 0.7)
        max_length = data.get('max_length', 100)
        top_p = data.get('top_p', 0.9)
        
        # Check if model exists
        if model_name not in models:
            return jsonify({
                "error": f"Model {model_name} not found. Available models: {list(models.keys())}"
            }), 404
        
        # Get model and tokenizer
        model = models[model_name]
        
        # Generate text based on model type
        if model_name == "simple-llm":
            char_to_idx, idx_to_char = tokenizers[model_name]
            
            # Track attention for visualization
            attention_weights = []
            token_probs = []
            
            # Generate text
            generated_text = simple_generate(
                model=model,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                seed_text=prompt,
                max_length=max_length,
                temperature=temperature
            )
            
            # For demonstration, create some dummy visualization data
            # In a real implementation, you would extract this from the model
            tokens = list(generated_text)
            dummy_attention = create_dummy_attention_data(tokens)
            dummy_probs = create_dummy_token_probs(tokens)
            
            return jsonify({
                "text": generated_text,
                "attention_weights": dummy_attention,
                "token_probabilities": dummy_probs
            })
        
        elif model_name == "fine-tuned-llm":
            # This would be implemented based on the fine-tuned model's interface
            return jsonify({
                "error": "Fine-tuned model generation not implemented"
            }), 501
        
        else:
            return jsonify({
                "error": f"Unknown model type: {model_name}"
            }), 400
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/attention', methods=['POST'])
def get_attention():
    """Get attention weights for visualization."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'simple-llm')
        
        # This would be implemented to extract attention weights from the model
        # For now, return dummy data
        tokens = list(prompt)
        dummy_data = create_dummy_attention_data(tokens)
        
        return jsonify(dummy_data)
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

def create_dummy_attention_data(tokens: List[str]) -> Dict[str, Any]:
    """Create dummy attention data for visualization."""
    num_tokens = len(tokens)
    num_heads = 4
    
    # Create random attention weights
    weights = []
    for i in range(num_tokens):
        head_weights = []
        for h in range(num_heads):
            # For causal attention, only attend to previous tokens
            row = np.zeros(num_tokens)
            for j in range(i + 1):
                row[j] = np.random.random()
            
            # Normalize to sum to 1
            if row.sum() > 0:
                row = row / row.sum()
            
            head_weights.append(row.tolist())
        weights.append(head_weights)
    
    return {
        "tokens": tokens,
        "weights": weights,
        "num_heads": num_heads
    }

def create_dummy_token_probs(tokens: List[str]) -> Dict[str, Any]:
    """Create dummy token probability data for visualization."""
    num_tokens = len(tokens)
    
    # Create random probabilities
    probs = np.random.random(num_tokens)
    probs = probs / probs.sum()
    
    return {
        "tokens": tokens,
        "probabilities": probs.tolist()
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLX LLM Demo API Server')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on')
    parser.add_argument('--model-dir', type=str, default='..',
                        help='Directory containing model files')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load models
    load_models(args.model_dir)
    
    # Start server
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)