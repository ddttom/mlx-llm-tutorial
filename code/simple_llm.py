#!/usr/bin/env python3
"""
Simple LLM Implementation using MLX

This script implements a small transformer-based language model using Apple's MLX framework.
It can be used to train a model on a text dataset and generate text with the trained model.

Usage:
    python simple_llm.py         # Train the model
    python simple_llm.py generate # Generate text with a trained model
"""

import os
import sys
import json
import time
import requests
import numpy as np
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Data preparation functions
def download_dataset(url, filename):
    """Download a text file if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def create_tokenizer(text, vocab_size=256):
    """Create a simple character-level tokenizer."""
    # For simplicity, we'll use a character-level tokenizer
    chars = sorted(list(set(text)))
    vocab_size = min(vocab_size, len(chars))
    chars = chars[:vocab_size]
    
    # Create character to index and index to character mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return char_to_idx, idx_to_char

def prepare_data(text, char_to_idx, seq_length=64, batch_size=32):
    """Prepare data for training."""
    # Convert text to indices
    data = [char_to_idx.get(ch, 0) for ch in text]
    
    # Create sequences
    n_sequences = len(data) // seq_length
    data = data[:n_sequences * seq_length]
    
    # Reshape data into sequences
    x = np.array(data).reshape(-1, seq_length)
    
    # Create input and target sequences
    # Input: sequences
    # Target: sequences shifted by 1 (next character prediction)
    inputs = x[:, :-1]
    targets = x[:, 1:]
    
    # Convert to MLX arrays
    inputs = mx.array(inputs)
    targets = mx.array(targets)
    
    return inputs, targets

# Model architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Create query, key, value projections
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # Output projection
        self.wo = nn.Linear(d_model, d_model)
    
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Apply softmax to get attention weights
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply attention weights to values
        context = mx.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Apply output projection
        output = self.wo(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def __call__(self, x):
        return self.linear2(mx.gelu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=4, d_ff=512, max_seq_len=63, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
    
    def __call__(self, x):
        batch_size, seq_len = x.shape
        
        # Create position ids
        positions = mx.arange(seq_len)
        
        # Get embeddings
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Create causal mask for decoder-only transformer
        mask = mx.tril(mx.ones((seq_len, seq_len)))
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits

# Training functions
def train_model(model, inputs, targets, epochs=10, lr=1e-3, batch_size=32):
    """Train the model on the dataset."""
    # Define loss function
    def loss_fn(model, x, y):
        logits = model(x)
        # Reshape for cross entropy
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        return mx.mean(mx.losses.cross_entropy(logits, y))
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=lr)
    
    # Get loss and gradient function
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # Number of batches
    n_samples = inputs.shape[0]
    n_batches = n_samples // batch_size
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        inputs_shuffled = inputs[indices]
        targets_shuffled = targets[indices]
        
        # Process batches
        for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = inputs_shuffled[start_idx:end_idx]
            y_batch = targets_shuffled[start_idx:end_idx]
            
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / n_batches
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    
    return model

# Text generation functions
def generate_text(model, char_to_idx, idx_to_char, seed_text, max_length=100, temperature=1.0):
    """Generate text using the trained model."""
    # Convert seed text to indices
    input_ids = [char_to_idx.get(ch, 0) for ch in seed_text]
    
    # Generate text
    for _ in range(max_length):
        # Convert to MLX array
        x = mx.array([input_ids])
        
        # Get model predictions
        logits = model(x)
        
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Convert to probabilities
        probs = mx.softmax(next_token_logits, axis=-1)
        
        # Sample from the distribution
        next_token = mx.random.categorical(probs, num_samples=1)[0].item()
        
        # Add to input ids
        input_ids.append(next_token)
        
        # If we generate a newline, break
        if idx_to_char[next_token] == '\n':
            break
    
    # Convert indices to text
    generated_text = ''.join([idx_to_char[idx] for idx in input_ids])
    
    return generated_text

def optimize_model_for_inference(model):
    """Apply optimizations for faster inference."""
    # Compile the model's forward pass with MLX's JIT
    model.eval()  # Set to evaluation mode (disables dropout)
    
    # Create a jitted version of the forward pass
    @mx.compile
    def forward(x):
        return model(x)
    
    # Return a wrapper that uses the jitted forward function
    class OptimizedModel:
        def __init__(self, model, forward_fn):
            self.model = model
            self.forward = forward_fn
        
        def __call__(self, x):
            return self.forward(x)
        
        def parameters(self):
            return self.model.parameters()
        
        def update(self, params):
            self.model.update(params)
    
    return OptimizedModel(model, forward)

def load_model_and_generate():
    """Load the model and generate text."""
    # Load tokenizer
    with open("tokenizer.json", "r") as f:
        tokenizer_data = json.load(f)
    
    char_to_idx = tokenizer_data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in tokenizer_data["idx_to_char"].items()}
    vocab_size = len(char_to_idx)
    
    # Create model
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
    params = mx.load("simple_llm_model.npz")
    model.update(params)
    
    # Optimize model for inference
    model = optimize_model_for_inference(model)
    
    # Generate text
    seed_texts = [
        "To be or not to be",
        "All the world's a stage",
        "The lady doth protest"
    ]
    
    for seed in seed_texts:
        print(f"\nSeed: {seed}")
        generated = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=seed,
            max_length=100,
            temperature=0.8
        )
        print(f"Generated: {generated}")

def main():
    """Main function to either train or generate text."""
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        load_model_and_generate()
    else:
        # Download a small dataset (Shakespeare's works)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = download_dataset(url, "shakespeare.txt")
        
        # Create tokenizer
        char_to_idx, idx_to_char = create_tokenizer(text)
        vocab_size = len(char_to_idx)
        print(f"Vocabulary size: {vocab_size}")
        
        # Prepare data
        inputs, targets = prepare_data(text, char_to_idx)
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Create model
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            d_ff=256,
            max_seq_len=inputs.shape[1],
            dropout=0.1
        )
        
        # Train model
        model = train_model(
            model=model,
            inputs=inputs,
            targets=targets,
            epochs=5,
            lr=1e-3,
            batch_size=32
        )
        
        # Save model
        mx.save("simple_llm_model.npz", model.parameters())
        
        # Save tokenizer
        tokenizer_data = {
            "char_to_idx": char_to_idx,
            "idx_to_char": {int(k): v for k, v in idx_to_char.items()}
        }
        with open("tokenizer.json", "w") as f:
            json.dump(tokenizer_data, f)

if __name__ == "__main__":
    main()