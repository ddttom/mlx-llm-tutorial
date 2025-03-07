#!/usr/bin/env python3
"""
Fine-tuning a Pre-trained LLM with MLX

This script demonstrates how to fine-tune a pre-trained language model using Apple's MLX framework.
It downloads a small pre-trained model from Hugging Face and fine-tunes it on a custom dataset.

Requirements:
    - MLX
    - Hugging Face Hub access
    - At least 8GB of RAM
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

try:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
except ImportError:
    print("Please install the required packages:")
    print("pip install huggingface_hub transformers")
    exit(1)

# Model architecture for a small transformer-based LLM
class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Add any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

class TransformerAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to query, key, value
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Use cached key/value if provided
        if cache is not None:
            past_key, past_value = cache
            key = mx.concatenate([past_key, key], axis=1)
            value = mx.concatenate([past_value, value], axis=1)
            cache = (key, value)
        
        # Transpose for batched matrix multiplication
        query = query.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) / mx.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = mx.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = mx.matmul(attention_weights, value)
        
        # Reshape back to original dimensions
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)
        
        # Apply output projection
        output = self.o_proj(context)
        
        return output, cache

class TransformerMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def __call__(self, x: mx.array) -> mx.array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = TransformerAttention(config)
        self.mlp = TransformerMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cache = self.attention(hidden_states, attention_mask, cache)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, cache

class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply_init()
    
    def apply_init(self):
        """Initialize weights with small random values."""
        # This is a simplified initialization
        # In practice, you would use more sophisticated initialization methods
        pass
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize cache if needed
        if cache is None and self.config.use_cache:
            cache = [(None, None) for _ in range(len(self.layers))]
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = mx.tril(mx.ones((seq_length, seq_length)))
            attention_mask = attention_mask.reshape(1, 1, seq_length, seq_length)
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Apply transformer layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states, new_layer_cache = layer(hidden_states, attention_mask, layer_cache)
            if self.config.use_cache:
                new_cache.append(new_layer_cache)
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, new_cache if self.config.use_cache else None

class TransformerLMHeadModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.transformer = TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        hidden_states, cache = self.transformer(input_ids, attention_mask, cache)
        logits = self.lm_head(hidden_states)
        return logits, cache

def load_pretrained_model(model_name: str, cache_dir: Optional[str] = None) -> Tuple[TransformerLMHeadModel, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer from Hugging Face Hub.
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        cache_dir: Directory to cache the model files
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading pre-trained model: {model_name}")
    
    # Download model files
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_dict = json.load(f)
    
    # Create config object
    config = TransformerConfig(**config_dict)
    
    # Create model
    model = TransformerLMHeadModel(config)
    
    # Load weights
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(weights_path):
        # This is a simplified version - in practice, you would need to convert PyTorch weights to MLX format
        print(f"Loading weights from: {weights_path}")
        # weights = torch.load(weights_path, map_location="cpu")
        # mlx_weights = convert_torch_to_mlx(weights)
        # model.update(mlx_weights)
        print("Note: Weight loading is not implemented in this example")
    else:
        print(f"Warning: Could not find weights at {weights_path}")
    
    return model, tokenizer

def prepare_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Tuple[mx.array, mx.array]:
    """
    Prepare a dataset for fine-tuning.
    
    Args:
        data_path: Path to the dataset file (JSON format)
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (input_ids, labels)
    """
    print(f"Preparing dataset from: {data_path}")
    
    # Load dataset
    with open(data_path, "r") as f:
        dataset = json.load(f)
    
    # Tokenize dataset
    input_ids = []
    labels = []
    
    for example in tqdm(dataset, desc="Tokenizing"):
        # Format example as instruction
        text = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
        
        # Tokenize
        encodings = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        
        # Add to dataset
        input_ids.append(encodings["input_ids"][0])
        
        # Create labels (same as input_ids, but -100 for non-response tokens)
        label = encodings["input_ids"][0].copy()
        
        # Find the position of [/INST]
        inst_end_token = tokenizer.encode(" [/INST] ", add_special_tokens=False)[-1]
        inst_end_pos = np.where(label == inst_end_token)[0][0]
        
        # Set labels for non-response tokens to -100 (ignored in loss calculation)
        label[:inst_end_pos + 1] = -100
        
        labels.append(label)
    
    # Convert to MLX arrays
    input_ids = mx.array(np.array(input_ids))
    labels = mx.array(np.array(labels))
    
    return input_ids, labels

def fine_tune_model(
    model: TransformerLMHeadModel,
    input_ids: mx.array,
    labels: mx.array,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    save_path: str = "fine_tuned_model",
) -> TransformerLMHeadModel:
    """
    Fine-tune a pre-trained model on a custom dataset.
    
    Args:
        model: Pre-trained model
        input_ids: Input token IDs
        labels: Labels for training
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        save_path: Path to save the fine-tuned model
    
    Returns:
        Fine-tuned model
    """
    print("Starting fine-tuning")
    
    # Define loss function
    def loss_fn(model, x, y):
        logits, _ = model(x)
        
        # Reshape for cross entropy
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        
        # Create mask for -100 labels (ignored)
        mask = y != -100
        
        # Apply mask
        logits = logits[mask]
        y = y[mask]
        
        # Compute cross entropy loss
        return mx.mean(mx.losses.cross_entropy(logits, y))
    
    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )
    
    # Get loss and gradient function
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # Number of batches
    n_samples = input_ids.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    # Training loop
    for epoch in range(epochs):
        start_time = mx.utils.get_time()
        epoch_loss = 0.0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        input_ids_shuffled = input_ids[indices]
        labels_shuffled = labels[indices]
        
        # Process batches
        for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = input_ids_shuffled[start_idx:end_idx]
            y_batch = labels_shuffled[start_idx:end_idx]
            
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / n_batches
        elapsed_time = mx.utils.get_time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    mx.save(os.path.join(save_path, "model.npz"), model.parameters())
    
    return model

def generate_text(
    model: TransformerLMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate text using the fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt: Text prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Generated text
    """
    # Format prompt as instruction
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize prompt
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    
    # Initialize cache
    cache = [(None, None) for _ in range(len(model.transformer.layers))]
    
    # Generate tokens
    generated_ids = input_ids[0].tolist()
    
    for _ in range(max_length):
        # Get model predictions
        logits, cache = model(mx.array([generated_ids]), cache=cache)
        
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = mx.sort(next_token_logits, descending=True)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove = mx.concatenate(
                [mx.zeros((1,), dtype=mx.bool_), sorted_indices_to_remove[:-1]],
                axis=0,
            )
            
            # Create a mask for indices to remove
            indices_to_remove = mx.zeros_like(next_token_logits, dtype=mx.bool_)
            indices_to_remove = indices_to_remove.at[sorted_indices[sorted_indices_to_remove]].set(True)
            
            # Set logits for removed indices to negative infinity
            next_token_logits = mx.where(
                indices_to_remove,
                mx.full_like(next_token_logits, -float("inf")),
                next_token_logits,
            )
        
        # Convert to probabilities
        probs = mx.softmax(next_token_logits, axis=-1)
        
        # Sample from the distribution
        next_token = mx.random.categorical(probs, num_samples=1)[0].item()
        
        # Add to generated ids
        generated_ids.append(next_token)
        
        # Check if we've generated an EOS token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained LLM with MLX")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Pre-trained model name")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file (JSON)")
    parser.add_argument("--output", type=str, default="fine_tuned_model", help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--generate", action="store_true", help="Generate text after fine-tuning")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms", help="Prompt for text generation")
    
    args = parser.parse_args()
    
    # Load pre-trained model and tokenizer
    model, tokenizer = load_pretrained_model(args.model)
    
    # Prepare dataset
    input_ids, labels = prepare_dataset(args.data, tokenizer)
    
    # Fine-tune model
    model = fine_tune_model(
        model=model,
        input_ids=input_ids,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.output,
    )
    
    # Generate text if requested
    if args.generate:
        generated_text = generate_text(model, tokenizer, args.prompt)
        print("\nGenerated Text:")
        print(generated_text)

if __name__ == "__main__":
    main()