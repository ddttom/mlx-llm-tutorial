# LLM Architecture

This document provides an overview of Large Language Model (LLM) architecture, focusing on transformer-based models which are the foundation of modern LLMs.

## Introduction to Transformer Architecture

Most modern LLMs are based on the Transformer architecture, introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. in 2017. The Transformer architecture revolutionized natural language processing by replacing recurrent neural networks (RNNs) with self-attention mechanisms.

The key innovation of Transformers is the ability to process all words in a sequence simultaneously (in parallel) rather than sequentially, while still capturing dependencies between words regardless of their distance in the text.

## Core Components of an LLM

### 1. Tokenization

Before text can be processed by an LLM, it must be converted into tokens. Tokens can be characters, subwords, or whole words, depending on the tokenization strategy.

Common tokenization methods include:

- Byte-Pair Encoding (BPE)
- WordPiece
- SentencePiece
- Unigram

Example of tokenization:

```terminal
Input: "I love machine learning"
Tokens: ["I", "love", "machine", "learning"]
```

Or with subword tokenization:

```terminal
Input: "I love machine learning"
Tokens: ["I", "love", "mach", "ine", "learn", "ing"]
```

### 2. Embedding Layer

The embedding layer converts tokens into dense vector representations. These embeddings capture semantic relationships between tokens.

In addition to token embeddings, positional embeddings are added to provide information about the position of each token in the sequence.

### 3. Transformer Blocks

The core of an LLM consists of multiple stacked Transformer blocks. Each block typically contains:

#### Self-Attention Mechanism

Self-attention allows the model to weigh the importance of different tokens when processing each token in the sequence. The formula for attention is:

```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:

- Q (Query), K (Key), and V (Value) are matrices derived from the input
- d_k is the dimension of the key vectors

Multi-head attention extends this by running multiple attention operations in parallel and concatenating the results.

#### Feed-Forward Neural Networks

Each Transformer block contains a feed-forward neural network that processes each position independently:

```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

This is typically implemented as two linear transformations with a ReLU activation in between.

#### Layer Normalization

Layer normalization is applied before each sub-layer (attention and feed-forward) to stabilize the learning process.

#### Residual Connections

Residual connections (skip connections) are used around each sub-layer to help with gradient flow during training.

### 4. Output Layer

The final layer of an LLM is typically a linear layer followed by a softmax function that converts the model's internal representations into probability distributions over the vocabulary.

## Decoder-Only vs. Encoder-Decoder Architecture

LLMs typically come in two architectural variants:

### Decoder-Only Models

Examples: GPT series, LLaMA, Mistral

- Focus on text generation
- Process text from left to right
- Use masked self-attention to prevent looking at future tokens
- Typically used for tasks like text completion, chatbots, and creative writing

### Encoder-Decoder Models

Examples: T5, BART

- Contain both encoder and decoder components
- Encoder processes the entire input sequence
- Decoder generates the output sequence
- Better suited for tasks like translation, summarization, and question answering

## Model Scaling

LLMs are characterized by their scale, which is typically measured by the number of parameters (weights) in the model.

Scaling dimensions include:

- **Depth**: Number of Transformer layers
- **Width**: Dimension of embeddings and hidden layers
- **Attention heads**: Number of parallel attention mechanisms

Research has shown that increasing model size generally improves performance, leading to the development of increasingly large models:

- GPT-3: 175 billion parameters
- LLaMA 2: Up to 70 billion parameters
- Mistral: 7 billion parameters

## Training LLMs

Training LLMs involves several key components:

### 1. Objective Functions

Most LLMs are trained using a language modeling objective:

- **Causal Language Modeling**: Predict the next token given previous tokens
- **Masked Language Modeling**: Predict masked tokens given surrounding context

### 2. Training Data

LLMs require massive amounts of text data for training, often hundreds of billions of tokens from diverse sources:

- Books
- Websites
- Code repositories
- Scientific papers
- Social media

### 3. Training Techniques

Several techniques are used to improve training efficiency and model performance:

- **Distributed Training**: Training across multiple GPUs/TPUs
- **Mixed Precision Training**: Using lower precision (e.g., FP16) to save memory and increase speed
- **Gradient Accumulation**: Accumulating gradients over multiple batches before updating weights
- **Curriculum Learning**: Starting with easier examples and gradually increasing difficulty

### 4. Fine-tuning

After pre-training on a large corpus, models are often fine-tuned on specific tasks or datasets:

- **Supervised Fine-tuning (SFT)**: Training on human-labeled examples
- **Reinforcement Learning from Human Feedback (RLHF)**: Using human preferences to guide learning
- **Instruction Tuning**: Fine-tuning on instruction-response pairs

## Optimizations for Inference

Several techniques can optimize LLM inference, especially on resource-constrained devices like Apple Silicon:

### 1. Quantization

Reducing the precision of model weights:

- FP16 (16-bit floating point)
- INT8 (8-bit integer)
- INT4 (4-bit integer)

### 2. Pruning

Removing less important weights from the model.

### 3. Knowledge Distillation

Training a smaller "student" model to mimic a larger "teacher" model.

### 4. Efficient Attention Mechanisms

- **Flash Attention**: More memory-efficient attention computation
- **Sparse Attention**: Only computing attention for a subset of token pairs
- **Sliding Window Attention**: Limiting attention to a local window around each token

### 5. KV Caching

Caching key and value tensors from previous forward passes to avoid recomputation.

## LLM Implementation in MLX

MLX provides efficient implementations of transformer architectures optimized for Apple Silicon. Key components include:

```python
import mlx.core as mx
import mlx.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def __call__(self, input_ids, attention_mask=None):
        # Get sequence length
        seq_len = input_ids.shape[1]
        
        # Create position ids
        position_ids = mx.arange(seq_len)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
            
        # Apply final layer norm
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
```

## Next Steps

Now that you understand the basic architecture of LLMs, you can:

1. Explore the [Building a Simple LLM](../tutorials/simple-llm.md) tutorial
2. Learn about [Training Techniques](training-techniques.md)
3. Dive into the code examples in the `code/` directory

## References

- Vaswani, A., et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- Brown, T., et al. (2020). ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165)
- Touvron, H., et al. (2023). ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
- Jiang, A., et al. (2023). ["Mistral 7B"](https://arxiv.org/abs/2310.06825)
