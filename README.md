# Building LLMs with Apple MLX

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-orange)
![Python](https://img.shields.io/badge/Python-3.10-green)

A comprehensive tutorial and code repository for building, training, and deploying Large Language Models (LLMs) using Apple's MLX framework. This project is designed for developers, researchers, and AI enthusiasts who want to understand and implement LLMs specifically optimized for Apple Silicon hardware.

## 🚀 Quick Start

> **IMPORTANT**: Always work in a dedicated directory outside of your system folders. Never run this project directly in your Downloads or Documents folders.
>
> **ALWAYS use the dedicated ~/ai-training directory for your projects**
>
> **Use the one-step installation method for Miniconda:**
> ```bash
> curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
> ```

```bash
# Create a dedicated directory for AI training projects
mkdir -p ~/ai-training
cd ~/ai-training

# Clone the repository
git clone https://github.com/ddttom/mlx-llm-tutorial
cd mlx-llm-tutorial

# Set up Miniconda environment
conda create -n mlx-env python=3.10
conda activate mlx-env

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook

# Explore the tutorials and code examples
```

## 📖 What is MLX?

MLX is an array framework developed by Apple for machine learning on Apple Silicon. It's designed to provide a familiar API for machine learning researchers and engineers, while leveraging the performance advantages of Apple's hardware.

Key features of MLX include:

- **Apple Silicon Optimization**: Specifically designed for M1/M2/M3 chips
- **NumPy-like API**: Familiar interface for ease of use
- **Automatic Differentiation**: Built-in support for training neural networks
- **Composable Function Transformations**: Efficient memory usage through transformations like `vmap` and `jit`
- **Unified Memory Architecture**: Takes advantage of shared memory between CPU and GPU
- **Multi-device Training**: Support for computation across multiple devices

## 🧠 What are LLMs?

Large Language Models (LLMs) are a type of artificial intelligence model designed to understand, generate, and manipulate human language. They're trained on vast amounts of text data and can perform a wide range of language tasks, from translation to summarization to creative writing.

Examples of LLMs include:

- GPT (Generative Pre-trained Transformer) models
- LLaMA (Large Language Model Meta AI)
- Mistral
- Gemma

## 🗂️ Project Structure

This tutorial is organized into the following sections:

- **public/docs/**: Documentation about MLX, LLMs, and theoretical concepts
  - [Installation Guide](public/docs/installation.md)
  - [Introduction to MLX](public/docs/intro-to-mlx.md)
  - [LLM Architecture](public/docs/llm-architecture.md)
  - [Project Requirements Document](public/docs/prd.md)
- **tutorials/**: Step-by-step guides for different tasks
  - [Building a Simple LLM](tutorials/simple-llm.md)
- **code/**: Example implementations for different aspects of building an LLM
  - `simple_llm.py`: Character-level language model implementation
  - `finetune_llm.py`: Code for fine-tuning pre-trained models
  - `sample_dataset.json`: Example dataset for fine-tuning
  - **web_interface/**: Interactive UI for model experimentation
- **notebooks/**: Jupyter notebooks with interactive tutorials and examples
- **resources/**: Links and references to additional materials
- **public/**: Public-facing website and demo materials

## 💻 Requirements

To get the most out of this tutorial, you'll need:

- A Mac with Apple Silicon (M1/M2/M3)
- macOS Monterey (12.0) or later
- Miniconda (for Python environment management)
- Jupyter Notebook (for interactive tutorials and experimentation)
- At least 8GB RAM (16GB recommended)
- At least 20GB of free disk space
- Basic knowledge of machine learning concepts
- Familiarity with Python programming

## 🛠️ Installation

Follow these steps to set up your environment:

1. **Install Miniconda**:

   Miniconda is required for this project as it provides the best environment management for machine learning on Apple Silicon. Download and install from the official site:

   ```bash
   # Download and run the Miniconda installer in one step (without saving the file)
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
   
   # Follow the prompts to complete installation
   # Make sure to initialize Miniconda when asked
   
   # Restart your terminal or run
   source ~/.zshrc  # or source ~/.bash_profile
   ```

2. **Create a dedicated directory and clone the repository**:

   > **WARNING**: Do NOT clone or run this project directly in your system folders (like Documents or Downloads). Always use a dedicated directory.

   ```bash
   # Create a dedicated directory for AI training projects
   mkdir -p ~/ai-training
   cd ~/ai-training
   
   # Clone the repository
   git clone https://github.com/ddttom/mlx-llm-tutorial
   cd mlx-llm-tutorial
   ```

3. **Create a Miniconda environment**:

   ```bash
   # Create a new environment with Python 3.10
   conda create -n mlx-env python=3.10
   
   # Activate the environment
   conda activate mlx-env
   ```

4. **Install dependencies**:

   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

5. **Verify installation**:

   ```bash
   # Check if MLX is installed correctly
   python -c "import mlx; print(mlx.__version__)"
   
   # Launch Jupyter Notebook to ensure it's working
   jupyter notebook
   ```

For detailed installation instructions, including troubleshooting tips, see the [Installation Guide](public/docs/installation.md).

## 🎓 Learning Path

We recommend following this learning path:

1. Start with the [Introduction to MLX](public/docs/intro-to-mlx.md) to understand the framework
2. Learn about [LLM Architecture](public/docs/llm-architecture.md) to grasp the theoretical concepts
3. Open the Jupyter notebooks to interactively explore the concepts and code
4. Follow the [Building a Simple LLM](tutorials/simple-llm.md) tutorial to create your first model
5. Explore the code examples in the `code/` directory to see practical implementations
6. Experiment with the web interface to interact with your trained models

## 📓 Jupyter Notebooks

This project uses Jupyter Notebooks for interactive learning and experimentation. Notebooks provide several advantages:

- **Interactive Execution**: Run code cells individually and see results immediately, which is crucial for understanding how LLMs work
- **Rich Visualization**: Display charts, graphs, and other visual elements inline to better understand model behavior and performance
- **Narrative Documentation**: Combine code, explanations, and results in a single document for a comprehensive learning experience
- **Experimentation**: Easily modify parameters and see the effects in real-time, encouraging exploration and deeper understanding
- **Documentation as Code**: Notebooks serve as living documentation that combines explanations with executable examples, reinforcing learning through practice

### Running Jupyter Notebooks

You have two options for running Jupyter notebooks:

#### Option 1: Classic Jupyter Notebook Interface

```bash
# Make sure your Miniconda environment is activated
conda activate mlx-env

# Launch the classic Jupyter Notebook interface
jupyter notebook
```

This will open a browser window where you can navigate to the `notebooks/` directory and open any of the tutorial notebooks.

#### Option 2: Visual Studio Code

If you prefer to stay within VS Code, you can use the Jupyter extension:

1. Install the Jupyter extension for VS Code:
   - Open VS Code
   - Go to the Extensions view (Ctrl+Shift+X or Cmd+Shift+X)
   - Search for "Jupyter"
   - Install the "Jupyter" extension by Microsoft

2. Open a Jupyter notebook file (.ipynb) in VS Code
3. Select the Python kernel from your Miniconda environment (mlx-env)
4. Use the interactive interface to run cells and view outputs directly in VS Code

This provides a more integrated development experience if you're already using VS Code for other parts of the project.

## 🌐 Web Interface

This project includes an interactive web interface for experimenting with your trained models. To use it:

1. Navigate to the web interface directory:

   ```bash
   cd code/web_interface
   ```

2. Start the server:

   ```bash
   python server.py
   ```

3. Open your browser and go to `http://localhost:8000`

The interface allows you to:

- Select different models
- Generate text based on prompts
- Adjust generation parameters
- Visualize attention patterns and token probabilities

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this tutorial or add new examples, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

- Apple for developing the MLX framework
- The open-source community for their contributions to LLM research
- All contributors to this tutorial project

## 📚 Additional Resources

For more information about MLX and LLMs, check out:

- [Official MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [MLX Examples Repository](https://github.com/ml-explore/mlx-examples)
- [Apple's MLX Blog Post](https://ml.apple.com/blog/mlx)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)

For a comprehensive list of resources, see our [Resources Guide](resources/resources.md).

## ❓ Troubleshooting

### Common Issues

1. **Conda command not found**:
   - Make sure you've initialized Miniconda during installation
   - Try restarting your terminal
   - Check if Miniconda is in your PATH: `echo $PATH`
   - You may need to run: `source ~/miniconda3/bin/activate`

2. **ImportError: No module named 'mlx'**:
   - Make sure you've activated your Miniconda environment: `conda activate mlx-env`
   - Try reinstalling MLX: `pip uninstall mlx && pip install mlx`
   - Check if you're using the correct Python version: `python --version`

3. **MLX not using Metal backend**:
   - Check if Metal is being used: `python -c "import mlx; print(mlx.metal.is_available())"`
   - If it returns `False`, make sure you're using a Mac with Apple Silicon

4. **Out of Memory Errors**:
   - Reduce model size or batch size
   - Close other memory-intensive applications
   - Consider using model quantization techniques

5. **Jupyter Notebook not launching**:
   - Make sure Jupyter is installed: `pip install jupyter`
   - Try reinstalling: `pip uninstall jupyter && pip install jupyter`
   - Check if the correct kernel is selected in the notebook
   - Verify your browser is not blocking the Jupyter interface
   - If using VS Code, ensure the Jupyter extension is installed and properly configured

If you encounter other issues, please check the [Installation Guide](public/docs/installation.md) or open an issue on GitHub.
