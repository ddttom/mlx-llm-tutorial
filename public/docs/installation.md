# Installation Guide

This guide will walk you through setting up your environment for developing LLMs with Apple's MLX framework.

## System Requirements

- Mac with Apple Silicon (M1/M2/M3)
- macOS Monterey (12.0) or later
- At least 16GB of RAM recommended (8GB minimum)
- At least 20GB of free disk space

## Create Project Directory

> **CRITICAL INSTRUCTION**: You must create and use a dedicated directory for your AI training projects. Do NOT clone or run this project directly in your system folders (like Documents or Downloads) to avoid potential issues with large files and system conflicts.
>
> **ALWAYS use the dedicated ~/ai-training directory for your projects**
>
> **Use the one-step installation method for Miniconda:**
>
> ```bash
> curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
> ```

```bash
# Create a new directory for AI training projects
mkdir -p ~/ai-training

# Navigate to the new directory
cd ~/ai-training
```

This will be your workspace for all MLX-related development and experiments. Always work within this directory to keep your projects organized and separate from system files.

## Installing Miniconda

Miniconda is required for this project as it provides the optimal environment for machine learning on Apple Silicon. It offers better dependency management, optimized packages, and a consistent experience across different systems.

### Download and Install Miniconda

```bash
# Download and run the Miniconda installer in one step (without saving the file)
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash

# Alternatively, if you prefer to download first:
# curl -o ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
# bash ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh
```

During the installation, you'll see several prompts. Here's what to expect and how to respond:

1. First, you'll need to review and accept the license agreement (press Enter to read through it, then type "yes" to accept).

2. Confirm the installation location (default is usually fine).

3. After the files are extracted, you'll see output similar to this:

```terminal
Downloading and Extracting Packages:

Preparing transaction: done
Executing transaction: -
done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[yes] >>> yes
no change     /Users/username/miniconda3/condabin/conda
no change     /Users/username/miniconda3/bin/conda
no change     /Users/username/miniconda3/bin/conda-env
no change     /Users/username/miniconda3/bin/activate
no change     /Users/username/miniconda3/bin/deactivate
no change     /Users/username/miniconda3/etc/profile.d/conda.sh
no change     /Users/username/miniconda3/etc/fish/conf.d/conda.fish
no change     /Users/username/miniconda3/shell/condabin/Conda.psm1
modified      /Users/username/miniconda3/shell/condabin/conda-hook.ps1
no change     /Users/username/miniconda3/lib/python3.12/site-packages/xontrib/conda.xsh
no change     /Users/username/miniconda3/etc/profile.d/conda.csh
modified      /Users/username/.zshrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

 **Important**: When asked if you want to update your shell profile to automatically initialize conda, type "yes". This ensures conda is properly set up in your environment.

After installation, either restart your terminal or run:

```bash
# Activate the changes in your current shell
source ~/.zshrc  # or source ~/.bash_profile
```

### Verify Miniconda Installation

```bash
# Check conda version
conda --version

# If you see a version number, Miniconda is installed correctly
```

## Setting Up Your MLX Environment

Create and activate a dedicated environment for MLX development:

```bash
# Create a new environment with Python 3.10
conda create -n mlx-env python=3.10

# Activate the environment
conda activate mlx-env
```

## Installing MLX

With your Miniconda environment activated, install MLX and MLX-LM:

```bash
# Make sure your environment is activated
conda activate mlx-env

# Install MLX and MLX-LM
pip install mlx
pip install mlx-lm
```

Verify the installation:

```bash
python -c "import mlx; print(mlx.__version__)"
```

## Installing Jupyter Notebook

Jupyter Notebook is required for this project as it provides an interactive environment for learning and experimentation:

```bash
# Make sure your environment is activated
conda activate mlx-env

# Install Jupyter
pip install jupyter
```

### Running Jupyter Notebooks

```bash
# Launch the Jupyter Notebook interface
jupyter notebook
```

This will open a browser window where you can navigate to the `notebooks/` directory and open any of the tutorial notebooks.

## Installing Additional Dependencies

Install additional dependencies required for the tutorials:

```bash
# Make sure your environment is activated
conda activate mlx-env

# Install dependencies
pip install numpy matplotlib tqdm requests huggingface_hub
```

## Downloading Pre-trained Models

Some tutorials require pre-trained models. You can download them using the Hugging Face Hub:

```bash
# Make sure your environment is activated
conda activate mlx-env

# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Log in to Hugging Face (you'll need an account)
huggingface-cli login

# Download a model (example)
huggingface-cli download mlx-community/Mistral-7B-v0.1-mlx --local-dir models/mistral-7b
```

## Troubleshooting

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

### Getting Help

If you encounter issues not covered here:

- Check the [MLX GitHub repository](https://github.com/ml-explore/mlx)
- Join the [MLX Discord community](https://discord.gg/ZKz3xgEKhv)
- File an issue on our GitHub repository

## Why Miniconda and Jupyter Notebooks?

### Miniconda

Miniconda is required for this project for several important reasons:

1. **Optimized for Machine Learning**: Miniconda provides the most reliable environment for machine learning packages on Apple Silicon, with better compatibility and performance.

2. **Dependency Isolation**: Machine learning projects often have complex dependency requirements. Miniconda creates fully isolated environments that prevent conflicts between different projects and system packages.

3. **Consistent Experience**: Using Miniconda ensures all users have the same development experience, reducing "works on my machine" issues and making troubleshooting more straightforward.

4. **Better Package Management**: Conda packages are often optimized specifically for scientific computing and machine learning, with pre-compiled binaries that work well on Apple Silicon.

5. **Simplified GPU Integration**: When working with GPU acceleration, Miniconda simplifies the installation and management of GPU-enabled packages.

### Jupyter Notebooks

Jupyter Notebooks are required for this project for several important reasons:

1. **Interactive Learning**: Jupyter Notebooks provide an interactive environment where code, explanations, and results coexist, making it ideal for learning complex concepts.

2. **Immediate Feedback**: Users can execute code cells individually and see results immediately, which is crucial for understanding how LLMs work.

3. **Rich Visualization**: Notebooks support inline visualizations, making it easier to understand model behavior, attention patterns, and other complex aspects of LLMs.

4. **Experimentation**: Notebooks facilitate rapid experimentation with different parameters and approaches, which is essential for learning about LLMs.

5. **Documentation as Code**: Notebooks serve as living documentation that combines explanations with executable examples, reinforcing learning through practice.

## Next Steps

Once you have your environment set up, proceed to [Introduction to MLX](intro-to-mlx.md) to start learning about the framework.
