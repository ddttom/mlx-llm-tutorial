# Understanding the .gitignore File

This document explains the various entries in our project's `.gitignore` file and why they're important for maintaining a clean repository.

## Table of Contents

- [Understanding the .gitignore File](#understanding-the-gitignore-file)
  - [Table of Contents](#table-of-contents)
  - [Python](#python)
  - [Virtual Environment](#virtual-environment)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Model Files and Datasets](#model-files-and-datasets)
  - [MLX Specific](#mlx-specific)
  - [Experiment Tracking](#experiment-tracking)
  - [Operating System Files](#operating-system-files)
  - [IDE and Editor Files](#ide-and-editor-files)
  - [Logs and Diagnostics](#logs-and-diagnostics)
  - [JavaScript and Web Development](#javascript-and-web-development)
  - [Build Outputs](#build-outputs)
  - [Environment Variables](#environment-variables)
  - [Temporary and Generated Files](#temporary-and-generated-files)
  - [Large Files and Data](#large-files-and-data)
    - [Installer Files](#installer-files)
  - [Documentation](#documentation)
  - [Testing](#testing)
  - [Deployment](#deployment)
  - [Project Specific](#project-specific)
  - [Best Practices for .gitignore](#best-practices-for-gitignore)
  - [Conclusion](#conclusion)

## Python

```bash
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
```

These patterns exclude Python bytecode files, compiled extensions, and packaging artifacts. Python creates these files during execution for performance optimization, but they should not be version controlled as they are platform-specific and can be regenerated.

## Virtual Environment

```bash
venv/
ENV/
env/
.venv/
.python-version
```

Virtual environments contain installed dependencies specific to a developer's machine. These directories can be large and are easily recreated with `requirements.txt` or other dependency management files. The `.python-version` file is used by tools like pyenv to specify the Python version.

## Jupyter Notebook

```bash
.ipynb_checkpoints
*.ipynb_checkpoints/
*.nbconvert.ipynb
.jupyter/
jupyter_notebook_config.py
```

Jupyter Notebook creates checkpoint files for recovery purposes. These are temporary files that don't need to be tracked. Configuration files are also excluded as they may contain user-specific settings.

## Model Files and Datasets

```bash
*.npz
*.bin
*.pt
*.pth
*.onnx
*.h5
*.hdf5
*.pkl
*.pickle
shakespeare.txt
*.safetensors
*.mlx
*.mlpackage
*.mlmodel
model_weights/
checkpoints/
pretrained/
embeddings/
vectors/
*.vec
*.npy
*.npz
*.arrow
*.feather
*.parquet
*.zarr
*.lmdb

# Keep sample dataset
!sample_dataset.json
```

Machine learning models and datasets can be extremely large (often gigabytes) and should not be stored in Git repositories. These files are better managed with specialized tools like Git LFS, model registries, or data versioning systems. We specifically exclude common model formats (PyTorch, ONNX, HDF5, MLX) and data formats (NumPy, Arrow, Parquet).

Note that we explicitly keep `sample_dataset.json` with the `!sample_dataset.json` pattern, as this is a small example file needed for tutorials.

## MLX Specific

```bash
.mlx_cache/
mlx_runs/
mlx_logs/
mlx_output/
mlx_checkpoints/
mlx_models/
```

These directories contain MLX-specific artifacts like cached computations, training runs, logs, and model checkpoints. They are regenerated during execution and don't need version control.

## Experiment Tracking

```bash
runs/
wandb/
mlruns/
.aim/
lightning_logs/
tb_logs/
ray_results/
tune_results/
.neptune/
.comet/
.clearml/
.dvc/
.guild/
```

These directories are created by experiment tracking tools like Weights & Biases, MLflow, TensorBoard, and others. They contain logs, metrics, and artifacts from training runs, which should be stored in their respective platforms rather than in Git.

## Operating System Files

```bash
# macOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent
```

Operating system files like macOS's `.DS_Store` contain metadata specific to a user's system and should not be included in repositories.

## IDE and Editor Files

```bash
# VSCode
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
*.code-workspace

# PyCharm
.idea/
*.iml
*.iws
*.ipr
.idea_modules/
```

IDE configuration files are generally user-specific. We exclude most IDE files but keep certain VSCode configuration files (using `!` patterns) that are useful for maintaining consistent development environments across the team.

## Logs and Diagnostics

```bash
logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# Diagnostic reports
report.[0-9]*.[0-9]*.[0-9]*.[0-9]*.json

# Runtime data
pids
*.pid
*.seed
*.pid.lock
```

Log files and diagnostic reports contain runtime information that's not relevant to version control. These files can grow large and change frequently.

## JavaScript and Web Development

```bash
# Directory for instrumented libs
lib-cov

# Coverage directory
coverage/
*.lcov
.nyc_output

# Dependency directories
node_modules/
jspm_packages/
bower_components/

# TypeScript
typings/
*.tsbuildinfo

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn
.yarn-integrity
.yarn/*
!.yarn/patches
!.yarn/plugins
!.yarn/releases
!.yarn/sdks
!.yarn/versions
.pnp.*

# Web related
.cache/
.parcel-cache/
dist/
coverage/
.rollup.cache/
.turbo/
.swc/
.esbuild/
```

These patterns exclude Node.js dependencies, package manager caches, and build artifacts. The `node_modules` directory in particular can contain thousands of files and should never be committed. We also exclude various bundler caches and build outputs.

## Build Outputs

```bash
.next/
.nuxt/
.vuepress/dist/
.serverless/
.fusebox/
.dynamodb/
.webpack/
.vite/
out/
storybook-static/
.docusaurus/
.astro/
.svelte-kit/
.vercel/
.netlify/
```

Build outputs from various frameworks and tools are excluded as they are generated from source code and don't need to be version controlled.

## Environment Variables

```bash
.env
.env.test
.env.local
.env.development
.env.test
.env.production
.env.development.local
.env.test.local
.env.production.local
.envrc
```

Environment variable files often contain sensitive information like API keys and credentials. They should never be committed to version control for security reasons.

## Temporary and Generated Files

```bash
tmp/
temp/
.tmp/
.temp/

# Generated files
__generated__/
generated/
auto-generated/
.gen/
.generated/
```

Temporary and automatically generated files don't need to be tracked as they can be recreated from source code.

## Large Files and Data

```bash
*.csv
*.tsv
*.jsonl
large_data/
data/raw/
data/processed/
data/interim/
data/external/
data/output/
*.zip
*.tar
*.tar.gz
*.tgz
*.gz
*.bz2
*.xz
*.7z
*.rar

# Installers
Miniconda3-latest-MacOSX-arm64.sh
*.exe
*.msi
*.pkg
*.dmg
*.deb
*.rpm
```

Large data files and archives should be excluded from Git repositories, which are designed for source code, not binary data. These files are better managed with specialized tools or stored externally.

### Installer Files

We specifically exclude installer files like the Miniconda installer (`Miniconda3-latest-MacOSX-arm64.sh`), which is over 100MB and exceeds GitHub's file size limit. This has caused issues in the past when users accidentally committed this file to the repository.

> **CRITICAL INSTRUCTION**:
>
> **ALWAYS use the dedicated ~/ai-training directory for your projects**
>
> **Use the one-step installation method for Miniconda that doesn't save the installer file:**
>
> ```bash
> curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
> ```

By following these instructions and working in the dedicated directory, you'll avoid accidentally committing large installer files to the repository.

## Documentation

```bash
docs/_build/
docs/api/
docs/_autosummary/
site/
public/api/
_site/
.docz/
.mkdocs/
.pdoc/
.sphinx/
```

Generated documentation files are excluded as they can be rebuilt from source. This keeps the repository size manageable.

## Testing

```bash
.cypress/
cypress/videos/
cypress/screenshots/
.playwright/
test-results/
playwright-report/
selenium-debug.log
chromedriver.log
geckodriver.log

.pytest_cache/
.hypothesis/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/
.pytype/
.pyre/
.dmypy.json
dmypy.json
```

Test artifacts, screenshots, videos, and coverage reports are excluded as they are generated during test runs and don't need to be version controlled.

## Deployment

```bash
# Docker
.docker/
docker-compose.override.yml
docker-compose.*.yml
!docker-compose.yml

# Cloud/deployment
.terraform/
.terragrunt-cache/
*.tfstate
*.tfstate.backup
.pulumi/
.serverless/
.amplify/
.aws-sam/
.chalice/
.gcloud/
.azure/
.vercel/
.netlify/
```

Deployment configuration files often contain environment-specific settings or credentials. We exclude most deployment files but keep the main `docker-compose.yml` file (using the `!` pattern) as it defines the core infrastructure.

## Project Specific

```bash
promptloc/
.codegpt
__pycache__
.pytest_cache/
.hypothesis/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/
.pytype/
.pyre/
.dmypy.json
dmypy.json
```

These are project-specific directories or files that don't need to be tracked in version control. This includes Python testing and type checking cache files that are specific to our project setup.

## Best Practices for .gitignore

1. **Start with templates**: Use language/framework-specific templates as a starting point
2. **Be specific**: Prefer specific patterns over broad ones to avoid accidentally excluding important files
3. **Use comments**: Group related patterns and add comments to explain their purpose
4. **Use negation patterns**: Use `!` to include specific files that would otherwise be excluded
5. **Test your patterns**: Before committing, check which files are being ignored with `git status --ignored`
6. **Update regularly**: Review and update your .gitignore as your project evolves
7. **Avoid large files**: Never commit large files (>100MB) to Git repositories, as they exceed GitHub's file size limits
8. **Use dedicated directories**: Always work in a dedicated directory (like `~/ai-training`) to avoid accidentally committing files from system folders
9. **Use one-step installation methods**: For installers, use methods that don't save the installer file locally (e.g., `curl URL | bash`)

## Conclusion

A well-maintained .gitignore file is essential for keeping your repository clean, secure, and efficient. It prevents unnecessary files from being tracked, reduces repository size, and avoids conflicts between team members' environments.
