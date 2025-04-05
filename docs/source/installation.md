# Installation Guide

This guide will help you set up the Trittention-Transformer project for development, experimentation, or usage in your own projects.

## Prerequisites

Trittention-Transformer requires:

- Python 3.10 or higher
- PyTorch 1.9.0 or higher
- CUDA (optional, for GPU acceleration)

## Installation Options

### Option 1: Install from GitHub (Recommended for Users)

You can install the latest version directly from GitHub:

```bash
git clone https://github.com/muditbhargava66/Trittention-Transformer.git
cd Trittention-Transformer
pip install -r requirements.txt
```

### Option 2: Development Installation

For development, it's recommended to install in development mode:

```bash
git clone https://github.com/muditbhargava66/Trittention-Transformer.git
cd Trittention-Transformer
pip install -e .
```

This allows you to modify the code and see changes without reinstalling the package.

Alternatively, you can use our helper script:

```bash
python scripts/install_dev.py
```

### Option 3: Install with Specific Dependencies

If you want to install only the core dependencies without development tools:

```bash
pip install torch>=1.9.0 numpy>=1.21.0 scikit-learn>=0.24.2
git clone https://github.com/muditbhargava66/Trittention-Transformer.git
cd Trittention-Transformer
pip install -e .
```

## Verification

You can verify your installation by running:

```bash
python scripts/initialize.py --check_only
```

This will check that all required dependencies are properly installed.

## Setting Up the Project

After installation, you can initialize the project directory structure:

```bash
python scripts/initialize.py
```

This will create the necessary directories for storing results, checkpoints, and visualizations.

## Optional Dependencies

### GPU Support

For GPU acceleration, ensure you have a compatible GPU and the appropriate CUDA version for your PyTorch installation. You can check PyTorch's CUDA compatibility on the [PyTorch website](https://pytorch.org/get-started/locally/).

### Visualization Tools

For creating visualizations, install additional dependencies:

```bash
pip install matplotlib>=3.5.1 seaborn>=0.12.0
```

### Hyperparameter Tuning

For hyperparameter tuning support:

```bash
pip install optuna>=2.10.0
```

### Documentation

To build the documentation locally:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
cd docs
make html
```

The documentation will be available at `docs/build/html/index.html`.

## Troubleshooting

### Common Issues

#### ImportError

If you encounter an import error, ensure you've installed the package correctly:

```python
import sys
sys.path.append('/path/to/Trittention-Transformer')
```

#### CUDA Issues

If you encounter CUDA-related errors:

1. Check that your PyTorch version matches your CUDA version
2. Try running with CPU only: `--gpu=False` or set environment variable `CUDA_VISIBLE_DEVICES=''`

#### Version Conflicts

If you encounter version conflicts with other packages, consider using a virtual environment:

```bash
python -m venv trittention-env
source trittention-env/bin/activate  # On Windows: trittention-env\Scripts\activate
pip install -r requirements.txt
```

### Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](faq.md) section
2. Open an issue on the [GitHub repository](https://github.com/muditbhargava66/Trittention-Transformer/issues)
