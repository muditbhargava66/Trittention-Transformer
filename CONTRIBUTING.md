# Contributing to Trittention-Transformer

Thank you for your interest in contributing to the Trittention-Transformer project! This document provides guidelines and instructions to help you contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Code Style](#code-style)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (coming soon). Please ensure that your interactions with the community are respectful and constructive.

## Getting Started

### Setting Up Your Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Trittention-Transformer.git
   cd Trittention-Transformer
   ```
3. Set up a virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

The project is organized as follows:

```
Trittention-Transformer/
├── README.md                   # Project overview and documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Project dependencies
├── config/                     # Configuration classes
│   ├── __init__.py
│   └── cfgs.py                 # TrittentionConfig and other configurations
├── data/                       # Sample datasets
│   └── toy_problems/           # Toy problems for testing
├── examples/                   # Example scripts
│   ├── evaluate_models.py      # Script for evaluating models
│   └── train_with_lightning.py # Script for training with PyTorch Lightning
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── attention.py            # Standard attention
│   ├── trittention.py          # Trittention implementation
│   ├── trittention_cube.py     # Trittention cube implementation
│   ├── local_trittention.py    # Local trittention
│   ├── mixed_attention.py      # Mixed attention
│   ├── sparse_trittention.py   # Sparse trittention optimizations
│   └── lightning_module.py     # PyTorch Lightning integration
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_trittention.py
│   └── test_sparse_trittention.py
└── utils/                      # Utility functions
    ├── __init__.py
    ├── data_utils.py           # Data loading and processing
    ├── evaluation_utils.py     # Evaluation metrics and utilities
    └── visualization_utils.py  # Visualization tools
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear descriptive title
2. A detailed description of the bug
3. Steps to reproduce the bug
4. Expected behavior
5. Actual behavior
6. Environment information (OS, Python version, package versions)
7. Screenshots (if applicable)

### Suggesting Enhancements

We welcome suggestions for enhancing the project. To suggest an enhancement:

1. Create an issue with a clear descriptive title prefixed with "[Enhancement]"
2. Provide a detailed description of the enhancement
3. Explain why this enhancement would be useful
4. Provide examples of how the enhancement would be used (if applicable)

### Pull Requests

1. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. Make your changes with appropriate tests and documentation

3. Ensure all tests pass:
   ```bash
   python -m unittest discover
   ```

4. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: your feature description"
   # or
   git commit -m "Fix: description of the bug you fixed"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a pull request from your fork to the main repository

7. Wait for maintainers to review your PR and address any feedback

## Development Guidelines

### Code Style

We follow PEP 8 and modern Python best practices:

- Use type hints for function parameters and return values
- Use docstrings for all functions, classes, and modules
- Use descriptive variable and function names
- Limit line length to 88 characters
- Use modern Python features (Python 3.10+)
- Use assertions liberally to validate assumptions

### Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Use unittest for testing (pytest is also acceptable)
- Test both positive and negative cases (including edge cases)

To run tests:
```bash
python -m unittest discover
```

For benchmarking tests:
```bash
# Recommended: Use the pytest runner script
python scripts/run_benchmark_pytest.py

# Alternative: Set the environment variable manually and run with pytest
PYTEST_BENCHMARK=1 pytest tests/test_sparse_trittention.py::TestSparseTrittention::test_benchmark -v
```

### Documentation

- Document all public functions, classes, and modules
- Keep documentation up to date with code changes
- Use clear, concise language in documentation
- Provide examples where appropriate
- Update the README.md when adding major features

## Community

We're excited to build a community around the Trittention-Transformer project. Feel free to reach out to maintainers with questions or suggestions.

Thank you for contributing to Trittention-Transformer!
