# Trittention-Transformer Scripts

This directory contains utility scripts for working with the Trittention-Transformer models.

## Available Scripts

### Setup and Installation

* `initialize.py` - Initialize the project directory structure and check dependencies
* `install_dev.py` - Install the package in development mode for easier imports

### Model Management

* `model_summary.py` - Summarize model architecture and parameter counts
* `convert_model.py` - Convert between different attention mechanism models
* `hyperparameter_tuning.py` - Tune hyperparameters for different attention mechanisms

### Visualization

* `visualize_attention.py` - Visualize attention patterns from different mechanisms
* `compare_complexity.py` - Compare computational complexity of different mechanisms

## Usage Examples

### Initialize Project

```bash
python scripts/initialize.py
```

### Install in Development Mode

```bash
python scripts/install_dev.py
```

### Model Summary

```bash
python scripts/model_summary.py --models standard trittention sparse windowed --hidden_size 256 --num_heads 4
```

### Hyperparameter Tuning

```bash
python scripts/hyperparameter_tuning.py --attention_type trittention --dataset arithmetic_operations --n_trials 20
```

### Convert Between Attention Types

```bash
python scripts/convert_model.py --input_model ./results/checkpoints/trittention_weights.pt --output_model ./results/checkpoints/sparse_weights.pt --source_type trittention --target_type sparse --hidden_size 128 --num_heads 4
```

### Visualize Attention Patterns

```bash
python scripts/visualize_attention.py --models standard trittention sparse --hidden_size 64 --mode both --save
```

### Compare Computational Complexity

```bash
python scripts/compare_complexity.py --models standard trittention sparse windowed --hidden_size 128 --min_seq_len 10 --max_seq_len 1000
```

## Adding New Scripts

When creating new scripts:

1. Follow the existing code style and organization
2. Include detailed docstrings and comments
3. Add command-line arguments for configuration
4. Update this README with a brief description of your script
