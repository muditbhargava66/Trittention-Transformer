# Trittention-Transformer Examples

This directory contains example scripts and notebooks for using and evaluating the Trittention-Transformer models.

## Example Scripts

* `evaluate_models.py` - Original script for evaluating different attention mechanisms on toy problems
* `train_with_lightning.py` - Train models using PyTorch Lightning integration
* `benchmark_attention.py` - Benchmark different attention mechanisms across varying sequence lengths
* `complexity_visualization.py` - Visualize computational complexity differences
* `finetune_sequence_task.py` - Fine-tune models on sequence modeling tasks

## Notebooks

* `trittention_demo.ipynb` - Interactive exploration of attention patterns and visualizations

## Usage Examples

### Basic Evaluation
```bash
python evaluate_models.py
```

### Training with PyTorch Lightning
```bash
python train_with_lightning.py --attention_type trittention --dataset arithmetic_operations --hidden_size 128 --max_epochs 30
```

### Running Benchmarks
```bash
python benchmark_attention.py --models standard trittention sparse windowed --min_seq_len 10 --max_seq_len 500
```

### Fine-tuning on Sequence Data
```bash
python finetune_sequence_task.py --data_path your_data.csv --attention_type sparse --hidden_size 128 --max_epochs 50
```

### Visualizing Complexity
```bash
python complexity_visualization.py
```

## Running on Google Colab

The notebook `trittention_demo.ipynb` can be run directly on Google Colab. Click the "Open in Colab" badge in the main README.

## Adding New Examples

When creating new examples:

1. Follow the existing code style and organization
2. Include detailed comments and docstrings
3. Add command-line arguments for configuration
4. Update this README with a brief description of your example
