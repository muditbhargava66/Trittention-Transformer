# Usage Examples

This page provides examples of how to use Trittention-Transformer for various tasks and scenarios.

## Basic Usage

### Creating a Model

```python
import torch
from config.cfgs import TrittentionConfig
from models import Trittention

# Create configuration
config = TrittentionConfig(
    hidden_size=128,
    num_attention_heads=4,
    attention_probs_dropout_prob=0.1,
    hidden_dropout_prob=0.1
)

# Initialize model
model = Trittention(config)

# Generate sample input
batch_size, seq_length = 2, 10
hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)

# Forward pass
output = model(hidden_states)
print(output.shape)  # [2, 10, 128]
```

### Using Different Attention Mechanisms

```python
from models import (
    Attention,
    Trittention,
    TrittentionCube,
    SparseTrittention,
    WindowedTrittention
)

# Standard Attention
standard_attn = Attention(config)

# Trittention
trittention = Trittention(config)

# Trittention Cube
trittention_cube = TrittentionCube(config)

# Sparse Trittention
config.sparsity_threshold = 0.1  # Set sparsity threshold
sparse_trittention = SparseTrittention(config)

# Windowed Trittention
config.window_size = 64  # Set window size
windowed_trittention = WindowedTrittention(config)
```

## Training with PyTorch Lightning

### Creating a LightningModule

```python
from models.lightning_module import TrittentionLightningModule, TrittentionDataModule
from utils.data_utils import load_toy_dataset

# Load dataset
dataset = load_toy_dataset("arithmetic_operations")

# Create Lightning module
model = TrittentionLightningModule(
    attention_type="trittention",
    config=config,
    input_size=1,  # Adjust based on your dataset
    hidden_size=128,
    output_size=1,  # Adjust based on your dataset
    learning_rate=1e-3,
    weight_decay=0.01
)

# Create data module
data_module = TrittentionDataModule(
    train_dataset=dataset,
    batch_size=32,
    num_workers=4,
    val_split=0.2
)

# Train model
import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

trainer.fit(model, data_module)
```

### Training from Command Line

```bash
python examples/train_with_lightning.py \
    --attention_type trittention \
    --dataset arithmetic_operations \
    --hidden_size 128 \
    --num_attention_heads 4 \
    --max_epochs 30 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --batch_size 32
```

## Benchmarking and Evaluation

### Benchmarking Different Attention Mechanisms

```python
from utils.evaluation_utils import benchmark_complexity

# Define models to benchmark
models = {
    "Standard Attention": Attention(config),
    "Trittention": Trittention(config),
    "Sparse Trittention": SparseTrittention(config),
    "Windowed Trittention": WindowedTrittention(config)
}

# Define sequence lengths to test
sequence_lengths = [10, 50, 100, 500, 1000]

# Run benchmark
time_results, memory_results = benchmark_complexity(
    models,
    sequence_lengths,
    hidden_size=128,
    batch_size=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Visualize results
from utils.visualization_utils import plot_complexity_analysis

plot_complexity_analysis(
    sequence_lengths=sequence_lengths,
    time_complexities=time_results,
    memory_complexities=memory_results,
    title="Attention Mechanisms Complexity Analysis",
    save_path="complexity_analysis.png"
)
```

### Command Line Benchmarking

```bash
python examples/benchmark_attention.py \
    --models standard trittention sparse windowed \
    --hidden_size 128 \
    --min_seq_len 10 \
    --max_seq_len 1000 \
    --num_steps 6
```

### Model Evaluation

```python
from utils.evaluation_utils import evaluate_model

# Define model and dataloader
model = Trittention(config)
dataloader = data_module.val_dataloader()

# Define loss function
loss_fn = torch.nn.MSELoss()

# Evaluate model
result = evaluate_model(
    model=model,
    dataloader=dataloader,
    loss_fn=loss_fn,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

print(f"Evaluation Result:\n{result}")
```

## Visualizing Attention Patterns

### Basic Visualization

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization_utils import visualize_attention_matrix

# Create sample input
seq_length = 10
hidden_size = 64
input_tensor = torch.randn(1, seq_length, hidden_size)

# Get attention weights
model = Trittention(config)
_, attention_weights = model(input_tensor, output_attentions=True)

# Extract a single head's attention pattern
attention_pattern = attention_weights[0, 0].detach().cpu()

# Visualize attention pattern
visualize_attention_matrix(
    attention_matrix=attention_pattern,
    title="Trittention Attention Pattern",
    figsize=(10, 8)
)
```

### Comparing Multiple Attention Mechanisms

```python
from utils.visualization_utils import visualize_attention_comparisons

# Define models
models = {
    "Standard": Attention(config),
    "Trittention": Trittention(config),
    "Sparse": SparseTrittention(config)
}

# Generate attention patterns
attention_patterns = {}
for name, model in models.items():
    _, attn_weights = model(input_tensor, output_attentions=True)
    attention_patterns[name] = attn_weights[0, 0].detach().cpu()

# Compare attention patterns
visualize_attention_comparisons(
    attention_matrices=attention_patterns,
    title="Comparison of Attention Mechanisms",
    figsize=(15, 5)
)
```

### Command Line Visualization

```bash
python scripts/visualize_attention.py \
    --models standard trittention sparse \
    --hidden_size 64 \
    --seq_length 20 \
    --mode both \
    --text "The quick brown fox jumps over the lazy dog" \
    --save \
    --output_dir ./results/visualizations
```

## Real-World Applications

### Sequence Modeling

```python
from examples.finetune_sequence_task import SequenceDataset

# Load your sequence data
dataset = SequenceDataset(
    data_path="path/to/your/data.csv",
    seq_length=128,
    target_cols=["target_column"],
    feature_cols=["feature1", "feature2", "feature3"],
    normalize=True
)

# Create dataloaders
from torch.utils.data import DataLoader, random_split

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create and train model
model = TrittentionLightningModule(
    attention_type="sparse",  # Use sparse trittention for efficiency
    config=config,
    input_size=len(dataset.feature_cols),
    hidden_size=128,
    output_size=len(dataset.target_cols)
)

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
```

### Command Line Sequence Modeling

```bash
python examples/finetune_sequence_task.py \
    --data_path path/to/your/data.csv \
    --attention_type sparse \
    --hidden_size 128 \
    --num_attention_heads 4 \
    --max_epochs 100 \
    --batch_size 32
```

## Advanced Usage

### Converting Between Attention Types

```python
# Train a model with standard attention
standard_model = TrittentionLightningModule(
    attention_type="standard",
    config=config,
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)
trainer.fit(standard_model, data_module)

# Save the model
torch.save(standard_model.state_dict(), "standard_model.pt")

# Convert to trittention
from scripts.convert_model import convert_model

convert_model({
    "input_model": "standard_model.pt",
    "output_model": "trittention_model.pt",
    "source_type": "standard",
    "target_type": "trittention",
    "hidden_size": hidden_size,
    "num_heads": config.num_attention_heads,
    "input_size": input_size,
    "output_size": output_size
})

# Load the converted model
trittention_model = TrittentionLightningModule(
    attention_type="trittention",
    config=config,
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)
trittention_model.load_state_dict(torch.load("trittention_model.pt"))
```

### Hyperparameter Tuning

```bash
python scripts/hyperparameter_tuning.py \
    --attention_type trittention \
    --dataset arithmetic_operations \
    --n_trials 20 \
    --epochs_per_trial 30 \
    --hidden_size_min 32 \
    --hidden_size_max 256 \
    --learning_rate_min 1e-5 \
    --learning_rate_max 1e-2
```

## Additional Resources

For more examples, check the `examples` directory in the repository:

- `evaluate_models.py` - Evaluate different attention mechanisms
- `train_with_lightning.py` - Train with PyTorch Lightning
- `benchmark_attention.py` - Benchmark performance
- `complexity_visualization.py` - Visualize computational complexity
- `finetune_sequence_task.py` - Fine-tune on sequence tasks
- `trittention_demo.ipynb` - Interactive demo notebook
