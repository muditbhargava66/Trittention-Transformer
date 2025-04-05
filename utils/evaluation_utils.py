"""
Evaluation utilities for the Trittention-Transformer project.

This module provides functions for evaluating and benchmarking different attention mechanisms,
as well as visualizing attention patterns and results.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Type, TypeVar, Union, cast

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Store results from model evaluation.
    
    Attributes:
        model_name: Name of the evaluated model.
        loss: Average loss value.
        accuracy: Accuracy score.
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        inference_time: Time taken for inference (seconds).
        memory_usage: Peak memory usage during inference (MB).
        attention_patterns: Optional dictionary of attention patterns.
    """
    model_name: str
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    inference_time: float
    memory_usage: Optional[float] = None
    attention_patterns: Optional[Dict[str, torch.Tensor]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation result to a dictionary.
        
        Returns:
            Dictionary representation of the result.
        """
        return {
            'model_name': self.model_name,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'inference_time': self.inference_time,
            'memory_usage': self.memory_usage
        }
    
    def __str__(self) -> str:
        """Get a string representation of the evaluation result.
        
        Returns:
            Formatted string with evaluation metrics.
        """
        return (
            f"Model: {self.model_name}\n"
            f"Loss: {self.loss:.4f}, Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}\n"
            f"Inference Time: {self.inference_time:.4f}s"
            + (f", Memory Usage: {self.memory_usage:.2f} MB" if self.memory_usage else "")
        )


class ModelProtocol(Protocol):
    """Protocol defining the interface for models that can be evaluated."""
    
    def __call__(self, input_tensor: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            input_tensor: Input tensor.
            attention_mask: Optional attention mask.
            
        Returns:
            Output tensor.
        """
        ...
    
    def train(self, mode: bool = True):
        """Set the model to training or evaluation mode.
        
        Args:
            mode: Whether to set to training mode (True) or evaluation mode (False).
        """
        ...
    
    def eval(self):
        """Set the model to evaluation mode."""
        ...


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                      task_type: Literal['classification', 'regression'] = 'classification',
                      num_classes: Optional[int] = None) -> Dict[str, float]:
    """Calculate evaluation metrics for model outputs.
    
    Args:
        outputs: Model output predictions (batch_size, ...).
        targets: Ground truth targets (batch_size, ...).
        task_type: Type of task ('classification' or 'regression').
        num_classes: Number of classes for classification tasks.
        
    Returns:
        Dictionary of metric names and values.
        
    Raises:
        ValueError: If the task_type is not supported.
    """
    # Convert tensors to numpy arrays
    if isinstance(outputs, torch.Tensor):
        outputs_np = outputs.detach().cpu().numpy()
    else:
        outputs_np = np.array(outputs)
        
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.array(targets)
    
    # Calculate metrics based on task type
    match task_type:
        case 'classification':
            # For classification, convert logits to class predictions
            if outputs_np.shape[-1] > 1:  # Multi-class case
                preds = np.argmax(outputs_np, axis=-1)
            else:  # Binary case
                preds = (outputs_np > 0.5).astype(int)
            
            # Flatten if necessary
            if preds.ndim > 1:
                preds = preds.flatten()
            if targets_np.ndim > 1:
                targets_np = targets_np.flatten()
            
            # Calculate classification metrics
            accuracy = accuracy_score(targets_np, preds)
            
            try:
                precision = precision_score(targets_np, preds, average='macro', zero_division=0)
                recall = recall_score(targets_np, preds, average='macro', zero_division=0)
                f1 = f1_score(targets_np, preds, average='macro', zero_division=0)
            except Exception as e:
                logger.warning(f"Error calculating precision/recall/f1: {e}")
                precision = recall = f1 = 0.0
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
        case 'regression':
            # Calculate regression metrics
            mse = np.mean((outputs_np - targets_np) ** 2)
            mae = np.mean(np.abs(outputs_np - targets_np))
            
            # Calculate R-squared if possible
            try:
                ss_total = np.sum((targets_np - np.mean(targets_np)) ** 2)
                ss_residual = np.sum((targets_np - outputs_np) ** 2)
                r2 = 1 - (ss_residual / ss_total)
            except:
                r2 = 0.0
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                # For compatibility with the EvaluationResult class
                'accuracy': 1.0 - float(mse),  # Use 1-MSE as a proxy for accuracy
                'precision': 1.0,
                'recall': 1.0,
                'f1': float(r2)  # Use R2 as a proxy for F1
            }
            
        case _:
            raise ValueError(f"Unsupported task type: {task_type}")


def evaluate_model(model: ModelProtocol, 
                   dataloader: DataLoader,
                   loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   device: torch.device = torch.device('cpu'),
                   task_type: Literal['classification', 'regression'] = 'classification',
                   attention_mask: Optional[torch.Tensor] = None,
                   collect_attention_patterns: bool = False) -> EvaluationResult:
    """Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader containing evaluation data.
        loss_fn: Loss function.
        device: Device to run evaluation on.
        task_type: Type of task ('classification' or 'regression').
        attention_mask: Optional attention mask to apply to all inputs.
        collect_attention_patterns: Whether to collect attention patterns.
        
    Returns:
        Evaluation results.
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    start_time = time.time()
    
    # Track memory usage if cuda is available
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Apply model
            if attention_mask is not None:
                batch_mask = attention_mask.expand(inputs.size(0), -1, -1, -1).to(device)
                outputs = model(inputs, batch_mask)
            else:
                outputs = model(inputs)
            
            # Calculate loss
            if task_type == 'classification' and outputs.size(-1) > 1:
                # Reshape for cross-entropy loss if needed
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            else:
                loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            
            # Store outputs and targets for metric calculation
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Get peak memory usage
    memory_usage = None
    if device.type == 'cuda':
        memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    
    # Concatenate all outputs and targets
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_outputs_tensor, all_targets_tensor, task_type)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Create and return evaluation result
    result = EvaluationResult(
        model_name=model.__class__.__name__,
        loss=avg_loss,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        inference_time=inference_time,
        memory_usage=memory_usage
    )
    
    return result


def compare_models(models: List[ModelProtocol],
                  dataloader: DataLoader,
                  loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  device: torch.device = torch.device('cpu'),
                  task_type: Literal['classification', 'regression'] = 'classification') -> List[EvaluationResult]:
    """Compare multiple models on the same dataset.
    
    Args:
        models: List of models to compare.
        dataloader: DataLoader containing evaluation data.
        loss_fn: Loss function.
        device: Device to run evaluation on.
        task_type: Type of task ('classification' or 'regression').
        
    Returns:
        List of evaluation results for each model.
    """
    results = []
    
    for model in models:
        logger.info(f"Evaluating model: {model.__class__.__name__}")
        result = evaluate_model(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            task_type=task_type
        )
        results.append(result)
        logger.info(f"Results:\n{result}")
    
    return results


def save_results(results: List[EvaluationResult], output_dir: str | Path) -> str:
    """Save evaluation results to a CSV file and create visualization.
    
    Args:
        results: List of evaluation results.
        output_dir: Directory to save results.
        
    Returns:
        Path to the saved CSV file.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to a DataFrame
    result_dicts = [result.to_dict() for result in results]
    df = pd.DataFrame(result_dicts)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"results_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(df['model_name'], df['accuracy'], color='blue')
    ax1.set_title('Accuracy')
    ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
    
    # Plot F1 score
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(df['model_name'], df['f1'], color='green')
    ax2.set_title('F1 Score')
    ax2.set_xticklabels(df['model_name'], rotation=45, ha='right')
    
    # Plot inference time
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(df['model_name'], df['inference_time'], color='orange')
    ax3.set_title('Inference Time (s)')
    ax3.set_xticklabels(df['model_name'], rotation=45, ha='right')
    
    # Plot loss
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(df['model_name'], df['loss'], color='red')
    ax4.set_title('Loss')
    ax4.set_xticklabels(df['model_name'], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"results_{timestamp}.png"
    plt.savefig(plot_path)
    
    return str(csv_path)


def visualize_attention(attention_patterns: Dict[str, torch.Tensor], 
                        output_path: Optional[str | Path] = None) -> None:
    """Visualize attention patterns.
    
    Args:
        attention_patterns: Dictionary mapping attention type to attention tensor.
        output_path: Path to save the visualization.
    """
    n_patterns = len(attention_patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(6*n_patterns, 5))
    
    # Handle case with only one pattern
    if n_patterns == 1:
        axes = [axes]
    
    for i, (name, pattern) in enumerate(attention_patterns.items()):
        # Convert to numpy and take the first batch and head
        if pattern.dim() > 2:
            # Assuming shape (batch, heads, seq_len, seq_len)
            pattern_np = pattern[0, 0].detach().cpu().numpy()
        else:
            pattern_np = pattern.detach().cpu().numpy()
        
        # Plot heatmap
        im = axes[i].imshow(pattern_np, cmap='viridis')
        axes[i].set_title(f"{name} Attention")
        axes[i].set_xlabel("Token Position")
        axes[i].set_ylabel("Token Position")
        
        # Add colorbar
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def compare_attention_mechanisms(mechanisms: List[torch.nn.Module], 
                                sequence_length: int = 10,
                                hidden_size: int = 64,
                                batch_size: int = 1,
                                output_path: Optional[str | Path] = None) -> Dict[str, torch.Tensor]:
    """Compare different attention mechanisms on the same input.
    
    Args:
        mechanisms: List of attention mechanism modules.
        sequence_length: Length of the sequence to test.
        hidden_size: Hidden size dimension.
        batch_size: Batch size for testing.
        output_path: Path to save visualization.
        
    Returns:
        Dictionary of attention patterns for each mechanism.
    """
    # Create a random hidden states tensor
    hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
    
    # Create a simple attention mask (optional)
    attention_mask = torch.ones(batch_size, sequence_length)
    
    # Dictionary to store attention patterns
    attention_patterns = {}
    
    # Process each attention mechanism
    for mechanism in mechanisms:
        mechanism.eval()
        
        # Get name of the mechanism
        name = mechanism.__class__.__name__
        
        with torch.no_grad():
            # Forward pass
            outputs = mechanism(hidden_states, attention_mask)
            
            # Extract attention patterns if available
            if hasattr(mechanism, 'get_attention_weights'):
                attention_pattern = mechanism.get_attention_weights()
            else:
                # This is a simplified approach - actual implementation would depend on the model
                # Using a dummy attention pattern based on output similarity
                attention_pattern = torch.matmul(outputs, outputs.transpose(-1, -2))
                attention_pattern = torch.softmax(attention_pattern / (hidden_size ** 0.5), dim=-1)
            
            attention_patterns[name] = attention_pattern
    
    # Visualize the patterns
    if output_path or len(mechanisms) > 0:
        visualize_attention(attention_patterns, output_path)
    
    return attention_patterns


def benchmark_complexity(models: List[ModelProtocol],
                         sequence_lengths: List[int],
                         hidden_size: int = 64,
                         batch_size: int = 1,
                         device: torch.device = torch.device('cpu')) -> Dict[str, Dict[str, List[float]]]:
    """Benchmark the computational and memory complexity of models with varying sequence lengths.
    
    Args:
        models: List of models to benchmark.
        sequence_lengths: List of sequence lengths to test.
        hidden_size: Hidden size dimension.
        batch_size: Batch size for testing.
        device: Device to run benchmark on.
        
    Returns:
        Dictionary with model names as keys and benchmark results as values.
    """
    results = {}
    
    for model in models:
        model.eval()
        model_name = model.__class__.__name__
        results[model_name] = {
            'time': [],
            'memory': [] if device.type == 'cuda' else None
        }
        
        for seq_len in sequence_lengths:
            # Create random input
            x = torch.randn(batch_size, seq_len, hidden_size).to(device)
            
            # Warm-up
            with torch.no_grad():
                _ = model(x)
            
            # Reset memory stats if using CUDA
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
            
            # Measure time
            start_time = time.time()
            with torch.no_grad():
                _ = model(x)
            inference_time = time.time() - start_time
            
            # Measure memory if using CUDA
            memory_usage = None
            if device.type == 'cuda':
                memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
            
            # Store results
            results[model_name]['time'].append(inference_time)
            if device.type == 'cuda':
                results[model_name]['memory'].append(memory_usage)
    
    return results


def plot_complexity_results(results: Dict[str, Dict[str, List[float]]],
                           sequence_lengths: List[int],
                           output_path: Optional[str | Path] = None) -> None:
    """Plot complexity benchmark results.
    
    Args:
        results: Benchmark results from benchmark_complexity().
        sequence_lengths: List of sequence lengths tested.
        output_path: Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot inference time
    for model_name, model_results in results.items():
        ax1.plot(sequence_lengths, model_results['time'], marker='o', label=model_name)
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Inference Time (s)')
    ax1.set_title('Inference Time vs. Sequence Length')
    ax1.legend()
    ax1.grid(True)
    
    # Plot memory usage if available
    has_memory = False
    for model_name, model_results in results.items():
        if model_results['memory'] is not None:
            has_memory = True
            ax2.plot(sequence_lengths, model_results['memory'], marker='o', label=model_name)
    
    if has_memory:
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs. Sequence Length')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def optimize_attention_mask(model: ModelProtocol,
                           inputs: torch.Tensor,
                           target: torch.Tensor,
                           loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                           num_iterations: int = 100,
                           learning_rate: float = 0.01) -> torch.Tensor:
    """Optimize an attention mask to improve model performance on a specific input.
    
    Args:
        model: Model to optimize for.
        inputs: Input tensor.
        target: Target tensor.
        loss_fn: Loss function.
        num_iterations: Number of optimization iterations.
        learning_rate: Learning rate for optimization.
        
    Returns:
        Optimized attention mask.
    """
    model.eval()
    
    # Initialize a learnable attention mask
    batch_size, seq_len = inputs.size(0), inputs.size(1)
    attention_mask = torch.zeros(batch_size, 1, 1, seq_len, requires_grad=True)
    
    # Use SGD for optimization
    optimizer = torch.optim.SGD([attention_mask], lr=learning_rate)
    
    # Optimize the attention mask
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply sigmoid to get values between 0 and 1
        mask = torch.sigmoid(attention_mask)
        
        # Forward pass with the mask
        outputs = model(inputs, mask)
        
        # Calculate loss
        loss = loss_fn(outputs, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")
    
    # Return the optimized mask
    return torch.sigmoid(attention_mask.detach())


# Additional utilities for sparse attention implementation

def create_sparse_attention_mask(seq_len: int, sparsity: float = 0.9) -> torch.Tensor:
    """Create a sparse attention mask.
    
    Args:
        seq_len: Sequence length.
        sparsity: Fraction of attention connections to mask out.
        
    Returns:
        Sparse attention mask tensor.
    """
    assert 0 <= sparsity < 1, "Sparsity must be between 0 and 1"
    
    # Start with a full attention mask
    mask = torch.ones(seq_len, seq_len)
    
    # Calculate number of connections to keep
    num_connections = seq_len * seq_len
    num_to_keep = int(num_connections * (1 - sparsity))
    
    # Create a mask with only the diagonal and some random connections
    mask = torch.zeros_like(mask)
    
    # Always keep the diagonal (self-attention)
    indices = torch.arange(seq_len)
    mask[indices, indices] = 1
    diagonal_connections = seq_len
    
    # Calculate remaining connections to add randomly
    remaining = num_to_keep - diagonal_connections
    if remaining > 0:
        # Create all possible indices excluding the diagonal
        i, j = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len), indexing='ij')
        indices = torch.stack([i.flatten(), j.flatten()], dim=1)
        off_diag = indices[i.flatten() != j.flatten()]
        
        # Randomly select indices to keep
        perm = torch.randperm(off_diag.size(0))
        selected = off_diag[perm[:remaining]]
        
        # Set selected indices to 1
        mask[selected[:, 0], selected[:, 1]] = 1
    
    return mask


def apply_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create a sliding window attention mask.
    
    Args:
        seq_len: Sequence length.
        window_size: Size of the attention window.
        
    Returns:
        Sliding window attention mask tensor.
    """
    assert window_size > 0, "Window size must be positive"
    
    # Start with a zero mask
    mask = torch.zeros(seq_len, seq_len)
    
    # Create a sliding window mask
    for i in range(seq_len):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        
        # Set window to 1
        mask[i, start:end] = 1
    
    return mask
