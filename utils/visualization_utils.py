"""
Visualization utilities for the Trittention-Transformer project.

This module provides functions for visualizing attention patterns, model performance,
and other relevant metrics for better understanding and analysis of the models.
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_attention_matrix(
    attention_matrix: torch.Tensor | np.ndarray,
    title: str = "Attention Matrix",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize an attention matrix as a heatmap.
    
    Args:
        attention_matrix: Attention matrix to visualize (shape: [seq_len, seq_len]).
        title: Title for the plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        cmap: Colormap to use.
        figsize: Figure size.
        
    Returns:
        The matplotlib figure object.
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention_matrix, cmap=cmap)
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label("Attention Weight")
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Token Position")
    
    # Add grid
    ax.grid(False)
    
    # Add ticks
    ax.set_xticks(np.arange(attention_matrix.shape[1]))
    ax.set_yticks(np.arange(attention_matrix.shape[0]))
    
    # Rotate x-axis labels if there are many
    if attention_matrix.shape[1] > 10:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_attention_comparisons(
    attention_matrices: Dict[str, torch.Tensor | np.ndarray],
    title: str = "Attention Comparison",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Compare multiple attention matrices side by side.
    
    Args:
        attention_matrices: Dict mapping names to attention matrices.
        title: Title for the overall plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        cmap: Colormap to use.
        figsize: Figure size (if None, calculated based on number of matrices).
        
    Returns:
        The matplotlib figure object.
    """
    n_matrices = len(attention_matrices)
    assert n_matrices > 0, "At least one attention matrix must be provided"
    
    # Determine layout
    n_cols = min(3, n_matrices)
    n_rows = math.ceil(n_matrices / n_cols)
    
    # Determine figure size if not specified
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Make axes indexable for a single row or column
    if n_matrices == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each attention matrix
    for i, (name, matrix) in enumerate(attention_matrices.items()):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]
        
        # Convert tensor to numpy if needed
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap)
        ax.set_title(name)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Token Position")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Hide any unused subplots
    for i in range(n_matrices, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary mapping metric names to lists of values.
        title: Title for the plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        figsize: Figure size.
        
    Returns:
        The matplotlib figure object.
    """
    assert len(history) > 0, "History dictionary cannot be empty"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each metric
    for name, values in history.items():
        ax.plot(values, label=name)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_model_comparisons(
    results: pd.DataFrame | Dict[str, Dict[str, float]],
    metrics: List[str] = ["loss", "accuracy", "f1", "inference_time"],
    title: str = "Model Comparison",
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot comparisons of different models across multiple metrics.
    
    Args:
        results: DataFrame or dictionary with model results.
        metrics: List of metrics to compare.
        title: Title for the plot.
        save_path: Path to save the visualization image.
        figsize: Figure size.
        
    Returns:
        The matplotlib figure object.
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(results, dict):
        results_df = pd.DataFrame.from_dict(results, orient="index")
    else:
        results_df = results
    
    # Check that all required metrics exist
    available_metrics = [m for m in metrics if m in results_df.columns]
    assert len(available_metrics) > 0, f"None of the requested metrics {metrics} found in results"
    
    # Create figure
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Make axes indexable for a single row or column
    if n_metrics == 1:
        axes = np.array([axes])
    elif n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        row_idx = i // n_cols
        col_idx = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[col_idx]
        elif n_cols == 1:
            ax = axes[row_idx]
        else:
            ax = axes[row_idx, col_idx]
        
        # Plot bar chart
        results_df[metric].plot(kind="bar", ax=ax, colormap="viridis")
        
        # Add labels
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        
        # Add value labels on top of each bar
        for j, v in enumerate(results_df[metric]):
            ax.text(j, v * 1.05, f"{v:.4f}", ha="center")
        
        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    # Hide any unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        if n_rows == 1:
            axes[col_idx].axis('off')
        elif n_cols == 1:
            axes[row_idx].axis('off')
        else:
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_complexity_analysis(
    sequence_lengths: List[int],
    time_complexities: Dict[str, List[float]],
    memory_complexities: Optional[Dict[str, List[float]]] = None,
    title: str = "Complexity Analysis",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 7)
) -> plt.Figure:
    """
    Plot time and memory complexity analysis for different models.
    
    Args:
        sequence_lengths: List of sequence lengths tested.
        time_complexities: Dict mapping model names to lists of time measurements.
        memory_complexities: Dict mapping model names to lists of memory measurements (optional).
        title: Title for the plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        figsize: Figure size.
        
    Returns:
        The matplotlib figure object.
    """
    # Create figure
    fig, axes = plt.subplots(1, 2 if memory_complexities else 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Make axes indexable for a single subplot
    if memory_complexities is None:
        axes = [axes]
    
    # Plot time complexity
    ax_time = axes[0]
    for model_name, times in time_complexities.items():
        ax_time.plot(sequence_lengths, times, marker="o", label=model_name)
    
    ax_time.set_title("Time Complexity")
    ax_time.set_xlabel("Sequence Length")
    ax_time.set_ylabel("Inference Time (s)")
    ax_time.grid(True, linestyle="--", alpha=0.7)
    ax_time.legend()
    
    # Plot memory complexity if provided
    if memory_complexities:
        ax_memory = axes[1]
        for model_name, memory in memory_complexities.items():
            ax_memory.plot(sequence_lengths, memory, marker="s", label=model_name)
        
        ax_memory.set_title("Memory Complexity")
        ax_memory.set_xlabel("Sequence Length")
        ax_memory.set_ylabel("Memory Usage (MB)")
        ax_memory.grid(True, linestyle="--", alpha=0.7)
        ax_memory.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_embeddings(
    embeddings: torch.Tensor | np.ndarray,
    labels: Optional[List[str] | np.ndarray] = None,
    method: str = "tsne",
    title: str = "Embedding Visualization",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    random_state: int = 42
) -> plt.Figure:
    """
    Visualize high-dimensional embeddings in 2D space.
    
    Args:
        embeddings: Embedding vectors (shape: [n_samples, n_features]).
        labels: Labels or classes for each embedding point (optional).
        method: Dimensionality reduction method ("tsne" or "pca").
        title: Title for the plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        figsize: Figure size.
        random_state: Random seed for reproducibility.
        
    Returns:
        The matplotlib figure object.
    """
    # Convert tensor to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        # Convert numeric labels to strings if needed
        if isinstance(labels, np.ndarray) and labels.dtype.kind in "ifu":
            unique_labels = np.unique(labels)
            # Use categorical colormap with enough colors
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    reduced[mask, 0], 
                    reduced[mask, 1], 
                    alpha=0.7, 
                    label=f"Class {label}",
                    c=[cmap(i)]
                )
            
            ax.legend()
        else:
            # Use categorical labels (strings)
            unique_labels = sorted(set(labels))
            # Use categorical colormap with enough colors
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = np.array([l == label for l in labels])
                ax.scatter(
                    reduced[mask, 0], 
                    reduced[mask, 1], 
                    alpha=0.7, 
                    label=label,
                    c=[cmap(i)]
                )
            
            ax.legend()
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    
    # Add labels and title
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_attention_animation(
    attention_matrices: List[torch.Tensor | np.ndarray],
    save_path: str | Path,
    titles: Optional[List[str]] = None,
    interval: int = 200,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis"
) -> None:
    """
    Create an animation of attention matrices changing over time.
    
    Args:
        attention_matrices: List of attention matrices to animate.
        save_path: Path to save the animation (GIF or MP4).
        titles: List of titles for each frame (optional).
        interval: Delay between frames in milliseconds.
        figsize: Figure size.
        cmap: Colormap to use.
    """
    from matplotlib.animation import FuncAnimation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert tensors to numpy if needed
    matrices = []
    for matrix in attention_matrices:
        if isinstance(matrix, torch.Tensor):
            matrices.append(matrix.detach().cpu().numpy())
        else:
            matrices.append(matrix)
    
    # Find global min and max for consistent colorbar
    all_values = np.concatenate([m.flatten() for m in matrices])
    vmin, vmax = all_values.min(), all_values.max()
    
    # Initial plot
    im = ax.imshow(matrices[0], cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label("Attention Weight")
    
    # Set labels
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Token Position")
    
    # Title placeholder
    title_obj = ax.set_title(titles[0] if titles else "Frame 0")
    
    # Update function for animation
    def update(frame):
        im.set_array(matrices[frame])
        title_obj.set_text(titles[frame] if titles else f"Frame {frame}")
        return [im, title_obj]
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(matrices), 
        interval=interval,
        blit=True
    )
    
    # Save animation
    anim.save(save_path)
    plt.close()


def plot_sequence_prediction(
    true_sequence: torch.Tensor | np.ndarray,
    predicted_sequence: torch.Tensor | np.ndarray,
    title: str = "Sequence Prediction",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot true vs. predicted sequence values.
    
    Args:
        true_sequence: Ground truth sequence values.
        predicted_sequence: Predicted sequence values.
        title: Title for the plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        figsize: Figure size.
        
    Returns:
        The matplotlib figure object.
    """
    # Convert tensors to numpy if needed
    if isinstance(true_sequence, torch.Tensor):
        true_sequence = true_sequence.detach().cpu().numpy()
    
    if isinstance(predicted_sequence, torch.Tensor):
        predicted_sequence = predicted_sequence.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot sequences
    x = np.arange(len(true_sequence))
    ax.plot(x, true_sequence, "b-", label="True", marker="o")
    ax.plot(x, predicted_sequence, "r--", label="Predicted", marker="x")
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_attention_map_for_text(
    text: List[str],
    attention_matrix: torch.Tensor | np.ndarray,
    title: str = "Attention Map",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Create an attention map visualization for text data.
    
    Args:
        text: List of tokens or words.
        attention_matrix: Attention matrix (shape: [seq_len, seq_len]).
        title: Title for the plot.
        save_path: Path to save the visualization image.
        show: Whether to display the plot.
        figsize: Figure size (if None, calculated based on text length).
        cmap: Colormap to use.
        
    Returns:
        The matplotlib figure object.
    """
    # Convert tensor to numpy if needed
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    # Ensure dimensions match
    assert attention_matrix.shape[0] == len(text), "Text length must match attention matrix dimensions"
    assert attention_matrix.shape[1] == len(text), "Text length must match attention matrix dimensions"
    
    # Determine figure size if not specified
    if figsize is None:
        # Scale figure size based on text length
        base_size = 8
        scale_factor = max(1, len(text) / 20)  # Adjust based on text length
        figsize = (base_size * scale_factor, base_size * scale_factor)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention_matrix, cmap=cmap)
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label("Attention Weight")
    
    # Set title
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(text)))
    ax.set_yticks(np.arange(len(text)))
    ax.set_xticklabels(text)
    ax.set_yticklabels(text)
    
    # Rotate x labels if text is long
    if len(text) > 10:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells with attention values
    if len(text) <= 20:  # Only annotate for smaller matrices
        for i in range(len(text)):
            for j in range(len(text)):
                text_color = "white" if attention_matrix[i, j] > 0.5 else "black"
                ax.text(j, i, f"{attention_matrix[i, j]:.2f}",
                        ha="center", va="center", color=text_color)
    
    # Add axis labels
    ax.set_xlabel("Token (Column)")
    ax.set_ylabel("Token (Row)")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
