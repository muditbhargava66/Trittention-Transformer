"""
Visualization script for computational complexity analysis.

This script creates visualizations to illustrate the computational complexity
differences between different attention mechanisms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Ensure matplotlib uses a modern style
plt.style.use('ggplot')


def plot_theoretical_complexity():
    """
    Plot theoretical computational complexity for different attention mechanisms.
    """
    # Create sequence lengths
    seq_lengths = np.arange(10, 1001, 10)
    
    # Calculate theoretical complexity
    standard_complexity = seq_lengths ** 2  # O(n²)
    trittention_complexity = seq_lengths ** 3  # O(n³)
    sparse_trittention_complexity = 0.1 * seq_lengths ** 3  # O(0.1n³)
    windowed_trittention_complexity = 128 * seq_lengths ** 2  # O(wn²) with w=128
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot complexities
    ax.plot(seq_lengths, standard_complexity, label='Standard Attention (O(n²))', linewidth=2)
    ax.plot(seq_lengths, trittention_complexity, label='Trittention (O(n³))', linewidth=2)
    ax.plot(seq_lengths, sparse_trittention_complexity, label='Sparse Trittention (O(0.1n³))', linewidth=2)
    ax.plot(seq_lengths, windowed_trittention_complexity, label='Windowed Trittention (O(wn²))', linewidth=2)
    
    # Set logarithmic scale for y-axis
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('Sequence Length', fontsize=14)
    ax.set_ylabel('Relative Computational Cost (log scale)', fontsize=14)
    ax.set_title('Theoretical Computational Complexity', fontsize=16)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_complexity_examples():
    """
    Plot complexity examples with real-world sequence length requirements.
    """
    # Define application domains and their typical sequence lengths
    applications = [
        "Social Media Post", 
        "News Article",
        "Code File",
        "Research Paper",
        "Book Chapter",
        "Genome Fragment"
    ]
    
    # Approximate token counts for each application
    token_counts = [50, 500, 2000, 5000, 10000, 20000]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate complexity for each application
    standard_complexity = np.array(token_counts) ** 2
    trittention_complexity = np.array(token_counts) ** 3
    sparse_trittention_complexity = 0.1 * np.array(token_counts) ** 3
    windowed_trittention_complexity = 128 * np.array(token_counts) ** 2
    
    # Normalize to make the chart more readable
    max_value = max(standard_complexity[-1], windowed_trittention_complexity[-1])
    standard_complexity = standard_complexity / max_value
    windowed_trittention_complexity = windowed_trittention_complexity / max_value
    trittention_bar = np.minimum(trittention_complexity / max_value, 1.0)  # Cap at 1.0
    sparse_bar = np.minimum(sparse_trittention_complexity / max_value, 1.0)  # Cap at 1.0
    
    # Set up positions for the bars
    width = 0.2
    positions = np.arange(len(applications))
    
    # Plot bars
    ax.bar(positions - 1.5*width, standard_complexity, width, label='Standard (O(n²))')
    ax.bar(positions - 0.5*width, trittention_bar, width, label='Trittention (O(n³))')
    ax.bar(positions + 0.5*width, sparse_bar, width, label='Sparse Trittention')
    ax.bar(positions + 1.5*width, windowed_trittention_complexity, width, label='Windowed Trittention')
    
    # Add annotations for off-scale bars
    for i, (trit, sparse) in enumerate(zip(trittention_complexity, sparse_trittention_complexity)):
        if trit / max_value > 1.0:
            ax.text(i - 0.5*width, 1.05, f"{trit/max_value:.1e}x", 
                    ha='center', va='bottom', fontsize=8, rotation=90)
        if sparse / max_value > 1.0:
            ax.text(i + 0.5*width, 1.05, f"{sparse/max_value:.1e}x", 
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Set labels and title
    ax.set_xlabel('Application Domain', fontsize=14)
    ax.set_ylabel('Relative Computational Cost (normalized)', fontsize=14)
    ax.set_title('Computational Requirements by Application Domain', fontsize=16)
    
    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(applications, rotation=45, ha='right')
    
    # Add token count labels
    for i, count in enumerate(token_counts):
        ax.text(i, -0.05, f"{count} tokens", ha='center', va='top', fontsize=10)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_scaling_comparison():
    """
    Plot comparison of how different attention mechanisms scale with sequence length.
    """
    # Create sequence lengths
    seq_lengths = np.arange(10, 1001, 10)
    
    # Define theoretical scaling factors
    standard_scale_fn = lambda x: x ** 2  # O(n²)
    trittention_scale_fn = lambda x: x ** 3  # O(n³)
    sparse_scale_fn = lambda x: 0.1 * x ** 3  # Sparse approximation
    windowed_scale_fn = lambda x: 128 * x ** 2  # Window-based
    efficient_scale_fn = lambda x: x ** 2 * np.log(x)  # Theoretical efficient implementation
    
    # Calculate maximum feasible sequence length for each method
    # (Assuming a computational budget equivalent to standard attention at n=1000)
    budget = standard_scale_fn(1000)
    
    def max_feasible_length(scale_fn, budget, max_length=10000):
        """Find maximum n where scale_fn(n) <= budget"""
        # Binary search
        low, high = 10, max_length
        while low < high:
            mid = (low + high + 1) // 2
            if scale_fn(mid) <= budget:
                low = mid
            else:
                high = mid - 1
        return low
    
    standard_max = max_feasible_length(standard_scale_fn, budget)
    trittention_max = max_feasible_length(trittention_scale_fn, budget)
    sparse_max = max_feasible_length(sparse_scale_fn, budget)
    windowed_max = max_feasible_length(windowed_scale_fn, budget)
    efficient_max = max_feasible_length(efficient_scale_fn, budget)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Scaling curves
    ax1.plot(seq_lengths, standard_scale_fn(seq_lengths), label='Standard (O(n²))', linewidth=2)
    ax1.plot(seq_lengths, trittention_scale_fn(seq_lengths), label='Trittention (O(n³))', linewidth=2)
    ax1.plot(seq_lengths, sparse_scale_fn(seq_lengths), label='Sparse Trittention', linewidth=2)
    ax1.plot(seq_lengths, windowed_scale_fn(seq_lengths), label='Windowed Trittention', linewidth=2)
    ax1.plot(seq_lengths, efficient_scale_fn(seq_lengths), label='Efficient Implementation', linewidth=2)
    
    # Set logarithmic scale for y-axis
    ax1.set_yscale('log')
    
    # Add budget line
    ax1.axhline(y=budget, color='r', linestyle='--', alpha=0.5, label='Computation Budget')
    
    # Set labels and title
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Computational Cost (log scale)', fontsize=12)
    ax1.set_title('Scaling Behavior of Attention Mechanisms', fontsize=14)
    
    # Add legend
    ax1.legend(fontsize=10)
    
    # Plot 2: Maximum feasible sequence lengths
    methods = ['Standard', 'Trittention', 'Sparse\nTrittention', 'Windowed\nTrittention', 'Efficient\nImplementation']
    max_lengths = [standard_max, trittention_max, sparse_max, windowed_max, efficient_max]
    
    # Create bars
    bars = ax2.bar(methods, max_lengths)
    
    # Add value labels on top of bars
    for bar, length in zip(bars, max_lengths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{length}', ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax2.set_xlabel('Attention Mechanism', fontsize=12)
    ax2.set_ylabel('Maximum Sequence Length', fontsize=12)
    ax2.set_title('Maximum Feasible Sequence Length\n(Fixed Computational Budget)', fontsize=14)
    
    # Add grid
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_all_visualizations(output_dir):
    """
    Create and save all visualizations.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save visualizations
    visualizations = {
        "theoretical_complexity": plot_theoretical_complexity,
        "complexity_examples": plot_complexity_examples,
        "scaling_comparison": plot_scaling_comparison
    }
    
    for name, plot_fn in visualizations.items():
        fig = plot_fn()
        save_path = output_dir / f"{name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate attention complexity visualizations")
    parser.add_argument("--output_dir", type=str, default="./results/visualizations",
                        help="Directory to save visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_all_visualizations(args.output_dir)
