"""
Script to compare the complexity of different attention mechanisms.

This script benchmarks different attention mechanisms and visualizes their
computational and memory complexity across varying sequence lengths.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.append(str(project_dir))

from config.cfgs import TrittentionConfig
from models import (
    Attention,
    Trittention,
    TrittentionCube,
    LocalTrittention,
    MixedAttention,
    SparseTrittention,
    WindowedTrittention
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare attention mechanism complexity")
    
    parser.add_argument("--models", type=str, nargs="+",
                        default=["standard", "trittention", "sparse", "windowed"],
                        help="Attention models to benchmark")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for models")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=64,
                        help="Window size for local attention mechanisms")
    parser.add_argument("--min_seq_len", type=int, default=10,
                        help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1000,
                        help="Maximum sequence length")
    parser.add_argument("--num_steps", type=int, default=8,
                        help="Number of sequence length steps")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs for each configuration")
    parser.add_argument("--output_dir", type=str, default="./results/benchmarks",
                        help="Directory to save results")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def create_models(args):
    """
    Create attention models according to arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary mapping model names to model instances
    """
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create configuration
    config = TrittentionConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        attention_probs_dropout_prob=0.0,  # Disable dropout for benchmarking
        window_size=args.window_size
    )
    setattr(config, 'sparsity_threshold', 0.1)  # For sparse trittention
    
    # Create models dictionary
    models = {}
    
    # Initialize requested models
    for model_name in args.models:
        if model_name.lower() == "standard":
            models["Standard Attention"] = Attention(config)
        elif model_name.lower() == "trittention":
            models["Trittention"] = Trittention(config)
        elif model_name.lower() == "trittention_cube":
            models["Trittention Cube"] = TrittentionCube(config)
        elif model_name.lower() == "local":
            config.use_local_trittention = True
            models["Local Trittention"] = LocalTrittention(config)
            config.use_local_trittention = False
        elif model_name.lower() == "mixed":
            config.use_mixed_attention = True
            models["Mixed Attention"] = MixedAttention(config)
            config.use_mixed_attention = False
        elif model_name.lower() == "sparse":
            models["Sparse Trittention"] = SparseTrittention(config)
        elif model_name.lower() == "windowed":
            models["Windowed Trittention"] = WindowedTrittention(config)
        else:
            print(f"Warning: Unknown model '{model_name}', skipping")
    
    # Set all models to eval mode
    for model in models.values():
        model.eval()
    
    return models


def generate_sequence_lengths(min_len, max_len, num_steps):
    """
    Generate sequence lengths on a logarithmic scale.
    
    Args:
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        num_steps: Number of steps
        
    Returns:
        List of sequence lengths
    """
    # Use logarithmic scale to better visualize complexity differences
    return [int(np.round(np.exp(x))) for x in np.linspace(
        np.log(min_len), np.log(max_len), num_steps
    )]


def benchmark_models(models, sequence_lengths, hidden_size, device, num_runs=3, verbose=False):
    """
    Benchmark models with different sequence lengths.
    
    Args:
        models: Dictionary mapping model names to model instances
        sequence_lengths: List of sequence lengths
        hidden_size: Hidden size dimension
        device: Device to run on
        num_runs: Number of runs for each configuration
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (time_results, memory_results)
    """
    # Move models to device
    for model in models.values():
        model.to(device)
    
    # Dictionaries to store results
    time_results = {name: [] for name in models.keys()}
    memory_results = {name: [] for name in models.keys()} if device.type == "cuda" else None
    
    # Run benchmarks for each sequence length
    for seq_len in sequence_lengths:
        if verbose:
            print(f"Benchmarking sequence length: {seq_len}")
        
        # Create random input
        x = torch.randn(1, seq_len, hidden_size, device=device)
        
        # Benchmark each model
        for name, model in models.items():
            if verbose:
                print(f"  Testing {name}...")
            
            # Run multiple times and take the minimum
            run_times = []
            
            for _ in range(num_runs):
                # Warm-up run
                with torch.no_grad():
                    _ = model(x)
                
                # Reset memory stats if using CUDA
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()
                
                # Timed run
                start_time = time.time()
                with torch.no_grad():
                    _ = model(x)
                run_times.append(time.time() - start_time)
            
            # Record the minimum time
            time_results[name].append(min(run_times))
            
            # Record memory usage if using CUDA
            if device.type == "cuda":
                memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                memory_results[name].append(memory_usage)
    
    return time_results, memory_results


def plot_results(sequence_lengths, time_results, memory_results=None, save_path=None):
    """
    Plot benchmark results.
    
    Args:
        sequence_lengths: List of sequence lengths
        time_results: Dictionary mapping model names to time measurements
        memory_results: Dictionary mapping model names to memory measurements
        save_path: Path to save the plot
    """
    # Determine if we plot memory results
    plot_memory = memory_results is not None
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if plot_memory else 1, figsize=(15 if plot_memory else 8, 6))
    
    # Make axes indexable for a single subplot
    if not plot_memory:
        axes = [axes]
    
    # Plot time results
    ax_time = axes[0]
    for name, times in time_results.items():
        ax_time.plot(sequence_lengths, times, marker="o", label=name)
    
    # Set logarithmic scales
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    
    # Set labels
    ax_time.set_xlabel("Sequence Length")
    ax_time.set_ylabel("Inference Time (s)")
    ax_time.set_title("Time Complexity")
    ax_time.grid(True, alpha=0.3, which="both")
    ax_time.legend()
    
    # Plot memory results if available
    if plot_memory:
        ax_memory = axes[1]
        for name, memory in memory_results.items():
            ax_memory.plot(sequence_lengths, memory, marker="s", label=name)
        
        # Set logarithmic scale for x-axis
        ax_memory.set_xscale("log")
        
        # Set labels
        ax_memory.set_xlabel("Sequence Length")
        ax_memory.set_ylabel("Memory Usage (MB)")
        ax_memory.set_title("Memory Complexity")
        ax_memory.grid(True, alpha=0.3, which="both")
        ax_memory.legend()
    
    # Add overall title
    plt.suptitle("Attention Mechanism Complexity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def calculate_complexity_factors(sequence_lengths, time_results):
    """
    Calculate empirical complexity factors for each model.
    
    Args:
        sequence_lengths: List of sequence lengths
        time_results: Dictionary mapping model names to time measurements
        
    Returns:
        Dictionary mapping model names to complexity exponents
    """
    complexity_factors = {}
    
    # For each model, fit a power law to estimate the complexity exponent
    for name, times in time_results.items():
        # Convert to numpy arrays
        x = np.array(sequence_lengths)
        y = np.array(times)
        
        # Fit a power law (linear in log-log space)
        # log(y) = alpha * log(x) + beta
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Simple linear regression
        alpha, beta = np.polyfit(log_x, log_y, 1)
        
        # Alpha is the complexity exponent
        complexity_factors[name] = alpha
    
    return complexity_factors


def print_summary(
    sequence_lengths, time_results, memory_results, complexity_factors
):
    """
    Print a summary of the benchmark results.
    
    Args:
        sequence_lengths: List of sequence lengths
        time_results: Dictionary mapping model names to time measurements
        memory_results: Dictionary mapping model names to memory measurements
        complexity_factors: Dictionary mapping model names to complexity exponents
    """
    # Print header
    print("\nBenchmark Summary:")
    print("-" * 80)
    
    # Print complexity factors
    print("Empirical Complexity Exponents:")
    for name, factor in complexity_factors.items():
        order = "O(n^{:.2f})".format(factor)
        print(f"  {name:20s}: {factor:.2f}  {order}")
    
    print("\nInference Time (ms):")
    print(f"{'Sequence Length':<15} | " + " | ".join(f"{name:<20}" for name in time_results.keys()))
    print("-" * 80)
    
    for i, seq_len in enumerate(sequence_lengths):
        times = [time_results[name][i] * 1000 for name in time_results.keys()]  # Convert to ms
        print(f"{seq_len:<15} | " + " | ".join(f"{t:>8.2f} ms        " for t in times))
    
    # Print memory usage if available
    if memory_results:
        print("\nMemory Usage (MB):")
        print(f"{'Sequence Length':<15} | " + " | ".join(f"{name:<20}" for name in memory_results.keys()))
        print("-" * 80)
        
        for i, seq_len in enumerate(sequence_lengths):
            memory = [memory_results[name][i] for name in memory_results.keys()]
            print(f"{seq_len:<15} | " + " | ".join(f"{m:>8.2f} MB        " for m in memory))


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    models = create_models(args)
    print(f"Benchmarking models: {list(models.keys())}")
    
    # Generate sequence lengths
    sequence_lengths = generate_sequence_lengths(
        args.min_seq_len, args.max_seq_len, args.num_steps
    )
    print(f"Testing sequence lengths: {sequence_lengths}")
    
    # Run benchmark
    time_results, memory_results = benchmark_models(
        models,
        sequence_lengths,
        args.hidden_size,
        device,
        args.num_runs,
        args.verbose
    )
    
    # Calculate complexity factors
    complexity_factors = calculate_complexity_factors(sequence_lengths, time_results)
    
    # Print summary
    print_summary(sequence_lengths, time_results, memory_results, complexity_factors)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = output_dir / f"benchmark_{timestamp}.json"
    
    with open(result_path, "w") as f:
        json.dump({
            "args": vars(args),
            "sequence_lengths": sequence_lengths,
            "time_results": time_results,
            "memory_results": memory_results,
            "complexity_factors": complexity_factors
        }, f, indent=2)
    
    print(f"Results saved to {result_path}")
    
    # Plot results
    plot_path = output_dir / f"benchmark_{timestamp}.png"
    plot_results(sequence_lengths, time_results, memory_results, plot_path)


if __name__ == "__main__":
    main()
