"""
Benchmark script for comparing different attention mechanisms.

This script evaluates and compares the computational efficiency and performance
of various attention mechanisms across different sequence lengths.
"""

import os
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from utils.evaluation_utils import benchmark_complexity
from utils.visualization_utils import plot_complexity_analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark different attention mechanisms")
    
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for the attention mechanisms")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--min_seq_len", type=int, default=10,
                        help="Minimum sequence length to test")
    parser.add_argument("--max_seq_len", type=int, default=1000,
                        help="Maximum sequence length to test")
    parser.add_argument("--num_steps", type=int, default=6,
                        help="Number of sequence length steps")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the benchmark")
    parser.add_argument("--window_size", type=int, default=128,
                        help="Window size for windowed attention mechanisms")
    parser.add_argument("--sparsity", type=float, default=0.1,
                        help="Sparsity threshold for sparse attention")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for benchmarking if available")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["standard", "trittention", "sparse", "windowed"],
                        help="Models to benchmark")
    
    args = parser.parse_args()
    return args


def create_attention_models(args) -> Dict[str, torch.nn.Module]:
    """
    Create attention mechanisms according to the provided arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary mapping model names to model instances
    """
    # Base configuration for all models
    config = TrittentionConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        window_size=args.window_size,
        attention_probs_dropout_prob=0.0  # Disable dropout for benchmarking
    )
    
    # Create models dictionary
    models = {}
    
    # Add requested models
    for model_name in args.models:
        if model_name.lower() == "standard":
            models["Standard Attention"] = Attention(config)
        elif model_name.lower() == "trittention":
            models["Trittention"] = Trittention(config)
        elif model_name.lower() == "cube":
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
            # Set sparsity threshold
            config.sparsity_threshold = args.sparsity
            models["Sparse Trittention"] = SparseTrittention(config)
        elif model_name.lower() == "windowed":
            models["Windowed Trittention"] = WindowedTrittention(config)
        else:
            print(f"Warning: Unknown model '{model_name}', skipping")
    
    # Set all models to eval mode
    for model in models.values():
        model.eval()
    
    return models


def generate_sequence_lengths(min_len: int, max_len: int, num_steps: int) -> List[int]:
    """
    Generate a list of sequence lengths for benchmarking.
    
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


def benchmark_attention_mechanisms(
    models: Dict[str, torch.nn.Module],
    sequence_lengths: List[int],
    hidden_size: int,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
    num_runs: int = 3
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Benchmark attention mechanisms across different sequence lengths.
    
    Args:
        models: Dictionary mapping model names to model instances
        sequence_lengths: List of sequence lengths to test
        hidden_size: Hidden size dimension
        batch_size: Batch size for testing
        device: Device to run the benchmark on
        num_runs: Number of runs for each configuration
        
    Returns:
        Tuple of dictionaries with timing and memory results
    """
    # Dictionaries to store results
    time_results = {name: [] for name in models.keys()}
    memory_results = {name: [] for name in models.keys()} if device.type == "cuda" else None
    
    # Move models to device
    for model in models.values():
        model.to(device)
    
    # Run benchmarks for each sequence length
    for seq_len in sequence_lengths:
        print(f"Benchmarking sequence length: {seq_len}")
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Benchmark each model
        for name, model in models.items():
            print(f"  - Testing {name}...")
            
            # Run multiple times and take the minimum
            run_times = []
            
            for _ in range(num_runs):
                # Warm-up run
                with torch.no_grad():
                    _ = model(x)
                
                # Reset CUDA memory stats if using GPU
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()
                
                # Timed run
                start_time = time.time()
                with torch.no_grad():
                    _ = model(x)
                run_times.append(time.time() - start_time)
            
            # Record the minimum time (to minimize impact of system variability)
            time_results[name].append(min(run_times))
            
            # Record memory usage if using CUDA
            if device.type == "cuda":
                memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                memory_results[name].append(memory_usage)
    
    return time_results, memory_results


def save_results(
    time_results: Dict[str, List[float]],
    memory_results: Dict[str, List[float]],
    sequence_lengths: List[int],
    output_dir: str,
    args_dict: Dict
) -> Tuple[str, str]:
    """
    Save benchmark results to disk.
    
    Args:
        time_results: Dictionary mapping model names to time measurements
        memory_results: Dictionary mapping model names to memory measurements
        sequence_lengths: List of sequence lengths
        output_dir: Directory to save results
        args_dict: Dictionary of benchmark arguments
        
    Returns:
        Tuple of paths to the saved files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create result dictionary
    results = {
        "timestamp": timestamp,
        "parameters": args_dict,
        "sequence_lengths": sequence_lengths,
        "time_results": time_results,
        "memory_results": memory_results
    }
    
    # Save results as JSON
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    plot_path = output_dir / f"benchmark_{timestamp}.png"
    plot_complexity_analysis(
        sequence_lengths=sequence_lengths,
        time_complexities=time_results,
        memory_complexities=memory_results,
        title="Attention Mechanisms Complexity Analysis",
        save_path=plot_path,
        show=False
    )
    
    return str(json_path), str(plot_path)


def main():
    """Main function to run the benchmark."""
    # Parse arguments
    args = parse_args()
    
    # Determine device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate sequence lengths
    sequence_lengths = generate_sequence_lengths(
        args.min_seq_len, args.max_seq_len, args.num_steps
    )
    print(f"Testing sequence lengths: {sequence_lengths}")
    
    # Create models
    models = create_attention_models(args)
    print(f"Benchmarking models: {list(models.keys())}")
    
    # Run benchmarks
    time_results, memory_results = benchmark_attention_mechanisms(
        models,
        sequence_lengths,
        args.hidden_size,
        args.batch_size,
        device
    )
    
    # Save results
    args_dict = vars(args)
    json_path, plot_path = save_results(
        time_results,
        memory_results,
        sequence_lengths,
        args.output_dir,
        args_dict
    )
    
    print(f"Results saved to {json_path}")
    print(f"Plot saved to {plot_path}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 60)
    print(f"{'Sequence Length':<15} | " + " | ".join(f"{name:<20}" for name in models.keys()))
    print("-" * 60)
    
    for i, seq_len in enumerate(sequence_lengths):
        times = [time_results[name][i] for name in models.keys()]
        print(f"{seq_len:<15} | " + " | ".join(f"{t*1000:>8.2f} ms        " for t in times))


if __name__ == "__main__":
    main()
