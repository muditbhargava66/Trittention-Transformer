"""
Script to summarize model architecture and parameters.

This script provides a detailed summary of model architecture, parameter counts,
and computational complexity for different attention mechanisms.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import numpy as np
import pandas as pd
from torch.nn.modules.module import _addindent

# Add parent directory to Python path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.append(str(project_dir))

from config.cfgs import TrittentionConfig
from config.sample_config import get_default_config
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
    parser = argparse.ArgumentParser(description="Summarize model architecture and parameters")
    
    parser.add_argument("--models", type=str, nargs="+",
                        default=["standard", "trittention", "trittention_cube", 
                                "local", "mixed", "sparse", "windowed"],
                        help="Attention models to summarize")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size for models")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=128,
                        help="Window size for local attention mechanisms")
    parser.add_argument("--config_preset", type=str, default=None,
                        choices=["default", "small", "large", "long_sequence"],
                        help="Use a preset configuration")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length for complexity analysis")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save summary (CSV format)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    return parser.parse_args()


def create_config(args) -> TrittentionConfig:
    """
    Create configuration based on arguments or preset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration object
    """
    from config.sample_config import (
        get_default_config,
        get_small_model_config,
        get_long_sequence_config
    )
    
    # Use preset if specified
    if args.config_preset == "default":
        config = get_default_config()
    elif args.config_preset == "small":
        config = get_small_model_config()
    elif args.config_preset == "long_sequence":
        config = get_long_sequence_config()
    else:
        # Create custom configuration
        config = TrittentionConfig(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads,
            window_size=args.window_size,
            max_position_embeddings=max(args.seq_length, 512)
        )
        
        # Set sparsity threshold for sparse trittention
        setattr(config, 'sparsity_threshold', 0.1)
    
    return config


def create_models(args, config: TrittentionConfig) -> Dict[str, torch.nn.Module]:
    """
    Create attention models according to arguments.
    
    Args:
        args: Command line arguments
        config: Configuration object
        
    Returns:
        Dictionary mapping model names to model instances
    """
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
            config_local = TrittentionConfig(**vars(config))
            config_local.use_local_trittention = True
            models["Local Trittention"] = LocalTrittention(config_local)
        elif model_name.lower() == "mixed":
            config_mixed = TrittentionConfig(**vars(config))
            config_mixed.use_mixed_attention = True
            models["Mixed Attention"] = MixedAttention(config_mixed)
        elif model_name.lower() == "sparse":
            models["Sparse Trittention"] = SparseTrittention(config)
        elif model_name.lower() == "windowed":
            models["Windowed Trittention"] = WindowedTrittention(config)
        else:
            print(f"Warning: Unknown model '{model_name}', skipping")
    
    return models


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: torch.nn.Module) -> str:
    """
    Get a string summary of a model's architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        String summary of model architecture
    """
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        
        child_lines = []
        for key, module in model._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        
        lines = extra_lines + child_lines
        
        main_str = model._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        
        main_str += ")"
        return main_str
    
    return repr(model)


def estimate_complexity(model_name: str, hidden_size: int, num_heads: int, seq_length: int) -> Dict[str, Any]:
    """
    Estimate computational complexity for different models.
    
    Args:
        model_name: Name of the model
        hidden_size: Hidden size dimension
        num_heads: Number of attention heads
        seq_length: Sequence length
        
    Returns:
        Dictionary with complexity estimates
    """
    # Basic attention complexity
    if "Standard" in model_name:
        # O(N^2 * D) for attention
        flops = 2 * seq_length ** 2 * hidden_size
        theoretical_complexity = "O(n²)"
        memory = seq_length ** 2 * 4  # 4 bytes per float
    elif "Trittention" in model_name and "Cube" in model_name:
        # O(N^3 * D) for cubic trittention
        flops = 3 * seq_length ** 3 * hidden_size
        theoretical_complexity = "O(n³)"
        memory = 2 * seq_length ** 2 * 4  # Additional memory for cubic attention
    elif "Trittention" in model_name:
        # O(N^3 * D) for standard trittention
        flops = 2 * seq_length ** 3 * hidden_size
        theoretical_complexity = "O(n³)"
        memory = seq_length ** 2 * 4
    elif "Local" in model_name:
        # O(N * W^2 * D) for local attention with window size W
        window_size = min(128, seq_length)  # Default window size
        flops = 2 * seq_length * window_size ** 2 * hidden_size
        theoretical_complexity = "O(n * w²)"
        memory = seq_length * window_size * 4
    elif "Mixed" in model_name:
        # Mix of global and local attention
        global_heads = 3 * num_heads // 4  # Default: 75% global, 25% local
        local_heads = num_heads - global_heads
        window_size = min(128, seq_length)
        
        global_flops = 2 * global_heads * seq_length ** 2 * (hidden_size // num_heads)
        local_flops = 2 * local_heads * seq_length * window_size ** 2 * (hidden_size // num_heads)
        flops = global_flops + local_flops
        
        theoretical_complexity = "O(mix of n² and n*w²)"
        memory = (global_heads * seq_length ** 2 + local_heads * seq_length * window_size) * 4
    elif "Sparse" in model_name:
        # Sparse attention assumes 90% of connections are pruned
        sparsity = 0.9
        flops = 2 * (1 - sparsity) * seq_length ** 3 * hidden_size
        theoretical_complexity = "O(s*n³) where s is sparsity factor"
        memory = (1 - sparsity) * seq_length ** 2 * 4
    elif "Windowed" in model_name:
        # Windowed attention focuses on local windows
        window_size = min(128, seq_length)
        flops = 2 * seq_length * window_size ** 2 * hidden_size
        theoretical_complexity = "O(n * w²)"
        memory = seq_length * window_size * 4
    else:
        # Default case
        flops = 2 * seq_length ** 2 * hidden_size
        theoretical_complexity = "Unknown"
        memory = seq_length ** 2 * 4
    
    # Convert to more human-readable units
    if flops > 1e12:
        flops_str = f"{flops / 1e12:.2f} TFLOPs"
    elif flops > 1e9:
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    elif flops > 1e6:
        flops_str = f"{flops / 1e6:.2f} MFLOPs"
    else:
        flops_str = f"{flops:.2f} FLOPs"
    
    if memory > 1e9:
        memory_str = f"{memory / 1e9:.2f} GB"
    elif memory > 1e6:
        memory_str = f"{memory / 1e6:.2f} MB"
    elif memory > 1e3:
        memory_str = f"{memory / 1e3:.2f} KB"
    else:
        memory_str = f"{memory:.2f} bytes"
    
    return {
        "theoretical_complexity": theoretical_complexity,
        "estimated_flops": flops,
        "estimated_flops_str": flops_str,
        "estimated_memory": memory,
        "estimated_memory_str": memory_str
    }


def summarize_models(models: Dict[str, torch.nn.Module], seq_length: int, verbose: bool = False) -> pd.DataFrame:
    """
    Create a summary of model architectures and complexities.
    
    Args:
        models: Dictionary mapping model names to model instances
        seq_length: Sequence length for complexity analysis
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with model summaries
    """
    # Create summary data
    summary_data = []
    
    for name, model in models.items():
        # Get model details
        hidden_size = getattr(model, "hidden_size", 0)
        num_heads = getattr(model, "num_attention_heads", 0)
        
        # Count parameters
        param_count = count_parameters(model)
        
        # Estimate complexity
        complexity = estimate_complexity(name, hidden_size, num_heads, seq_length)
        
        # Print detailed information if requested
        if verbose:
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"Parameter count: {param_count:,}")
            print(f"Theoretical complexity: {complexity['theoretical_complexity']}")
            print(f"Estimated FLOPs: {complexity['estimated_flops_str']}")
            print(f"Estimated memory: {complexity['estimated_memory_str']}")
            print(f"\nModel architecture:")
            print(f"{'-'*40}")
            print(get_model_summary(model))
        
        # Add to summary data
        summary_data.append({
            "Model": name,
            "Parameters": param_count,
            "Theoretical Complexity": complexity["theoretical_complexity"],
            "Estimated FLOPs": complexity["estimated_flops"],
            "Estimated FLOPs (str)": complexity["estimated_flops_str"],
            "Estimated Memory": complexity["estimated_memory"],
            "Estimated Memory (str)": complexity["estimated_memory_str"],
            "Hidden Size": hidden_size,
            "Attention Heads": num_heads
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    return df


def print_summary_table(df: pd.DataFrame):
    """
    Print a summary table of model information.
    
    Args:
        df: DataFrame with model summaries
    """
    # Print header
    print("\nModel Summary:")
    print("-" * 100)
    
    # Print formatted table
    headers = ["Model", "Parameters", "Complexity", "FLOPs", "Memory", "Hidden Size", "Heads"]
    row_format = "{:20s} | {:12s} | {:15s} | {:15s} | {:15s} | {:11s} | {:5s}"
    
    # Print header row
    print(row_format.format(*headers))
    print("-" * 100)
    
    # Print each row
    for _, row in df.iterrows():
        print(row_format.format(
            str(row["Model"]),
            f"{row['Parameters']:,}",
            str(row["Theoretical Complexity"]),
            str(row["Estimated FLOPs (str)"]),
            str(row["Estimated Memory (str)"]),
            str(row["Hidden Size"]),
            str(row["Attention Heads"])
        ))


def main():
    """Main function."""
    args = parse_args()
    
    # Create configuration
    config = create_config(args)
    
    # Print configuration
    print("\nModel Configuration:")
    print("-" * 40)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    
    # Create models
    models = create_models(args, config)
    
    # Summarize models
    summary_df = summarize_models(models, args.seq_length, args.verbose)
    
    # Print summary table
    print_summary_table(summary_df)
    
    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
