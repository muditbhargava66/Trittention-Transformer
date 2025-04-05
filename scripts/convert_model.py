"""
Script to convert between different attention mechanism models.

This script helps convert a model from one attention mechanism to another,
allowing for easy experimentation and comparison between different mechanisms.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import numpy as np

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
from models.lightning_module import TrittentionLightningModule, TrittentionBaseModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert between attention mechanism models")
    
    parser.add_argument("--input_model", type=str, required=True,
                        help="Path to input model checkpoint")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save converted model")
    parser.add_argument("--source_type", type=str, required=True,
                        choices=["standard", "trittention", "trittention_cube", 
                                "local", "mixed", "sparse", "windowed"],
                        help="Source attention mechanism type")
    parser.add_argument("--target_type", type=str, required=True,
                        choices=["standard", "trittention", "trittention_cube", 
                                "local", "mixed", "sparse", "windowed"],
                        help="Target attention mechanism type")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model configuration (JSON)")
    parser.add_argument("--hidden_size", type=int, default=None,
                        help="Hidden size (required if config_path not provided)")
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Number of attention heads (required if config_path not provided)")
    parser.add_argument("--window_size", type=int, default=128,
                        help="Window size for local attention mechanisms")
    parser.add_argument("--input_size", type=int, default=None,
                        help="Input feature size")
    parser.add_argument("--output_size", type=int, default=None,
                        help="Output feature size")
    parser.add_argument("--init_weights", action="store_true",
                        help="Initialize weights of the new attention mechanism")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    return parser.parse_args()


def load_configuration(args) -> TrittentionConfig:
    """
    Load or create configuration for the model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration object
    """
    # Load from file if provided
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config from dictionary
        config = TrittentionConfig(
            hidden_size=config_dict.get('hidden_size', 768),
            num_attention_heads=config_dict.get('num_attention_heads', 12),
            attention_probs_dropout_prob=config_dict.get('attention_probs_dropout_prob', 0.1),
            hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
            window_size=config_dict.get('window_size', args.window_size)
        )
        
        # Set additional attributes
        for key, value in config_dict.items():
            if not hasattr(config, key):
                setattr(config, key, value)
    else:
        # Validate required parameters
        if args.hidden_size is None or args.num_heads is None:
            raise ValueError("Either config_path or hidden_size and num_heads must be provided")
        
        # Create config from arguments
        config = TrittentionConfig(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads,
            window_size=args.window_size
        )
    
    # Set use flags based on target type
    config.use_trittention = args.target_type in ["trittention", "trittention_cube"]
    config.use_local_trittention = args.target_type == "local"
    config.use_mixed_attention = args.target_type == "mixed"
    
    # Set sparsity threshold for sparse trittention
    if args.target_type == "sparse":
        setattr(config, 'sparsity_threshold', 0.1)
    
    return config


def create_attention_mechanism(attention_type: str, config: TrittentionConfig) -> torch.nn.Module:
    """
    Create an attention mechanism by type.
    
    Args:
        attention_type: Type of attention mechanism
        config: Configuration object
        
    Returns:
        Attention mechanism module
    """
    if attention_type == "standard":
        return Attention(config)
    elif attention_type == "trittention":
        return Trittention(config)
    elif attention_type == "trittention_cube":
        return TrittentionCube(config)
    elif attention_type == "local":
        # Ensure local trittention flag is set
        config_local = TrittentionConfig(**vars(config))
        config_local.use_local_trittention = True
        return LocalTrittention(config_local)
    elif attention_type == "mixed":
        # Ensure mixed attention flag is set
        config_mixed = TrittentionConfig(**vars(config))
        config_mixed.use_mixed_attention = True
        return MixedAttention(config_mixed)
    elif attention_type == "sparse":
        return SparseTrittention(config)
    elif attention_type == "windowed":
        return WindowedTrittention(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def load_model(
    model_path: str, 
    config: TrittentionConfig, 
    attention_type: str,
    input_size: Optional[int] = None,
    output_size: Optional[int] = None
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    """
    Load model from file.
    
    Args:
        model_path: Path to model file
        config: Configuration object
        attention_type: Type of attention mechanism
        input_size: Input feature size
        output_size: Output feature size
        
    Returns:
        Tuple of (model, state_dict)
    """
    # Try to load as a complete model first
    try:
        model = torch.load(model_path)
        if isinstance(model, torch.nn.Module):
            return model, model.state_dict()
    except Exception as e:
        if "verbose" in vars(args) and args.verbose:
            print(f"Could not load as complete model: {e}")
    
    # Try to load as state dictionary
    try:
        state_dict = torch.load(model_path)
        if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
            # Get input and output sizes from state_dict if possible
            if input_size is None or output_size is None:
                for key, value in state_dict.items():
                    if "input_projection.weight" in key:
                        input_size = value.size(1)
                    if "output_projection.weight" in key:
                        output_size = value.size(0)
            
            # Validate input and output sizes
            if input_size is None or output_size is None:
                raise ValueError("Could not determine input or output size from model. "
                                "Please specify input_size and output_size arguments.")
            
            # Create model
            attention_module = create_attention_mechanism(attention_type, config)
            model = TrittentionBaseModel(
                attention_mechanism=attention_module,
                input_size=input_size,
                hidden_size=config.hidden_size,
                output_size=output_size,
                dropout_prob=config.hidden_dropout_prob
            )
            
            # Filter state dict to only include matching keys
            filtered_state_dict = {}
            for key, value in state_dict.items():
                try:
                    if key in model.state_dict():
                        model_param = model.state_dict()[key]
                        if value.size() == model_param.size():
                            filtered_state_dict[key] = value
                    elif "module." + key in model.state_dict():
                        model_param = model.state_dict()["module." + key]
                        if value.size() == model_param.size():
                            filtered_state_dict["module." + key] = value
                except Exception as e:
                    if "verbose" in vars(args) and args.verbose:
                        print(f"Could not load parameter {key}: {e}")
            
            # Load state dict
            model.load_state_dict(filtered_state_dict, strict=False)
            
            return model, state_dict
    except Exception as e:
        if "verbose" in vars(args) and args.verbose:
            print(f"Could not load as state dictionary: {e}")
    
    raise ValueError(f"Could not load model from {model_path}")


def convert_model(args):
    """
    Convert a model from one attention mechanism to another.
    
    Args:
        args: Command line arguments
    """
    # Load or create configuration
    config = load_configuration(args)
    
    # Print configuration
    if args.verbose:
        print("\nModel Configuration:")
        print("-" * 40)
        for key, value in vars(config).items():
            print(f"{key}: {value}")
    
    # Load source model
    print(f"\nLoading source model from {args.input_model}...")
    source_model, source_state_dict = load_model(
        args.input_model,
        config,
        args.source_type,
        args.input_size,
        args.output_size
    )
    
    # Extract input and output sizes
    if isinstance(source_model, TrittentionBaseModel):
        input_size = source_model.input_projection.in_features
        output_size = source_model.output_projection.out_features
    elif hasattr(source_model, "model") and isinstance(source_model.model, TrittentionBaseModel):
        input_size = source_model.model.input_projection.in_features
        output_size = source_model.model.output_projection.out_features
    elif args.input_size is not None and args.output_size is not None:
        input_size = args.input_size
        output_size = args.output_size
    else:
        raise ValueError("Could not determine input or output size from model. "
                        "Please specify input_size and output_size arguments.")
    
    print(f"Detected input size: {input_size}, output size: {output_size}")
    
    # Create target attention mechanism
    target_attention = create_attention_mechanism(args.target_type, config)
    
    # Create target model
    if isinstance(source_model, TrittentionBaseModel):
        # Create new base model
        target_model = TrittentionBaseModel(
            attention_mechanism=target_attention,
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=output_size,
            dropout_prob=config.hidden_dropout_prob
        )
        
        # Copy non-attention weights from source model
        target_dict = target_model.state_dict()
        for key in target_dict.keys():
            if "attention" not in key and key in source_state_dict:
                target_dict[key] = source_state_dict[key]
        
        # Initialize with updated weights
        target_model.load_state_dict(target_dict, strict=False)
    
    elif isinstance(source_model, TrittentionLightningModule):
        # Extract the attention mechanism
        source_attention = source_model.model.attention
        
        # Create new lightning module
        target_model = TrittentionLightningModule(
            attention_type=args.target_type,
            config=config,
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=output_size,
            learning_rate=getattr(source_model, "learning_rate", 1e-3),
            weight_decay=getattr(source_model, "weight_decay", 0.01),
            optimizer_type=getattr(source_model, "optimizer_type", "adam"),
            scheduler_type=getattr(source_model, "scheduler_type", None)
        )
        
        # Copy non-attention weights from source model
        target_dict = target_model.state_dict()
        source_dict = source_model.state_dict()
        
        for key in target_dict.keys():
            if "attention" not in key and key in source_dict:
                target_dict[key] = source_dict[key]
        
        # Initialize with updated weights
        target_model.load_state_dict(target_dict, strict=False)
    
    else:
        # For unknown model types, try creating a base model
        target_model = TrittentionBaseModel(
            attention_mechanism=target_attention,
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=output_size,
            dropout_prob=config.hidden_dropout_prob
        )
    
    # Save target model
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as state dictionary
    torch.save(target_model.state_dict(), output_path)
    print(f"\nConverted model saved to {output_path}")
    
    # Print parameter counts
    source_params = sum(p.numel() for p in source_model.parameters())
    target_params = sum(p.numel() for p in target_model.parameters())
    
    print(f"\nSource model ({args.source_type}): {source_params:,} parameters")
    print(f"Target model ({args.target_type}): {target_params:,} parameters")
    
    # Save configuration
    config_path = output_path.with_suffix('.json')
    with open(config_path, 'w') as f:
        config_dict = {
            'model_type': args.target_type,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'input_size': input_size,
            'output_size': output_size,
            'window_size': config.window_size
        }
        json.dump(config_dict, f, indent=2)
    
    print(f"Model configuration saved to {config_path}")


def main():
    """Main function."""
    global args
    args = parse_args()
    
    print(f"\nConverting model from {args.source_type} to {args.target_type}")
    print(f"Source model: {args.input_model}")
    print(f"Target model: {args.output_model}")
    
    # Convert model
    convert_model(args)


if __name__ == "__main__":
    main()
