"""
Script to visualize attention patterns from different attention mechanisms.

This script provides a simple way to visualize and compare attention patterns
from different attention mechanisms included in the Trittention-Transformer project.
"""

import os
import sys
import argparse
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
from utils.visualization_utils import (
    visualize_attention_matrix,
    visualize_attention_comparisons,
    create_attention_map_for_text
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize attention patterns")
    
    parser.add_argument("--models", type=str, nargs="+",
                        default=["standard", "trittention", "trittention_cube", "sparse", "windowed"],
                        help="Attention models to visualize")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden size for models")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=16,
                        help="Window size for local attention mechanisms")
    parser.add_argument("--seq_length", type=int, default=20,
                        help="Sequence length for visualization")
    parser.add_argument("--mode", type=str, default="pattern",
                        choices=["pattern", "text", "both"],
                        help="Visualization mode")
    parser.add_argument("--text", type=str,
                        default="The quick brown fox jumps over the lazy dog.",
                        help="Text to visualize attention for (in text mode)")
    parser.add_argument("--output_dir", type=str, default="./results/visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--save", action="store_true",
                        help="Save visualizations instead of displaying")
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
        attention_probs_dropout_prob=0.0,  # Disable dropout for visualization
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
    
    # Set all models to eval mode
    for model in models.values():
        model.eval()
    
    return models


def generate_structured_input(seq_length, hidden_size, batch_size=1):
    """
    Generate a structured input to showcase distinct attention patterns.
    
    Args:
        seq_length: Sequence length
        hidden_size: Hidden size
        batch_size: Batch size
        
    Returns:
        Tensor of shape [batch_size, seq_length, hidden_size]
    """
    # Create a zero tensor
    hidden_states = torch.zeros(batch_size, seq_length, hidden_size)
    
    # Create a structured pattern with 4 different token groups
    for i in range(4):
        # Base pattern for this group
        group_pattern = torch.randn(1, 1, hidden_size)
        
        # Place the pattern at every 4th position, starting from i
        for j in range(i, seq_length, 4):
            # Add the group pattern plus some noise
            hidden_states[:, j, :] = group_pattern + 0.3 * torch.randn(1, hidden_size)
    
    return hidden_states


def get_attention_patterns(models, hidden_states):
    """
    Extract attention patterns from models.
    
    Args:
        models: Dictionary mapping model names to model instances
        hidden_states: Input tensor
        
    Returns:
        Dictionary mapping model names to attention patterns
    """
    attention_patterns = {}
    
    for name, model in models.items():
        # Forward pass
        with torch.no_grad():
            # Try different methods to extract attention weights
            if hasattr(model, 'forward') and 'output_attentions' in model.forward.__code__.co_varnames:
                # Model supports returning attention weights directly
                _, attn_weights = model(hidden_states, output_attentions=True)
            else:
                # Run forward pass and try to extract weights afterwards
                _ = model(hidden_states)
                if hasattr(model, 'get_attention_weights'):
                    attn_weights = model.get_attention_weights()
                elif hasattr(model, 'attention_weights'):
                    attn_weights = model.attention_weights
                else:
                    print(f"Warning: Could not extract attention patterns for {name}")
                    continue
            
            # For multi-head attention, average across heads
            if attn_weights.dim() > 3:  # [batch, heads, seq_len, seq_len]
                attn_weights = attn_weights.mean(dim=1)
            
            # Extract the first batch item
            attention_patterns[name] = attn_weights[0].cpu().numpy()
    
    return attention_patterns


def tokenize_text(text):
    """
    Simple tokenization of text by splitting on spaces.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    # Basic tokenization by splitting on spaces and punctuation
    tokens = []
    current_token = ""
    
    for char in text:
        if char.isalnum() or char == "'":  # Keep alphanumeric chars and apostrophes
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if not char.isspace():  # Add punctuation as separate tokens
                tokens.append(char)
    
    # Add the last token if it exists
    if current_token:
        tokens.append(current_token)
    
    return tokens


def text_to_tensor(tokens, hidden_size):
    """
    Convert tokens to a tensor representation.
    
    Args:
        tokens: List of tokens
        hidden_size: Hidden size dimension
        
    Returns:
        Tensor of shape [1, len(tokens), hidden_size]
    """
    # Create a simple embedding
    seq_length = len(tokens)
    embedding = torch.nn.Embedding(128, hidden_size)  # Simple ASCII embedding
    
    # Convert tokens to indices (using ASCII values)
    indices = [ord(token[0]) % 128 if token else 0 for token in tokens]
    token_ids = torch.tensor(indices)
    
    # Get embeddings with batch dimension
    return embedding(token_ids).unsqueeze(0)


def visualize_pattern_mode(models, args):
    """
    Visualize attention patterns using structured input.
    
    Args:
        models: Dictionary mapping model names to model instances
        args: Command line arguments
    """
    # Generate structured input
    hidden_states = generate_structured_input(
        args.seq_length, args.hidden_size
    )
    
    # Get attention patterns
    attention_patterns = get_attention_patterns(models, hidden_states)
    
    # Create output directory if saving
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print(f"Generating visualizations for {len(attention_patterns)} models...")
    
    # Individual visualizations
    for name, pattern in attention_patterns.items():
        save_path = Path(args.output_dir) / f"{name.lower().replace(' ', '_')}_pattern.png" if args.save else None
        
        fig = visualize_attention_matrix(
            attention_matrix=pattern,
            title=f"{name} Attention Pattern",
            save_path=save_path,
            show=not args.save,
            figsize=(10, 8)
        )
    
    # Comparison visualization
    save_path = Path(args.output_dir) / "attention_comparison.png" if args.save else None
    
    fig = visualize_attention_comparisons(
        attention_matrices=attention_patterns,
        title="Comparison of Attention Mechanisms",
        save_path=save_path,
        show=not args.save,
        figsize=(5 * min(len(attention_patterns), 3), 4 * ((len(attention_patterns) + 2) // 3))
    )
    
    if args.save:
        print(f"Visualizations saved to {args.output_dir}")


def visualize_text_mode(models, args):
    """
    Visualize attention patterns for text.
    
    Args:
        models: Dictionary mapping model names to model instances
        args: Command line arguments
    """
    # Tokenize text
    tokens = tokenize_text(args.text)
    print(f"Tokenized text: {tokens}")
    
    # Convert to tensor
    text_tensor = text_to_tensor(tokens, args.hidden_size)
    
    # Get attention patterns
    attention_patterns = get_attention_patterns(models, text_tensor)
    
    # Create output directory if saving
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print(f"Generating text visualizations for {len(attention_patterns)} models...")
    
    for name, pattern in attention_patterns.items():
        save_path = Path(args.output_dir) / f"{name.lower().replace(' ', '_')}_text.png" if args.save else None
        
        create_attention_map_for_text(
            text=tokens,
            attention_matrix=pattern,
            title=f"{name} Attention for Text",
            save_path=save_path,
            show=not args.save,
            figsize=(12, 10)
        )
    
    if args.save:
        print(f"Text visualizations saved to {args.output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create models
    models = create_models(args)
    
    # Run visualizations based on mode
    if args.mode in ["pattern", "both"]:
        visualize_pattern_mode(models, args)
    
    if args.mode in ["text", "both"]:
        visualize_text_mode(models, args)


if __name__ == "__main__":
    main()
