"""
Sample configuration file for Trittention-Transformer.

This file demonstrates different configurations for various attention mechanisms
and provides example setups for common use cases.
"""

from config.cfgs import TrittentionConfig

# Standard configuration for trittention
def get_default_config():
    """
    Get the default configuration for trittention.
    
    Returns:
        TrittentionConfig: Default configuration
    """
    return TrittentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        intermediate_size=3072,
        num_hidden_layers=12,
        max_position_embeddings=512
    )

# Configuration for standard attention (disable trittention)
def get_standard_attention_config():
    """
    Get configuration for standard attention (disabling trittention).
    
    Returns:
        TrittentionConfig: Configuration for standard attention
    """
    config = get_default_config()
    config.use_trittention = False
    return config

# Configuration for local trittention (with smaller window)
def get_local_trittention_config(window_size=128):
    """
    Get configuration for local trittention with specified window size.
    
    Args:
        window_size: Size of the local attention window
        
    Returns:
        TrittentionConfig: Configuration for local trittention
    """
    config = get_default_config()
    config.use_local_trittention = True
    config.window_size = window_size
    return config

# Configuration for mixed attention
def get_mixed_attention_config(num_local_heads=4, window_size=128):
    """
    Get configuration for mixed attention with specified local heads.
    
    Args:
        num_local_heads: Number of attention heads to use local attention
        window_size: Size of the local attention window
        
    Returns:
        TrittentionConfig: Configuration for mixed attention
    """
    config = get_default_config()
    config.use_mixed_attention = True
    config.num_local_heads = num_local_heads
    config.window_size = window_size
    return config

# Configuration for sparse trittention
def get_sparse_trittention_config(sparsity_threshold=0.01):
    """
    Get configuration for sparse trittention.
    
    Args:
        sparsity_threshold: Threshold for pruning attention scores
        
    Returns:
        TrittentionConfig: Configuration for sparse trittention
    """
    config = get_default_config()
    config.sparsity_threshold = sparsity_threshold
    return config

# Configuration optimized for longer sequences
def get_long_sequence_config(max_seq_length=8192):
    """
    Get configuration optimized for longer sequences.
    
    Args:
        max_seq_length: Maximum sequence length
        
    Returns:
        TrittentionConfig: Configuration for long sequences
    """
    config = get_default_config()
    config.max_position_embeddings = max_seq_length
    config.use_local_trittention = True
    config.window_size = 512
    config.sparsity_threshold = 0.05
    return config

# Configuration optimized for small models (mobile/edge)
def get_small_model_config():
    """
    Get configuration for small models suitable for mobile/edge deployment.
    
    Returns:
        TrittentionConfig: Configuration for small models
    """
    return TrittentionConfig(
        hidden_size=256,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        intermediate_size=1024,
        num_hidden_layers=4,
        max_position_embeddings=512,
        use_trittention=True,
        sparsity_threshold=0.1  # More aggressive sparsity for efficiency
    )

# Configuration for low-rank approximation
def get_lowrank_config(rank_factor=4):
    """
    Get configuration for low-rank approximation of trittention.
    
    Args:
        rank_factor: Factor to divide hidden size by for rank
        
    Returns:
        TrittentionConfig: Configuration with low-rank settings
    """
    config = get_default_config()
    config.use_low_rank = True
    config.rank = config.hidden_size // rank_factor
    return config

# Usage examples:
if __name__ == "__main__":
    # Print sample configurations
    configs = {
        "Default": get_default_config(),
        "Standard Attention": get_standard_attention_config(),
        "Local Trittention": get_local_trittention_config(),
        "Mixed Attention": get_mixed_attention_config(),
        "Sparse Trittention": get_sparse_trittention_config(),
        "Long Sequence": get_long_sequence_config(),
        "Small Model": get_small_model_config(),
        "Low-Rank": get_lowrank_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print("-" * 40)
        for key, value in vars(config).items():
            print(f"{key}: {value}")
