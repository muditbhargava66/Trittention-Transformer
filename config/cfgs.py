# config/cfgs.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrittentionConfig:
    """Configuration class for Trittention models."""
    
    hidden_size: int = 768
    """The size of the hidden states in the model."""
    
    num_attention_heads: int = 12
    """The number of attention heads in the model."""
    
    attention_probs_dropout_prob: float = 0.1
    """The dropout probability for attention scores."""
    
    hidden_dropout_prob: float = 0.1
    """The dropout probability for hidden states."""
    
    intermediate_size: int = 3072
    """The size of the intermediate layer in the feed-forward network."""
    
    num_hidden_layers: int = 12
    """The number of hidden layers in the model."""
    
    max_position_embeddings: int = 512
    """The maximum sequence length that the model can handle."""
    
    type_vocab_size: int = 2
    """The vocabulary size of the token types."""
    
    initializer_range: float = 0.02
    """The standard deviation of the truncated normal initializer."""
    
    layer_norm_eps: float = 1e-12
    """The epsilon used by the layer normalization layers."""
    
    use_trittention: bool = True
    """Whether to use trittention or standard attention."""
    
    window_size: int = 128
    """The size of the local attention window in trittention."""
    
    use_local_trittention: bool = False
    """Whether to use local trittention or global trittention."""
    
    use_mixed_attention: bool = False
    """Whether to use mixed attention (combination of global and local attention)."""
    
    num_local_heads: int = 4
    """The number of local attention heads in mixed attention."""
    
    def __post_init__(self):
        """Validate and finalize the configuration."""
        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by the number of attention heads."
        assert self.window_size <= self.max_position_embeddings, "Window size cannot exceed the maximum sequence length."
        
        if self.use_mixed_attention:
            assert self.num_local_heads <= self.num_attention_heads, "Number of local attention heads cannot exceed the total number of attention heads."
            self.num_global_heads = self.num_attention_heads - self.num_local_heads
        else:
            self.num_global_heads = self.num_attention_heads
            self.num_local_heads = 0
        
        self.attention_head_size = self.hidden_size // self.num_attention_heads
