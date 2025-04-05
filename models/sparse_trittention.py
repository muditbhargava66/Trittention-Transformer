"""
Implementation of sparse trittention mechanism for transformer models.

This module provides an efficient implementation of the trittention mechanism
with sparsity optimizations to address the O(n³) computational complexity.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTrittention(nn.Module):
    """
    Sparse implementation of the tri-attention mechanism with reduced complexity.
    
    This implementation applies sparsity techniques to reduce the computational
    complexity of the trittention mechanism from O(n³) to approximately O(n²).
    
    Attributes:
        hidden_size (int): The size of the hidden layers.
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): Total size of all attention heads.
        query (nn.Linear): Linear layer to project hidden states to query.
        key (nn.Linear): Linear layer to project hidden states to key.
        value (nn.Linear): Linear layer to project hidden states to value.
        dropout (nn.Dropout): Dropout layer.
        sparsity_threshold (float): Threshold for pruning attention scores.
        window_size (int): Size of the attention window for locality optimization.
        use_low_rank (bool): Whether to use low-rank approximation.
        rank (int): Rank for low-rank approximation if enabled.
    """
    
    def __init__(self, config):
        """
        Initialize the SparseTrittention module.
        
        Args:
            config: Configuration object with required attributes:
                - hidden_size (int): Size of hidden layers
                - num_attention_heads (int): Number of attention heads
                - attention_probs_dropout_prob (float): Dropout probability
                
                Additional optional attributes for sparse optimization:
                - sparsity_threshold (float): Threshold for attention pruning
                - window_size (int): Size of local attention window
                - use_low_rank (bool): Whether to use low-rank approximation
                - rank (int): Rank for low-rank approximation
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Sparsity parameters (use defaults if not specified in config)
        self.sparsity_threshold = getattr(config, 'sparsity_threshold', 0.01)
        self.window_size = getattr(config, 'window_size', 128)
        self.use_low_rank = getattr(config, 'use_low_rank', False)
        self.rank = getattr(config, 'rank', self.attention_head_size // 4)
        
        assert self.sparsity_threshold >= 0 and self.sparsity_threshold < 1, \
            f"Sparsity threshold must be between 0 and 1, got {self.sparsity_threshold}"
        
        assert self.window_size > 0, \
            f"Window size must be positive, got {self.window_size}"
        
        if self.use_low_rank:
            assert self.rank > 0 and self.rank <= self.attention_head_size, \
                f"Rank must be between 1 and {self.attention_head_size}, got {self.rank}. " \
                f"For hidden_size={self.hidden_size} and num_attention_heads={self.num_attention_heads}, " \
                f"attention_head_size={self.attention_head_size}. Rank cannot exceed attention_head_size."
        
        # Standard projection layers
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Additional layers for low-rank approximation
        if self.use_low_rank:
            self.key_low_rank = nn.Linear(config.hidden_size, self.num_attention_heads * self.rank)
            self.value_low_rank = nn.Linear(config.hidden_size, self.num_attention_heads * self.rank)
            self.key_projection = nn.Linear(self.rank, self.attention_head_size)
            self.value_projection = nn.Linear(self.rank, self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Store attention weights for visualization and analysis
        self.register_buffer('attention_weights', None, persistent=False)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose the input tensor for attention computation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, all_head_size)
            
        Returns:
            Transposed tensor of shape (batch_size, num_heads, seq_length, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def transpose_for_low_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose the input tensor for low-rank attention computation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_heads * rank)
            
        Returns:
            Transposed tensor of shape (batch_size, num_heads, seq_length, rank)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.rank)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def create_sliding_window_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Create a sliding window attention mask.
        
        Args:
            seq_length: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            Window mask of shape (seq_length, seq_length)
        """
        mask = torch.zeros(seq_length, seq_length, device=device)
        window_radius = self.window_size // 2
        
        for i in range(seq_length):
            window_start = max(0, i - window_radius)
            window_end = min(seq_length, i + window_radius + 1)
            mask[i, window_start:window_end] = 1.0
            
        return mask
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                use_sliding_window: bool = True,
                output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for SparseTrittention.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, 1, 1, seq_length)
            use_sliding_window: Whether to use sliding window optimization
            output_attentions: Whether to return attention weights
            
        Returns:
            If output_attentions=False:
                context_layer: Output tensor of shape (batch_size, seq_length, hidden_size)
            Else:
                Tuple of (context_layer, attention_weights)
        """
        batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
        device = hidden_states.device
        
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        
        if self.use_low_rank:
            # Low-rank projection for keys and values
            mixed_key_low_rank = self.key_low_rank(hidden_states)
            mixed_value_low_rank = self.value_low_rank(hidden_states)
            
            # Transpose for attention computation
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer_low_rank = self.transpose_for_low_rank(mixed_key_low_rank)
            value_layer_low_rank = self.transpose_for_low_rank(mixed_value_low_rank)
            
            # Project low-rank representations to full dimension
            key_layer = self.key_projection(key_layer_low_rank)
            value_layer = self.value_projection(value_layer_low_rank)
        else:
            # Standard projection for keys and values
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            
            # Transpose for attention computation
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply sliding window mask if enabled
        if use_sliding_window and seq_length > self.window_size:
            window_mask = self.create_sliding_window_mask(seq_length, device)
            # Convert to proper shape for broadcasting
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)
            # Apply window mask (set negative values to -inf for softmax)
            attention_scores = attention_scores.masked_fill(window_mask == 0, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply sparsity threshold to prune small attention values
        if self.sparsity_threshold > 0:
            # Create sparse mask
            sparse_mask = attention_probs < self.sparsity_threshold
            # Apply sparse mask (set small values to 0)
            attention_probs = attention_probs.masked_fill(sparse_mask, 0.0)
            # Renormalize the remaining values to sum to 1
            attention_sum = attention_probs.sum(dim=-1, keepdim=True)
            attention_probs = attention_probs / (attention_sum + 1e-6)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Save attention weights if needed
        if output_attentions:
            self.attention_weights = attention_probs
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        if output_attentions:
            return context_layer, attention_probs
        else:
            return context_layer
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the last computed attention weights.
        
        Returns:
            Attention weights tensor if available, otherwise None.
        """
        return self.attention_weights


class WindowedTrittention(nn.Module):
    """
    Windowed implementation of the tri-attention mechanism with reduced complexity.
    
    This implementation uses a sliding window approach to limit the context each
    position can attend to, reducing the O(n³) complexity to O(w²n) where w is
    the window size.
    
    Attributes:
        hidden_size (int): The size of the hidden layers.
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): Total size of all attention heads.
        window_size (int): Size of the attention window.
        overlap (int): Overlap between adjacent windows.
        global_tokens (int): Number of global tokens that attend to all positions.
    """
    
    def __init__(self, config):
        """
        Initialize the WindowedTrittention module.
        
        Args:
            config: Configuration object with required attributes:
                - hidden_size (int): Size of hidden layers
                - num_attention_heads (int): Number of attention heads
                - attention_probs_dropout_prob (float): Dropout probability
                - window_size (int): Size of local attention window
                - window_overlap (int): Overlap between windows
                - num_global_tokens (int): Number of global tokens
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Window parameters
        self.window_size = getattr(config, 'window_size', 128)
        self.overlap = getattr(config, 'window_overlap', self.window_size // 4)
        self.global_tokens = getattr(config, 'num_global_tokens', 0)
        
        assert self.window_size > 0, f"Window size must be positive, got {self.window_size}"
        assert self.overlap >= 0 and self.overlap < self.window_size, \
            f"Overlap must be between 0 and {self.window_size-1}, got {self.overlap}"
        assert self.global_tokens >= 0, f"Number of global tokens cannot be negative, got {self.global_tokens}"
        
        # Linear projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Store attention weights for visualization and analysis
        self.register_buffer('attention_weights', None, persistent=False)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose the input tensor for attention computation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, all_head_size)
            
        Returns:
            Transposed tensor of shape (batch_size, num_heads, seq_length, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def _create_window_indices(self, seq_length: int) -> torch.Tensor:
        """
        Create indices for windowed attention.
        
        Args:
            seq_length: Length of the sequence
            
        Returns:
            Tensor of window indices
        """
        effective_window = self.window_size - self.overlap
        
        # Calculate the number of windows needed to cover the entire sequence
        # We need to ensure we cover the entire sequence even if it doesn't divide evenly
        if seq_length <= self.window_size:
            num_windows = 1
        else:
            num_windows = math.ceil((seq_length - self.window_size) / effective_window) + 1
        
        indices = []
        for i in range(num_windows):
            start_idx = i * effective_window
            end_idx = min(start_idx + self.window_size, seq_length)
            indices.extend(list(range(start_idx, end_idx)))
        
        # Use dict.fromkeys to maintain order while removing duplicates
        indices = list(dict.fromkeys(indices))
        
        # Verify we've covered all positions and add any missing ones
        if len(indices) < seq_length:
            missing = set(range(seq_length)) - set(indices)
            indices.extend(sorted(missing))
        
        # Ensure we don't return more indices than the sequence length
        return torch.tensor(indices[:seq_length])
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for WindowedTrittention.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, 1, 1, seq_length)
            output_attentions: Whether to return attention weights
            
        Returns:
            If output_attentions=False:
                context_layer: Output tensor of shape (batch_size, seq_length, hidden_size)
            Else:
                Tuple of (context_layer, attention_weights)
        """
        batch_size, seq_length, _ = hidden_states.size()
        device = hidden_states.device
        
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Transpose for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Process each window separately
        context_layers = []
        attention_probs_list = [] if output_attentions else None
        
        # Handle short sequences with a single window
        if seq_length <= self.window_size:
            # Standard attention calculation
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            if output_attentions:
                attention_probs_list.append(attention_probs)
            
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layers.append(context_layer)
        else:
            # Create window indices
            window_indices = self._create_window_indices(seq_length).to(device)
            
            # Process windows
            effective_window = self.window_size - self.overlap
            
            for i in range(0, seq_length, effective_window):
                end_idx = min(i + self.window_size, seq_length)
                window_len = end_idx - i
                
                # Extract window tensors
                window_query = query_layer[:, :, i:end_idx, :]
                window_key = key_layer[:, :, i:end_idx, :]
                window_value = value_layer[:, :, i:end_idx, :]
                
                # Compute attention within window
                window_attention_scores = torch.matmul(window_query, window_key.transpose(-1, -2))
                window_attention_scores = window_attention_scores / math.sqrt(self.attention_head_size)
                
                # Apply mask if provided
                if attention_mask is not None:
                    window_mask = attention_mask[:, :, :, i:end_idx]
                    window_attention_scores = window_attention_scores + window_mask
                
                window_attention_probs = F.softmax(window_attention_scores, dim=-1)
                window_attention_probs = self.dropout(window_attention_probs)
                
                if output_attentions:
                    attention_probs_list.append(window_attention_probs)
                
                window_context = torch.matmul(window_attention_probs, window_value)
                context_layers.append(window_context)
        
        # Combine window outputs
        if len(context_layers) == 1:
            combined_context = context_layers[0]
        else:
            # Reshape and concatenate context layers with proper handling of overlaps
            reshaped_contexts = []
            for i, context in enumerate(context_layers):
                start_idx = i * effective_window
                end_idx = min(start_idx + self.window_size, seq_length)
                # Only keep the non-overlapping part except for the first and last windows
                if i == 0:
                    keep_end = min(self.window_size, seq_length)
                    context_to_keep = context[:, :, :keep_end, :]
                elif i == len(context_layers) - 1:
                    keep_start = max(0, (end_idx - start_idx) - effective_window)
                    context_to_keep = context[:, :, keep_start:, :]
                else:
                    keep_start = self.overlap // 2
                    keep_end = self.window_size - (self.overlap // 2)
                    context_to_keep = context[:, :, keep_start:keep_end, :]
                
                reshaped_contexts.append(context_to_keep)
            
            # Concatenate along sequence dimension
            combined_context = torch.cat(reshaped_contexts, dim=2)
            
            # Ensure the output has the correct sequence length
            combined_context = combined_context[:, :, :seq_length, :]
        
        # Reshape output
        combined_context = combined_context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = combined_context.size()[:-2] + (self.all_head_size,)
        combined_context = combined_context.view(*new_context_shape)
        
        # Combine attention probabilities if needed
        if output_attentions:
            # Instead of concatenating the attention probabilities (which have different dimensions),
            # we'll create a full attention matrix filled with zeros and fill in each window's values
            batch_size, num_heads = query_layer.shape[0], query_layer.shape[1]
            combined_attention = torch.zeros(batch_size, num_heads, seq_length, seq_length, device=device)
            
            if seq_length <= self.window_size:
                # For short sequences, just use the original attention matrix
                combined_attention = attention_probs_list[0]
            else:
                # For longer sequences, merge the window attention probabilities
                effective_window = self.window_size - self.overlap
                for i, window_probs in enumerate(attention_probs_list):
                    start_idx = i * effective_window
                    end_idx = min(start_idx + self.window_size, seq_length)
                    combined_attention[:, :, start_idx:end_idx, start_idx:end_idx] = window_probs
            
            self.attention_weights = combined_attention
            return combined_context, combined_attention
        
        return combined_context
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the last computed attention weights.
        
        Returns:
            Attention weights tensor if available, otherwise None.
        """
        return self.attention_weights
