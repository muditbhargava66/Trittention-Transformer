# models/trittention_cube.py

import torch
import torch.nn as nn
import math

class TrittentionCube(nn.Module):
    """
    Implements a cubic tri-attention mechanism.

    Attributes:
        hidden_size (int): The size of the hidden layers.
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): Total size of all attention heads.
        query (nn.Linear): Linear layer to project hidden states to query.
        key (nn.Linear): Linear layer to project hidden states to key.
        value (nn.Linear): Linear layer to project hidden states to value.
        cube_key (nn.Linear): Linear layer to project hidden states to cubic key.
        cube_value (nn.Linear): Linear layer to project hidden states to cubic value.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, config):
        """
        Initializes the TrittentionCube class with the given configuration.

        Args:
            config: Configuration object with attributes:
                hidden_size (int): Size of hidden layers.
                num_attention_heads (int): Number of attention heads.
                attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.cube_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.cube_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transposes the input tensor to shape (batch_size, num_attention_heads, seq_length, attention_head_size).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transposed tensor.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs forward pass for the cubic tri-attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor, optional): Attention mask to prevent attention to certain positions.

        Returns:
            torch.Tensor: Contextualized output after applying cubic tri-attention.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_cube_key_layer = self.cube_key(hidden_states)
        mixed_cube_value_layer = self.cube_value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        cube_key_layer = self.transpose_for_scores(mixed_cube_key_layer)
        cube_value_layer = self.transpose_for_scores(mixed_cube_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = torch.matmul(attention_scores, cube_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = torch.matmul(context_layer, cube_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
