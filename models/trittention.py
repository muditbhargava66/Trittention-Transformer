# models/trittention.py

import torch
import torch.nn as nn
import math

class Trittention(nn.Module):
    """
    Implements tri-attention mechanism combining multiple types of attention.

    Attributes:
        hidden_size (int): The size of the hidden layers.
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): Total size of all attention heads.
        query (nn.Linear): Linear layer to project hidden states to query.
        key (nn.Linear): Linear layer to project hidden states to key.
        value (nn.Linear): Linear layer to project hidden states to value.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, config):
        """
        Initializes the Trittention class with the given configuration.

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
        Performs forward pass for the tri-attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor, optional): Attention mask to prevent attention to certain positions.

        Returns:
            torch.Tensor: Contextualized output after applying tri-attention.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous().view(context_layer.size(0), -1, self.all_head_size)

        return context_layer

    def backward(self, grad_output):
        """
        Performs backward pass for the tri-attention mechanism.

        Args:
            grad_output (torch.Tensor): Gradient of the output.

        Returns:
            torch.Tensor: Gradient of the input.
        """
        grad_input = torch.autograd.grad(outputs=self.forward_output, inputs=self.hidden_states, grad_outputs=grad_output, retain_graph=True)[0]
        return grad_input
