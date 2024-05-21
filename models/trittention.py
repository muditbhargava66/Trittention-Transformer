# models/trittention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Trittention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key1_layer = self.key1(hidden_states)
        mixed_key2_layer = self.key2(hidden_states)
        mixed_value1_layer = self.value1(hidden_states)
        mixed_value2_layer = self.value2(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key1_layer = self.transpose_for_scores(mixed_key1_layer)
        key2_layer = self.transpose_for_scores(mixed_key2_layer)
        value1_layer = self.transpose_for_scores(mixed_value1_layer)
        value2_layer = self.transpose_for_scores(mixed_value2_layer)

        attention_scores = torch.matmul(query_layer, key1_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=attention_scores.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value1_layer)
        context_layer = torch.matmul(context_layer, key2_layer.transpose(-1, -2))
        context_layer = torch.matmul(context_layer, value2_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer