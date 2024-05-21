# models/mixed_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MixedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_local_heads = config.num_local_heads
        self.num_global_heads = config.num_attention_heads - config.num_local_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.local_query = nn.Linear(config.hidden_size, self.num_local_heads * self.attention_head_size)
        self.local_key1 = nn.Linear(config.hidden_size, self.num_local_heads * self.attention_head_size)
        self.local_key2 = nn.Linear(config.hidden_size, self.num_local_heads * self.attention_head_size)
        self.local_value1 = nn.Linear(config.hidden_size, self.num_local_heads * self.attention_head_size)
        self.local_value2 = nn.Linear(config.hidden_size, self.num_local_heads * self.attention_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, num_heads):
        new_x_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, self.num_global_heads)
        key_layer = self.transpose_for_scores(mixed_key_layer, self.num_global_heads)
        value_layer = self.transpose_for_scores(mixed_value_layer, self.num_global_heads)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        mixed_local_query_layer = self.local_query(hidden_states)
        mixed_local_key1_layer = self.local_key1(hidden_states)
        mixed_local_key2_layer = self.local_key2(hidden_states)
        mixed_local_value1_layer = self.local_value1(hidden_states)
        mixed_local_value2_layer = self.local_value2(hidden_states)

        local_query_layer = self.transpose_for_scores(mixed_local_query_layer, self.num_local_heads)
        local_key1_layer = self.transpose_for_scores(mixed_local_key1_layer, self.num_local_heads)
        local_key2_layer = self.transpose_for_scores(mixed_local_key2_layer, self.num_local_heads)
        local_value1_layer = self.transpose_for_scores(mixed_local_value1_layer, self.num_local_heads)
        local_value2_layer = self.transpose_for_scores(mixed_local_value2_layer, self.num_local_heads)

        local_attention_scores = torch.matmul(local_query_layer, local_key1_layer.transpose(-1, -2))
        local_attention_scores = torch.matmul(local_attention_scores, local_key2_layer.transpose(-1, -2))
        local_attention_scores = local_attention_scores / math.sqrt(self.attention_head_size)

        local_attention_probs = nn.Softmax(dim=-1)(local_attention_scores)
        local_attention_probs = self.dropout(local_attention_probs)

        local_context1_layer = torch.matmul(local_attention_probs, local_value1_layer)
        local_context2_layer = torch.matmul(local_attention_probs, local_value2_layer)
        local_context_layer = local_context1_layer + local_context2_layer

        context_layer = torch.cat([context_layer, local_context_layer], dim=1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer