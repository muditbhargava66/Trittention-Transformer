# models/trittention.py

import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from config.cfgs import TrittentionConfig as Config
import math

class Trittention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = cfg.hidden_size // cfg.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value1 = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(cfg.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

        self.dense = nn.Linear(self.all_head_size, cfg.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def apply_causal_mask(self, attn_scores):
        batch_size, num_heads, seq_length, _ = attn_scores.size()
        mask = t.triu(t.ones(seq_length, seq_length), diagonal=1).unsqueeze(0).unsqueeze(0)
        mask = mask.to(dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores.masked_fill_(mask == 1, float("-inf"))
        return attn_scores

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        query_layer = self.query(hidden_states)
        key1_layer = self.key1(hidden_states)
        key2_layer = self.key2(hidden_states)
        value1_layer = self.value1(hidden_states)
        value2_layer = self.value2(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key1_layer = self.transpose_for_scores(key1_layer)
        key2_layer = self.transpose_for_scores(key2_layer)
        value1_layer = self.transpose_for_scores(value1_layer)
        value2_layer = self.transpose_for_scores(value2_layer)

        # Compute attention scores
        attention_scores = t.matmul(query_layer, key1_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply causal mask
        attention_scores = self.apply_causal_mask(attention_scores)

        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=attention_scores.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Apply dropout
        attention_probs = self.dropout(attention_probs)

        # Compute context layer
        context_layer = t.matmul(attention_probs, value1_layer)
        context_layer = t.matmul(context_layer, key2_layer.transpose(-1, -2))
        context_layer = t.matmul(context_layer, value2_layer)

        # Transpose and reshape the context layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Apply dense layer
        output = self.dense(context_layer)

        return output
    