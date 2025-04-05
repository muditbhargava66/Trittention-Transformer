# Attention Mechanisms

This page details the various attention mechanisms implemented in the Trittention-Transformer project. Each mechanism offers different trade-offs between expressivity, computational complexity, and memory usage.

## Standard Attention

Standard attention is the baseline attention mechanism used in transformer models as introduced in "Attention Is All You Need" (Vaswani et al., 2017).

### Implementation Details

```python
def forward(self, hidden_states, attention_mask=None):
    # Project inputs to queries, keys, and values
    mixed_query_layer = self.query(hidden_states)
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
    
    # Apply softmax to get attention probabilities
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    attention_probs = self.dropout(attention_probs)
    
    # Apply attention to values
    context_layer = torch.matmul(attention_probs, value_layer)
    
    return context_layer
```

### Complexity

- Computational Complexity: O(n²) where n is the sequence length
- Memory Complexity: O(n²)

## Trittention

Trittention extends standard attention to capture three-way interactions between tokens, offering increased expressivity for modeling complex dependencies.

### Implementation Details

```python
def forward(self, hidden_states, attention_mask=None):
    # Implementation similar to standard attention
    # but with additional operations for three-way interactions
    
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)
    
    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)
    
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = torch.matmul(attention_scores, value_layer)
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    
    # ... rest of the implementation
```

### Complexity

- Computational Complexity: O(n³) where n is the sequence length
- Memory Complexity: O(n²)

## Trittention Cube

Trittention Cube is a variant of trittention that uses additional cubic attention patterns for even more expressive modeling.

### Implementation Details

```python
def forward(self, hidden_states, attention_mask=None):
    # Projects inputs to query, key, value, and additional cubic key/value
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)
    mixed_cube_key_layer = self.cube_key(hidden_states)
    mixed_cube_value_layer = self.cube_value(hidden_states)
    
    # ... process projections ...
    
    # Two levels of attention interactions
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = torch.matmul(attention_scores, cube_key_layer.transpose(-1, -2))
    
    # ... apply attention ...
    
    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = torch.matmul(context_layer, cube_value_layer)
    
    # ... final processing ...
```

### Complexity

- Computational Complexity: O(n³)
- Memory Complexity: O(n²)

## Sparse Trittention

Sparse Trittention applies sparsity optimizations to reduce the computational complexity of trittention while preserving its modeling capabilities.

### Implementation Details

```python
def forward(self, hidden_states, attention_mask=None, use_sliding_window=True):
    # ... standard projections and initial processing ...
    
    # Apply sparsity threshold to prune small attention values
    if self.sparsity_threshold > 0:
        # Create sparse mask
        sparse_mask = attention_probs < self.sparsity_threshold
        # Apply sparse mask (set small values to 0)
        attention_probs = attention_probs.masked_fill(sparse_mask, 0.0)
        # Renormalize the remaining values to sum to 1
        attention_sum = attention_probs.sum(dim=-1, keepdim=True)
        attention_probs = attention_probs / (attention_sum + 1e-6)
    
    # Apply sliding window if enabled
    if use_sliding_window and seq_length > self.window_size:
        window_mask = self.create_sliding_window_mask(seq_length, device)
        # Apply window mask (set negative values to -inf for softmax)
        attention_scores = attention_scores.masked_fill(window_mask == 0, float('-inf'))
    
    # ... rest of implementation ...
```

### Complexity

- Computational Complexity: O(s·n³) where s is the sparsity factor
- Memory Complexity: O(s·n²)

## Windowed Trittention

Windowed Trittention limits attention to a local window, significantly reducing computational complexity.

### Implementation Details

```python
def forward(self, hidden_states, attention_mask=None):
    # ... standard projections ...
    
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
        
        # ... process window attention ...
        
        context_layers.append(window_context)
    
    # Combine window outputs
    combined_context = torch.cat(reshaped_contexts, dim=2)
```

### Complexity

- Computational Complexity: O(n·w²) where w is the window size
- Memory Complexity: O(n·w)

## Mixed Attention

Mixed Attention combines different attention mechanisms, allocating some heads for global attention and others for local attention.

### Implementation Details

```python
def forward(self, hidden_states, attention_mask=None):
    # ... standard projections ...
    
    # Split heads into global and local groups
    global_heads_q = query_layer[:, :self.num_global_heads]
    global_heads_k = key_layer[:, :self.num_global_heads]
    global_heads_v = value_layer[:, :self.num_global_heads]
    
    local_heads_q = query_layer[:, self.num_global_heads:]
    local_heads_k = key_layer[:, self.num_global_heads:]
    local_heads_v = value_layer[:, self.num_global_heads:]
    
    # Process global attention (standard attention)
    global_attn = self._global_attention(
        global_heads_q, global_heads_k, global_heads_v, attention_mask
    )
    
    # Process local attention (windowed attention)
    local_attn = self._local_attention(
        local_heads_q, local_heads_k, local_heads_v, attention_mask
    )
    
    # Combine global and local attention outputs
    combined_attn = torch.cat([global_attn, local_attn], dim=1)
```

### Complexity

- Computational Complexity: Mix of O(n²) and O(n·w²)
- Memory Complexity: Mix of O(n²) and O(n·w)

## Performance Comparison

| Mechanism | Expressivity | Computational Complexity | Memory Usage | Best For |
|-----------|--------------|--------------------------|--------------|----------|
| Standard Attention | Moderate | O(n²) | O(n²) | Baseline transformer tasks |
| Trittention | High | O(n³) | O(n²) | Complex pattern modeling |
| Trittention Cube | Very High | O(n³) | O(n²) | Maximum expressivity needs |
| Sparse Trittention | High | O(s·n³) | O(s·n²) | Balancing expressivity and efficiency |
| Windowed Trittention | Moderate | O(n·w²) | O(n·w) | Long sequence modeling |
| Mixed Attention | High | Mixed | Mixed | Adaptive modeling needs |

## Choosing the Right Mechanism

- **For maximum expressivity**: Use Trittention or Trittention Cube
- **For long sequences**: Use Windowed Trittention or Sparse Trittention
- **For balanced performance**: Use Mixed Attention
- **For baseline comparison**: Use Standard Attention

Each mechanism offers different trade-offs, and the best choice depends on your specific task requirements, sequence lengths, and computational constraints.
