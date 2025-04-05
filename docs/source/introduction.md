# Introduction to Trittention

## What is Trittention?

Trittention is a novel attention mechanism that extends the traditional self-attention in transformer models from pairwise (2-way) interactions to three-way (3-way) interactions. By capturing higher-order relationships between tokens, trittention offers a more expressive framework for modeling complex dependencies in sequential data.

### Standard Attention vs. Trittention

**Standard Self-Attention**:
- Computes pairwise relationships between tokens
- Complexity: O(n²) where n is the sequence length
- Captures direct relationships between pairs of tokens

**Trittention (3-way Attention)**:
- Computes relationships between triples of tokens
- Complexity: O(n³) where n is the sequence length
- Captures higher-order dependencies and more complex patterns

## Motivation

Transformer models have revolutionized natural language processing and many other domains, but standard self-attention has inherent limitations in capturing complex, higher-order relationships in sequential data. Some complex patterns in language, code, or other sequential data may require modeling interactions between more than two elements.

Trittention aims to address this limitation by:

1. Extending attention to capture relationships between triples of tokens
2. Enabling more expressive modeling of complex patterns
3. Providing a framework for exploring higher-order attention mechanisms

## Mathematical Formulation

### Standard Self-Attention

Standard self-attention can be formulated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, and $V$ are the query, key, and value matrices, and $d_k$ is the dimensionality of the key vectors.

### Trittention

Trittention extends this to three-way interactions:

$$\text{Trittention}(Q, K, V) = \text{softmax}\left(\frac{QK^TV}{\sqrt{d_k}}\right)$$

This formulation captures interactions between triples of elements in the sequence, leading to a higher-order attention mechanism.

## Computational Considerations

While trittention offers increased expressivity, it comes with increased computational complexity (from O(n²) to O(n³)). To address this challenge, we've implemented several optimized variants:

- **Sparse Trittention**: Reduces computation by focusing on the most important interactions
- **Windowed Trittention**: Limits attention to local windows, reducing complexity to O(n·w²) where w is the window size
- **Mixed Attention**: Combines standard attention and trittention for a balance of efficiency and expressivity

These optimizations make trittention practical for real-world applications while preserving its modeling benefits.

## Applications

Trittention can be particularly beneficial for tasks involving complex dependencies, such as:

- **Complex Sequence Modeling**: Tasks where understanding the relationship between multiple elements is critical
- **Code Understanding**: Capturing relationships between variables, functions, and operations
- **Structured Data**: Modeling data with inherent hierarchical or graph-like structure
- **Long-range Dependencies**: Tasks requiring understanding connections between distant elements

The next sections will guide you through installing and using Trittention-Transformer for your own projects.
