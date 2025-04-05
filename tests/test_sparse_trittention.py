"""
Unit tests for the sparse trittention implementation.

This module contains tests to verify the correctness and efficiency of the 
sparse trittention attention mechanisms.
"""

import unittest
import torch
import time
import math
from typing import Tuple, List

from models.sparse_trittention import SparseTrittention, WindowedTrittention
from config.cfgs import TrittentionConfig


class TestSparseTrittention(unittest.TestCase):
    """Test suite for the SparseTrittention implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a small config for testing
        self.small_config = TrittentionConfig(
            hidden_size=64,
            num_attention_heads=4,
            attention_probs_dropout_prob=0.1,
            sparsity_threshold=0.05,
            window_size=8
        )
        
        # Create a larger config for benchmarking
        self.large_config = TrittentionConfig(
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            sparsity_threshold=0.1,
            window_size=32
        )
        
        # Config with low-rank approximation enabled
        self.lowrank_config = TrittentionConfig(
            hidden_size=64,
            num_attention_heads=4,
            attention_probs_dropout_prob=0.1,
            sparsity_threshold=0.05,
            window_size=8
        )
        setattr(self.lowrank_config, 'use_low_rank', True)
        setattr(self.lowrank_config, 'rank', 16)
        
        # Initialize the models
        self.sparse_trittention = SparseTrittention(self.small_config)
        self.low_rank_trittention = SparseTrittention(self.lowrank_config)
        
        # Set eval mode to disable dropout for deterministic results
        self.sparse_trittention.eval()
        self.low_rank_trittention.eval()
    
    def test_initialization(self):
        """Test initialization of SparseTrittention."""
        # Check that the model was initialized correctly
        model = self.sparse_trittention
        
        # Check dimensions
        self.assertEqual(model.hidden_size, self.small_config.hidden_size)
        self.assertEqual(model.num_attention_heads, self.small_config.num_attention_heads)
        self.assertEqual(model.attention_head_size, self.small_config.hidden_size // self.small_config.num_attention_heads)
        self.assertEqual(model.all_head_size, model.num_attention_heads * model.attention_head_size)
        
        # Check sparsity parameters
        self.assertEqual(model.sparsity_threshold, self.small_config.sparsity_threshold)
        self.assertEqual(model.window_size, self.small_config.window_size)
        
        # Check low-rank related attributes
        model = self.low_rank_trittention
        self.assertTrue(model.use_low_rank)
        self.assertEqual(model.rank, 16)
        
        # Check that the required projection layers exist
        self.assertIsNotNone(model.key_low_rank)
        self.assertIsNotNone(model.value_low_rank)
        self.assertIsNotNone(model.key_projection)
        self.assertIsNotNone(model.value_projection)
    
    def test_forward_shape(self):
        """Test that the forward pass returns the correct shape."""
        # Create input tensors
        batch_size = 2
        seq_length = 10
        hidden_size = self.small_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass through regular sparse trittention
        output = self.sparse_trittention(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))
        
        # Forward pass through low-rank sparse trittention
        output = self.low_rank_trittention(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))
    
    def test_forward_with_mask(self):
        """Test that attention mask is correctly applied."""
        # Create input tensors
        batch_size = 2
        seq_length = 10
        hidden_size = self.small_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Create attention mask (allow attention only to the first 5 tokens)
        attention_mask = torch.zeros(batch_size, 1, 1, seq_length)
        attention_mask[:, :, :, :5] = 0  # Set to 0 for allowed positions
        attention_mask[:, :, :, 5:] = -10000.0  # Set to large negative for masked positions
        
        # Forward pass with mask
        output_with_mask = self.sparse_trittention(hidden_states, attention_mask)
        
        # Forward pass without mask
        output_no_mask = self.sparse_trittention(hidden_states)
        
        # Check that outputs are different (mask should change the output)
        self.assertFalse(torch.allclose(output_with_mask, output_no_mask, rtol=1e-3))
    
    def test_sliding_window(self):
        """Test that sliding window attention is correctly applied."""
        # Create input tensors
        batch_size = 2
        seq_length = 20  # Longer than window size
        hidden_size = self.small_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass with sliding window
        output_with_window = self.sparse_trittention(hidden_states, use_sliding_window=True)
        
        # Forward pass without sliding window
        output_no_window = self.sparse_trittention(hidden_states, use_sliding_window=False)
        
        # Check that outputs are different (window should change the output)
        self.assertFalse(torch.allclose(output_with_window, output_no_window, rtol=1e-3))
    
    def test_sparsity(self):
        """Test that sparsity threshold is correctly applied."""
        # Create a simple input for visualization
        batch_size = 1
        seq_length = 10
        hidden_size = self.small_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass with attention weights output
        output, attention_weights = self.sparse_trittention(hidden_states, output_attentions=True)
        
        # Check that some attention weights are exactly zero (due to sparsity)
        zero_weights = (attention_weights == 0).float().sum()
        total_weights = attention_weights.numel()
        
        # There should be some zeros due to sparsity threshold
        self.assertGreater(zero_weights, 0)
        
        # Percentage of zeros should be roughly proportional to sparsity threshold
        zero_ratio = zero_weights / total_weights
        # Allow for some variation due to randomness
        self.assertGreaterEqual(zero_ratio, self.small_config.sparsity_threshold * 0.5)
    
    def test_get_attention_weights(self):
        """Test that attention weights are correctly stored and retrievable."""
        # Create input tensors
        batch_size = 2
        seq_length = 10
        hidden_size = self.small_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass with attention weights output
        _, attention_weights = self.sparse_trittention(hidden_states, output_attentions=True)
        
        # Get attention weights through the getter method
        retrieved_weights = self.sparse_trittention.get_attention_weights()
        
        # Check that retrieved weights match the original weights
        self.assertTrue(torch.allclose(attention_weights, retrieved_weights))
    
    def test_create_sliding_window_mask(self):
        """Test that the sliding window mask is correctly created."""
        seq_length = 15
        window_size = self.small_config.window_size
        device = torch.device("cpu")
        
        # Create mask
        mask = self.sparse_trittention.create_sliding_window_mask(seq_length, device)
        
        # Check shape
        self.assertEqual(mask.shape, (seq_length, seq_length))
        
        # Check window pattern
        window_radius = window_size // 2
        for i in range(seq_length):
            # Positions within the window should be 1.0
            window_start = max(0, i - window_radius)
            window_end = min(seq_length, i + window_radius + 1)
            
            # Check window boundaries
            for j in range(seq_length):
                if window_start <= j < window_end:
                    self.assertEqual(mask[i, j].item(), 1.0)
                else:
                    self.assertEqual(mask[i, j].item(), 0.0)
    
    def test_output_stability(self):
        """Test that the output is stable across multiple calls."""
        # Create input tensors
        batch_size = 2
        seq_length = 10
        hidden_size = self.small_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Multiple forward passes
        output1 = self.sparse_trittention(hidden_states)
        output2 = self.sparse_trittention(hidden_states)
        
        # Outputs should be identical (model is in eval mode)
        self.assertTrue(torch.allclose(output1, output2, rtol=1e-6))
    
    def benchmark_attention_mechanisms(self, sequence_lengths: List[int]) -> Tuple[dict, dict]:
        """
        Benchmark different attention mechanisms with varying sequence lengths.
        
        Args:
            sequence_lengths: List of sequence lengths to test.
            
        Returns:
            Tuple of dictionaries with timing and memory results.
        """
        # Create models to benchmark
        config = self.large_config
        sparse_model = SparseTrittention(config)
        sparse_model.eval()
        
        lowrank_config = self.large_config
        setattr(lowrank_config, 'use_low_rank', True)
        setattr(lowrank_config, 'rank', config.hidden_size // 4)
        lowrank_model = SparseTrittention(lowrank_config)
        lowrank_model.eval()
        
        # Batch size and hidden size
        batch_size = 1
        hidden_size = config.hidden_size
        
        # Dictionaries to store results
        time_results = {
            "SparseTrittention": [],
            "LowRankTrittention": []
        }
        
        for seq_len in sequence_lengths:
            # Create input
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            
            # Benchmark sparse model
            start_time = time.time()
            with torch.no_grad():
                _ = sparse_model(hidden_states)
            sparse_time = time.time() - start_time
            time_results["SparseTrittention"].append(sparse_time)
            
            # Benchmark low-rank model
            start_time = time.time()
            with torch.no_grad():
                _ = lowrank_model(hidden_states)
            lowrank_time = time.time() - start_time
            time_results["LowRankTrittention"].append(lowrank_time)
        
        return time_results, {}  # No memory tracking in this test
    
    def test_benchmark(self):
        """Run benchmarks and display results."""
        # Only run this test if explicitly enabled
        import sys
        if not any("--benchmark" in arg for arg in sys.argv):
            self.skipTest("Benchmark test skipped (enable with --benchmark)")
        
        # Sequence lengths to test
        sequence_lengths = [10, 20, 50, 100, 200, 500]
        
        # Run benchmark
        time_results, _ = self.benchmark_attention_mechanisms(sequence_lengths)
        
        # Print results
        print("\nBenchmark Results (Inference Time in seconds):")
        print("Sequence Length | SparseTrittention | LowRankTrittention")
        print("-" * 60)
        
        for i, seq_len in enumerate(sequence_lengths):
            sparse_time = time_results["SparseTrittention"][i]
            lowrank_time = time_results["LowRankTrittention"][i]
            print(f"{seq_len:15d} | {sparse_time:16.6f} | {lowrank_time:17.6f}")
        
        # For each model, check that time complexity scales with sequence length
        for model_name, times in time_results.items():
            # Time should generally increase with sequence length
            for i in range(1, len(times)):
                # Allow some exceptions due to measurement noise for very small sequences
                if sequence_lengths[i] >= 100:
                    self.assertGreater(times[i], times[0])


class TestWindowedTrittention(unittest.TestCase):
    """Test suite for the WindowedTrittention implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a config for testing
        self.config = TrittentionConfig(
            hidden_size=64,
            num_attention_heads=4,
            attention_probs_dropout_prob=0.1,
            window_size=16,
            window_overlap=4,
            num_global_tokens=2
        )
        
        # Initialize the model
        self.windowed_trittention = WindowedTrittention(self.config)
        
        # Set eval mode to disable dropout for deterministic results
        self.windowed_trittention.eval()
    
    def test_initialization(self):
        """Test initialization of WindowedTrittention."""
        # Check that the model was initialized correctly
        model = self.windowed_trittention
        
        # Check dimensions
        self.assertEqual(model.hidden_size, self.config.hidden_size)
        self.assertEqual(model.num_attention_heads, self.config.num_attention_heads)
        self.assertEqual(model.attention_head_size, self.config.hidden_size // self.config.num_attention_heads)
        self.assertEqual(model.all_head_size, model.num_attention_heads * model.attention_head_size)
        
        # Check window parameters
        self.assertEqual(model.window_size, 16)
        self.assertEqual(model.overlap, 4)
        self.assertEqual(model.global_tokens, 2)
    
    def test_forward_shape(self):
        """Test that the forward pass returns the correct shape."""
        # Create input tensors
        batch_size = 2
        seq_length = 32
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass
        output = self.windowed_trittention(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))
    
    def test_short_sequence(self):
        """Test with a sequence shorter than the window size."""
        # Create a short input
        batch_size = 2
        seq_length = 8  # Shorter than window size (16)
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass
        output = self.windowed_trittention(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))
    
    def test_window_indices(self):
        """Test window indices generation."""
        seq_length = 50
        indices = self.windowed_trittention._create_window_indices(seq_length)
        
        # Check that all indices are covered
        unique_indices = torch.unique(indices)
        self.assertEqual(len(unique_indices), seq_length)
        
        # Check min and max indices
        self.assertEqual(unique_indices.min().item(), 0)
        self.assertEqual(unique_indices.max().item(), seq_length - 1)
    
    def test_output_attentions(self):
        """Test that attention weights are correctly returned."""
        # Create input tensors
        batch_size = 2
        seq_length = 24
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Forward pass with attention weights output
        output, attention_weights = self.windowed_trittention(hidden_states, output_attentions=True)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))
        
        # Check that attention weights are not None
        self.assertIsNotNone(attention_weights)
        
        # Get attention weights through the getter method
        retrieved_weights = self.windowed_trittention.get_attention_weights()
        
        # Check that retrieved weights match the original weights
        self.assertTrue(torch.allclose(attention_weights, retrieved_weights))


if __name__ == '__main__':
    unittest.main()
