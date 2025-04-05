"""
Unit tests for the data_utils module.

This module contains tests to verify the correctness of the data loading and
processing functions in the utils.data_utils module.
"""

import unittest
import os
import tempfile
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.data_utils import (
    ArithmeticDataset,
    SequenceDataset,
    load_dataset,
    create_dataloaders,
    pad_sequences,
    collate_variable_length_sequences,
    get_toy_datasets_path,
    load_toy_dataset
)


class TestArithmeticDataset(unittest.TestCase):
    """Test suite for the ArithmeticDataset class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary file with arithmetic expressions
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.write("3 + 5 =\n")
        self.temp_file.write("10 - 4 =\n")
        self.temp_file.write("7 * 2 =\n")
        self.temp_file.flush()
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = ArithmeticDataset(self.temp_file.name)
        
        # Check expressions
        self.assertEqual(len(dataset.expressions), 3)
        self.assertEqual(dataset.expressions[0], "3 + 5")
        self.assertEqual(dataset.expressions[1], "10 - 4")
        self.assertEqual(dataset.expressions[2], "7 * 2")
        
        # Check results
        self.assertEqual(len(dataset.results), 3)
        self.assertEqual(dataset.results[0], 8)
        self.assertEqual(dataset.results[1], 6)
        self.assertEqual(dataset.results[2], 14)
        
        # Check __len__
        self.assertEqual(len(dataset), 3)
    
    def test_getitem(self):
        """Test retrieving items from the dataset."""
        dataset = ArithmeticDataset(self.temp_file.name)
        
        # Get first item
        input_tensor, result_tensor = dataset[0]
        
        # Check types
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(result_tensor, torch.Tensor)
        
        # Check values - input should be tokenized
        self.assertEqual(result_tensor.item(), 8)
    
    def test_default_tokenizer(self):
        """Test the default tokenizer."""
        dataset = ArithmeticDataset(self.temp_file.name)
        
        # Test tokenization of expression
        tokenized = dataset._default_tokenizer("3 + 5")
        
        # Check that tokens are mapped to integers
        self.assertIsInstance(tokenized, list)
        self.assertTrue(all(isinstance(token, int) for token in tokenized))
        
        # Check specific mappings
        self.assertEqual(tokenized[0], 3)  # '3' -> 3
        self.assertEqual(tokenized[1], 14)  # ' ' -> 14
        self.assertEqual(tokenized[2], 10)  # '+' -> 10
        self.assertEqual(tokenized[3], 14)  # ' ' -> 14
        self.assertEqual(tokenized[4], 5)  # '5' -> 5
    
    def test_safe_eval(self):
        """Test the safe_eval method."""
        # Valid expressions
        self.assertEqual(ArithmeticDataset._safe_eval("3 + 5"), 8)
        self.assertEqual(ArithmeticDataset._safe_eval("10 - 4"), 6)
        self.assertEqual(ArithmeticDataset._safe_eval("7 * 2"), 14)
        self.assertEqual(ArithmeticDataset._safe_eval("8 / 2"), 4)
        self.assertEqual(ArithmeticDataset._safe_eval("(2 + 3) * 4"), 20)
        
        # Invalid expressions should raise ValueError
        with self.assertRaises(ValueError):
            ArithmeticDataset._safe_eval("print('hello')")
        
        with self.assertRaises(ValueError):
            ArithmeticDataset._safe_eval("__import__('os').system('ls')")
        
        with self.assertRaises(ValueError):
            ArithmeticDataset._safe_eval("3 ** 1000")  # Too large
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for nonexistent files."""
        with self.assertRaises(FileNotFoundError):
            ArithmeticDataset("nonexistent_file.txt")


class TestSequenceDataset(unittest.TestCase):
    """Test suite for the SequenceDataset class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary file with sequences
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.write("1 3 5 7 9\n")
        self.temp_file.write("2 4 6 8 10\n")
        self.temp_file.write("5 10 15 20 25\n")
        self.temp_file.flush()
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = SequenceDataset(self.temp_file.name)
        
        # Check sequences
        self.assertEqual(len(dataset.sequences), 3)
        self.assertEqual(dataset.sequences[0], [1, 3, 5, 7, 9])
        self.assertEqual(dataset.sequences[1], [2, 4, 6, 8, 10])
        self.assertEqual(dataset.sequences[2], [5, 10, 15, 20, 25])
        
        # Check targets (should be binary masks for LIS)
        self.assertEqual(len(dataset.targets), 3)
        
        # The longest increasing subsequence for [1, 3, 5, 7, 9] is all elements
        self.assertEqual(dataset.targets[0], [1, 1, 1, 1, 1])
        
        # Check __len__
        self.assertEqual(len(dataset), 3)
    
    def test_getitem(self):
        """Test retrieving items from the dataset."""
        dataset = SequenceDataset(self.temp_file.name)
        
        # Get first item
        sequence_tensor, target_tensor = dataset[0]
        
        # Check types
        self.assertIsInstance(sequence_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)
        
        # Check values
        self.assertEqual(sequence_tensor.tolist(), [1, 3, 5, 7, 9])
        self.assertEqual(target_tensor.tolist(), [1, 1, 1, 1, 1])
    
    def test_longest_increasing_subsequence(self):
        """Test the longest increasing subsequence calculation."""
        # Test with increasing sequence
        lis = SequenceDataset._longest_increasing_subsequence([1, 2, 3, 4, 5])
        self.assertEqual(lis, [1, 1, 1, 1, 1])  # All elements form LIS
        
        # Test with decreasing sequence
        lis = SequenceDataset._longest_increasing_subsequence([5, 4, 3, 2, 1])
        self.assertEqual(sum(lis), 1)  # Only one element in LIS
        
        # Test with non-monotonic sequence
        lis = SequenceDataset._longest_increasing_subsequence([1, 3, 2, 4, 1])
        self.assertEqual(sum(lis), 3)  # LIS has 3 elements
        
        # Test with empty sequence
        lis = SequenceDataset._longest_increasing_subsequence([])
        self.assertEqual(lis, [])
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for nonexistent files."""
        with self.assertRaises(FileNotFoundError):
            SequenceDataset("nonexistent_file.txt")


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for the utility functions in data_utils."""
    
    def test_load_dataset(self):
        """Test the load_dataset function."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as arith_file:
            arith_file.write("3 + 5 =\n")
            arith_file.write("10 - 4 =\n")
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as seq_file:
            seq_file.write("1 3 5 7 9\n")
            seq_file.write("2 4 6 8 10\n")
        
        try:
            # Test arithmetic dataset
            dataset = load_dataset("arithmetic", arith_file.name)
            self.assertIsInstance(dataset, ArithmeticDataset)
            self.assertEqual(len(dataset), 2)
            
            # Test sequence dataset
            dataset = load_dataset("sequence", seq_file.name)
            self.assertIsInstance(dataset, SequenceDataset)
            self.assertEqual(len(dataset), 2)
            
            # Test invalid data_type
            with self.assertRaises(ValueError):
                load_dataset("invalid_type", seq_file.name)
        
        finally:
            os.unlink(arith_file.name)
            os.unlink(seq_file.name)
    
    def test_create_dataloaders(self):
        """Test the create_dataloaders function."""
        # Create a simple dataset
        x = torch.randn(100, 10)
        y = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(x, y)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(dataset, batch_size=16, val_split=0.2)
        
        # Check types
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        
        # Check batch size
        self.assertEqual(train_loader.batch_size, 16)
        self.assertEqual(val_loader.batch_size, 16)
        
        # Check dataset sizes (80% train, 20% val)
        self.assertEqual(len(train_loader.dataset), 80)
        self.assertEqual(len(val_loader.dataset), 20)
    
    def test_pad_sequences(self):
        """Test the pad_sequences function."""
        # Create sequences of different lengths
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9])
        ]
        
        # Pad sequences
        padded = pad_sequences(sequences, padding_value=0)
        
        # Check shape (should be [3, 4] - 3 sequences, max length 4)
        self.assertEqual(padded.shape, (3, 4))
        
        # Check values
        self.assertEqual(padded[0].tolist(), [1, 2, 3, 0])
        self.assertEqual(padded[1].tolist(), [4, 5, 0, 0])
        self.assertEqual(padded[2].tolist(), [6, 7, 8, 9])
    
    def test_collate_variable_length_sequences(self):
        """Test the collate_variable_length_sequences function."""
        # Create batch of sequences and targets
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([1, 0, 1])),
            (torch.tensor([4, 5]), torch.tensor([0, 1])),
            (torch.tensor([6, 7, 8, 9]), torch.tensor([1, 0, 1, 0]))
        ]
        
        # Collate batch
        inputs, targets = collate_variable_length_sequences(batch)
        
        # Check shapes
        self.assertEqual(inputs.shape, (3, 4))
        self.assertEqual(targets.shape, (3, 4))
        
        # Check values
        self.assertEqual(inputs[0].tolist(), [1, 2, 3, 0])
        self.assertEqual(targets[1].tolist(), [0, 1, 0, 0])
    
    def test_get_toy_datasets_path(self):
        """Test the get_toy_datasets_path function."""
        # Get toy datasets path
        toy_path = get_toy_datasets_path()
        
        # Check that it's a Path object
        self.assertIsInstance(toy_path, Path)
        
        # Check that the directory exists
        self.assertTrue(toy_path.exists())
        self.assertTrue(toy_path.is_dir())
        
        # Check that it contains the expected files
        self.assertTrue((toy_path / "arithmetic_operations.txt").exists())
        self.assertTrue((toy_path / "longest_increasing_subsequence.txt").exists())
    
    def test_load_toy_dataset(self):
        """Test the load_toy_dataset function."""
        # Load arithmetic dataset
        arith_dataset = load_toy_dataset("arithmetic_operations")
        self.assertIsInstance(arith_dataset, ArithmeticDataset)
        self.assertGreater(len(arith_dataset), 0)
        
        # Load sequence dataset
        seq_dataset = load_toy_dataset("longest_increasing_subsequence")
        self.assertIsInstance(seq_dataset, SequenceDataset)
        self.assertGreater(len(seq_dataset), 0)
        
        # Test invalid dataset name
        with self.assertRaises(ValueError):
            load_toy_dataset("invalid_dataset")


if __name__ == "__main__":
    unittest.main()
