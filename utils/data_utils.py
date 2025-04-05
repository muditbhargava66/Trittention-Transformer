"""
Data utilities for the Trittention-Transformer project.

This module provides functions for loading, preprocessing, and managing datasets
for various attention mechanism experiments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Literal, TypeAlias, TypeVar, cast
from collections.abc import Iterable

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


# Type aliases for type hinting (Python 3.10+)
BatchSizeLike: TypeAlias = int | None
T = TypeVar('T')


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic operations.
    
    This dataset loads arithmetic expressions and their results for training
    sequence models on solving arithmetic problems.
    
    Attributes:
        expressions (list): List of arithmetic expressions.
        results (list): List of corresponding results.
        tokenizer (Callable): Function to tokenize expressions.
    """
    
    def __init__(self, 
                 file_path: str | Path, 
                 tokenizer: Callable[[str], list[int]] | None = None,
                 calculate_results: bool = True):
        """Initialize the ArithmeticDataset.
        
        Args:
            file_path: Path to the arithmetic operations file.
            tokenizer: Function to convert expressions to token ids.
                       If None, a simple character-level tokenizer is used.
            calculate_results: Whether to calculate results for expressions.
                              If False, results will be empty.
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            ValueError: If the file format is unexpected.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.tokenizer = tokenizer or self._default_tokenizer
        self.expressions = []
        self.results = []
        
        # Load expressions from file
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove '=' if present
                if line.endswith('='):
                    expression = line[:-1].strip()
                else:
                    expression = line
                
                self.expressions.append(expression)
                
                # Calculate results if needed
                if calculate_results:
                    try:
                        # Safe evaluation of arithmetic expressions
                        result = self._safe_eval(expression)
                        self.results.append(result)
                    except Exception as e:
                        print(f"Error evaluating expression '{expression}': {e}")
                        # Use 0 as default for failed evaluations
                        self.results.append(0)
        
        assert len(self.expressions) > 0, "No valid expressions found in the file"
        if calculate_results:
            assert len(self.expressions) == len(self.results), "Mismatch between expressions and results"
    
    def __len__(self) -> int:
        """Return the number of expressions in the dataset."""
        return len(self.expressions)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get an expression and its result by index.
        
        Args:
            idx: Index of the expression to retrieve.
            
        Returns:
            Tuple of (tokenized_expression, result).
            If calculate_results was False, result will be None.
        """
        expression = self.expressions[idx]
        tokenized = self.tokenizer(expression)
        
        input_tensor = torch.tensor(tokenized, dtype=torch.long)
        
        if self.results:
            result_tensor = torch.tensor(self.results[idx], dtype=torch.float)
            return input_tensor, result_tensor
        
        return input_tensor, None
    
    @staticmethod
    def _default_tokenizer(text: str) -> list[int]:
        """Default tokenizer that maps characters to integers.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of token ids.
        """
        # Simple character-level tokenization
        char_to_id = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '+': 10, '-': 11, '*': 12, '/': 13, ' ': 14,
            '(': 15, ')': 16, '.': 17
        }
        
        return [char_to_id.get(c, 14) for c in text]  # Use space token for unknown chars
    
    @staticmethod
    def _safe_eval(expression: str) -> float:
        """Safely evaluate an arithmetic expression.
        
        Args:
            expression: Arithmetic expression to evaluate.
            
        Returns:
            Result of the expression.
            
        Raises:
            ValueError: If the expression is deemed unsafe.
        """
        # Check for potentially unsafe code constructs
        unsafe_patterns = ['import', '__', 'exec', 'eval', 'os', 'sys', 'subprocess']
        for pattern in unsafe_patterns:
            if pattern in expression:
                raise ValueError(f"Expression contains unsafe pattern '{pattern}': {expression}")
        
        # Check for potentially unsafe code using alpha characters
        if re.search(r'[a-zA-Z_]', expression):
            raise ValueError(f"Expression contains unsafe characters: {expression}")
        
        # Check for exponentiation which could lead to very large numbers
        if '**' in expression:
            raise ValueError(f"Expression contains exponentiation which can lead to excessive computation: {expression}")
        
        # Use a safer version of eval for arithmetic expressions only
        allowed_chars = set('0123456789+-*/.() ')
        if not set(expression).issubset(allowed_chars):
            raise ValueError(f"Expression contains disallowed characters: {expression}")
        
        # Evaluate the expression
        try:
            return eval(expression, {"__builtins__": {}})
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {expression}") from e


class SequenceDataset(Dataset):
    """Dataset for sequence-based problems like longest increasing subsequence.
    
    Attributes:
        sequences (list): List of sequence data.
        targets (list): List of target values for each sequence.
    """
    
    def __init__(self, file_path: str | Path, target_fn: Callable[[list], list] | None = None):
        """Initialize the SequenceDataset.
        
        Args:
            file_path: Path to the sequence data file.
            target_fn: Function to calculate target values for sequences.
                       If None, longest increasing subsequence is calculated.
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.sequences = []
        self.targets = []
        self.target_fn = target_fn or self._longest_increasing_subsequence
        
        # Load sequences from file
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the sequence - expecting space-separated numbers
                try:
                    sequence = [int(x) for x in line.split()]
                    self.sequences.append(sequence)
                    
                    # Calculate the target for this sequence
                    target = self.target_fn(sequence)
                    self.targets.append(target)
                except Exception as e:
                    print(f"Error processing sequence '{line}': {e}")
        
        assert len(self.sequences) > 0, "No valid sequences found in the file"
        assert len(self.sequences) == len(self.targets), "Mismatch between sequences and targets"
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target by index.
        
        Args:
            idx: Index of the sequence to retrieve.
            
        Returns:
            Tuple of (sequence_tensor, target_tensor).
        """
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.float)
        
        return sequence_tensor, target_tensor
    
    @staticmethod
    def _longest_increasing_subsequence(sequence: list[int]) -> list[int]:
        """Calculate the longest increasing subsequence (LIS).
        
        Args:
            sequence: Input sequence of integers.
            
        Returns:
            Binary mask where 1 indicates that the element is part of the LIS.
        """
        if not sequence:
            return []
        
        n = len(sequence)
        # Initialize list for the length of LIS ending at each position
        lis_length = [1] * n
        # Initialize list for backtracking the LIS
        lis_prev = [-1] * n
        
        # Calculate LIS length at each position
        for i in range(1, n):
            for j in range(i):
                if sequence[j] < sequence[i] and lis_length[j] + 1 > lis_length[i]:
                    lis_length[i] = lis_length[j] + 1
                    lis_prev[i] = j
        
        # Find the position of the maximum LIS length
        max_length_idx = max(range(n), key=lambda i: lis_length[i])
        
        # Backtrack to construct the LIS
        lis_indices = set()
        while max_length_idx != -1:
            lis_indices.add(max_length_idx)
            max_length_idx = lis_prev[max_length_idx]
        
        # Create a binary mask for the LIS
        lis_mask = [1 if i in lis_indices else 0 for i in range(n)]
        return lis_mask


def load_dataset(data_type: Literal['arithmetic', 'sequence'], 
                 file_path: str | Path) -> Dataset:
    """Load a dataset of the specified type.
    
    Args:
        data_type: Type of dataset to load ('arithmetic' or 'sequence').
        file_path: Path to the data file.
        
    Returns:
        Loaded dataset object.
        
    Raises:
        ValueError: If the data_type is not supported.
    """
    match data_type:
        case 'arithmetic':
            return ArithmeticDataset(file_path)
        case 'sequence':
            return SequenceDataset(file_path)
        case _:
            raise ValueError(f"Unsupported data type: {data_type}")


def create_dataloaders(dataset: Dataset, 
                       batch_size: int = 32, 
                       val_split: float = 0.2,
                       seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders from a dataset.
    
    Args:
        dataset: The dataset to split.
        batch_size: Batch size for the dataloaders.
        val_split: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    assert 0.0 <= val_split < 1.0, "Validation split must be between 0 and 1"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_dataloader, val_dataloader


def pad_sequences(sequences: list[torch.Tensor], 
                  padding_value: int = 0) -> torch.Tensor:
    """Pad sequences to the same length.
    
    Args:
        sequences: List of sequences to pad.
        padding_value: Value to use for padding.
        
    Returns:
        Tensor of padded sequences.
    """
    # Find the length of the longest sequence
    max_length = max(len(seq) for seq in sequences)
    
    # Pad each sequence to the maximum length
    padded_sequences = []
    for seq in sequences:
        padded = torch.full((max_length,), padding_value, dtype=seq.dtype)
        padded[:len(seq)] = seq
        padded_sequences.append(padded)
    
    # Stack all sequences into a single tensor
    return torch.stack(padded_sequences)


def collate_variable_length_sequences(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for batches with variable length sequences.
    
    Args:
        batch: List of (input, target) tuples.
        
    Returns:
        Tuple of (padded_inputs, padded_targets).
    """
    # Separate inputs and targets
    inputs, targets = zip(*batch)
    
    # Pad inputs
    padded_inputs = pad_sequences(inputs)
    
    # Pad targets if they are sequences, otherwise stack them
    if isinstance(targets[0], torch.Tensor) and targets[0].dim() > 0:
        padded_targets = pad_sequences(targets)
    else:
        padded_targets = torch.stack(targets)
    
    return padded_inputs, padded_targets


def get_toy_datasets_path() -> Path:
    """Get the path to the toy datasets directory.
    
    Returns:
        Path to the toy datasets directory.
    """
    # Assuming the directory structure from the project layout
    project_root = Path(__file__).parent.parent
    toy_problems_dir = project_root / 'data' / 'toy_problems'
    
    assert toy_problems_dir.exists(), f"Toy problems directory not found: {toy_problems_dir}"
    return toy_problems_dir


def load_toy_dataset(dataset_name: Literal['arithmetic_operations', 'longest_increasing_subsequence']) -> Dataset:
    """Load one of the toy datasets by name.
    
    Args:
        dataset_name: Name of the toy dataset to load.
        
    Returns:
        Loaded dataset.
        
    Raises:
        ValueError: If the dataset name is not recognized.
    """
    toy_dir = get_toy_datasets_path()
    
    match dataset_name:
        case 'arithmetic_operations':
            file_path = toy_dir / 'arithmetic_operations.txt'
            return ArithmeticDataset(file_path)
        case 'longest_increasing_subsequence':
            file_path = toy_dir / 'longest_increasing_subsequence.txt'
            return SequenceDataset(file_path)
        case _:
            raise ValueError(f"Unknown toy dataset: {dataset_name}")
