"""
Data generation script for Trittention-Transformer examples.

This script generates synthetic datasets for testing and benchmarking
different attention mechanisms. It provides options to create time series
and text classification datasets with controllable properties.
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable

import numpy as np
import pandas as pd


def generate_time_series(
    n_samples: int = 10000,
    n_features: int = 5,
    seq_length: int = 200,
    freq: str = 'H',
    seasonality: bool = True,
    trend: bool = True,
    noise_level: float = 0.1,
    missing_values: float = 0.0,
    n_anomalies: int = 0,
    start_date: str = '2023-01-01'
) -> pd.DataFrame:
    """
    Generate synthetic time series data.
    
    Args:
        n_samples: Number of samples (time steps)
        n_features: Number of features
        seq_length: Seasonality period (e.g., 24 for hourly data with daily seasonality)
        freq: Time series frequency (e.g., 'H' for hourly, 'D' for daily)
        seasonality: Whether to include seasonality
        trend: Whether to include trend
        noise_level: Level of noise (standard deviation)
        missing_values: Fraction of missing values
        n_anomalies: Number of anomalies to add
        start_date: Start date for the time series
        
    Returns:
        DataFrame with synthetic time series data
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # Initialize DataFrame
    df = pd.DataFrame(index=dates)
    
    # Add features
    for i in range(n_features):
        # Base signal
        signal = np.zeros(n_samples)
        
        # Add trend
        if trend:
            # Random trend strength
            trend_strength = np.random.uniform(0.001, 0.01)
            signal += np.arange(n_samples) * trend_strength
        
        # Add seasonality
        if seasonality:
            # Different seasonality patterns for each feature
            for j in range(1, 4):  # Multiple seasonal components
                # Random seasonality strength and phase
                seas_strength = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                # Create seasonality with different frequencies
                freq_multiplier = j * np.random.randint(1, 4)
                seasonal_component = seas_strength * np.sin(
                    2 * np.pi * freq_multiplier * np.arange(n_samples) / seq_length + phase
                )
                signal += seasonal_component
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        signal += noise
        
        # Add to DataFrame
        df[f'feature_{i+1}'] = signal
    
    # Add target variable (combination of features with noise)
    weights = np.random.uniform(-1, 1, n_features)
    target = np.zeros(n_samples)
    
    for i in range(n_features):
        target += weights[i] * df[f'feature_{i+1}'].values
    
    # Add noise to target
    target += np.random.normal(0, noise_level * 0.5, n_samples)
    
    # Add target to DataFrame
    df['target'] = target
    
    # Add anomalies
    if n_anomalies > 0:
        # Get random indices for anomalies
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Add spike or drop anomaly
            if np.random.rand() > 0.5:
                # Spike (multiply by random factor)
                df.iloc[idx, -1] *= np.random.uniform(2.0, 5.0)
            else:
                # Drop (divide by random factor)
                df.iloc[idx, -1] /= np.random.uniform(2.0, 5.0)
    
    # Add missing values
    if missing_values > 0:
        # Calculate number of missing values
        n_missing = int(n_samples * n_features * missing_values)
        
        # Get random indices for missing values
        missing_rows = np.random.choice(n_samples, n_missing, replace=True)
        missing_cols = np.random.choice(n_features, n_missing, replace=True)
        
        for row, col in zip(missing_rows, missing_cols):
            df.iloc[row, col] = np.nan
    
    return df


def generate_text_classification(
    n_samples: int = 1000,
    n_classes: int = 2,
    vocab_size: int = 5000,
    max_length: int = 100,
    min_length: int = 10,
    class_names: Optional[List[str]] = None,
    class_distribution: Optional[List[float]] = None,
    include_special_tokens: bool = True
) -> Tuple[List[str], List[int], List[str]]:
    """
    Generate synthetic text classification data.
    
    Args:
        n_samples: Number of samples
        n_classes: Number of classes
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        class_names: Names of classes (if None, will be generated)
        class_distribution: Distribution of classes (if None, will be uniform)
        include_special_tokens: Whether to include special tokens
        
    Returns:
        Tuple of (texts, labels, class_names)
    """
    # Generate vocabulary
    vocab = [f"token_{i}" for i in range(vocab_size)]
    
    # Add special tokens
    if include_special_tokens:
        vocab.extend(["<pad>", "<unk>", "<bos>", "<eos>"])
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    else:
        assert len(class_names) == n_classes, "Number of class names must match n_classes"
    
    # Generate class distribution if not provided
    if class_distribution is None:
        class_distribution = [1.0 / n_classes] * n_classes
    else:
        assert len(class_distribution) == n_classes, "Class distribution must have n_classes elements"
        assert sum(class_distribution) == 1.0, "Class distribution must sum to 1.0"
    
    # Generate class-specific word distributions
    class_word_probs = []
    for _ in range(n_classes):
        # Generate random word probabilities for this class
        word_probs = np.random.dirichlet(np.ones(vocab_size) * 0.5)
        class_word_probs.append(word_probs)
    
    # Generate samples
    texts = []
    labels = []
    
    for _ in range(n_samples):
        # Choose class based on distribution
        label = np.random.choice(n_classes, p=class_distribution)
        labels.append(label)
        
        # Choose sequence length
        seq_length = np.random.randint(min_length, max_length + 1)
        
        # Generate tokens based on class-specific distribution
        tokens = np.random.choice(
            vocab,
            size=seq_length,
            p=class_word_probs[label]
        )
        
        # Convert tokens to text
        text = " ".join(tokens)
        texts.append(text)
    
    return texts, labels, class_names


def save_time_series(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
    include_metadata: bool = True
) -> None:
    """
    Save time series data to file.
    
    Args:
        df: DataFrame with time series data
        output_path: Path to save file
        format: File format ('csv' or 'parquet')
        include_metadata: Whether to include metadata
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    if format.lower() == 'csv':
        df.to_csv(output_path, index=True)
    elif format.lower() == 'parquet':
        df.to_parquet(output_path, index=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Save metadata if requested
    if include_metadata:
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
        
        metadata = {
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,  # Excluding target
            "feature_names": list(df.columns[:-1]),
            "target_name": df.columns[-1],
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "frequency": pd.infer_freq(df.index)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Saved time series data to {output_path}")
    if include_metadata:
        print(f"Saved metadata to {metadata_path}")


def save_text_classification(
    texts: List[str],
    labels: List[int],
    class_names: List[str],
    output_path: str,
    format: str = 'csv',
    include_metadata: bool = True
) -> None:
    """
    Save text classification data to file.
    
    Args:
        texts: List of text samples
        labels: List of labels
        class_names: Names of classes
        output_path: Path to save file
        format: File format ('csv' or 'parquet')
        include_metadata: Whether to include metadata
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Save data
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Save metadata if requested
    if include_metadata:
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
        
        # Calculate some statistics
        avg_length = sum(len(text.split()) for text in texts) / len(texts)
        min_length = min(len(text.split()) for text in texts)
        max_length = max(len(text.split()) for text in texts)
        
        class_counts = {}
        for label in labels:
            class_name = class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        metadata = {
            "n_samples": len(texts),
            "n_classes": len(class_names),
            "class_names": class_names,
            "class_counts": class_counts,
            "avg_text_length": avg_length,
            "min_text_length": min_length,
            "max_text_length": max_length
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Saved text classification data to {output_path}")
    if include_metadata:
        print(f"Saved metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for Trittention-Transformer")
    
    # General parameters
    parser.add_argument("--output_dir", type=str, default="./data/synthetic",
                        help="Directory to save generated data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Subparsers for different data types
    subparsers = parser.add_subparsers(dest="data_type", help="Type of data to generate")
    
    # Time series parameters
    ts_parser = subparsers.add_parser("time_series", help="Generate synthetic time series data")
    ts_parser.add_argument("--n_samples", type=int, default=10000,
                          help="Number of samples (time steps)")
    ts_parser.add_argument("--n_features", type=int, default=5,
                          help="Number of features")
    ts_parser.add_argument("--seq_length", type=int, default=24,
                          help="Seasonality period")
    ts_parser.add_argument("--freq", type=str, default="H",
                          help="Time series frequency")
    ts_parser.add_argument("--no_seasonality", action="store_true",
                          help="Disable seasonality")
    ts_parser.add_argument("--no_trend", action="store_true",
                          help="Disable trend")
    ts_parser.add_argument("--noise_level", type=float, default=0.1,
                          help="Level of noise")
    ts_parser.add_argument("--missing_values", type=float, default=0.0,
                          help="Fraction of missing values")
    ts_parser.add_argument("--n_anomalies", type=int, default=0,
                          help="Number of anomalies")
    ts_parser.add_argument("--start_date", type=str, default="2023-01-01",
                          help="Start date for the time series")
    ts_parser.add_argument("--format", type=str, default="csv",
                          choices=["csv", "parquet"],
                          help="Output file format")
    
    # Text classification parameters
    text_parser = subparsers.add_parser("text", help="Generate synthetic text classification data")
    text_parser.add_argument("--n_samples", type=int, default=1000,
                            help="Number of samples")
    text_parser.add_argument("--n_classes", type=int, default=2,
                            help="Number of classes")
    text_parser.add_argument("--vocab_size", type=int, default=5000,
                            help="Size of vocabulary")
    text_parser.add_argument("--max_length", type=int, default=100,
                            help="Maximum sequence length")
    text_parser.add_argument("--min_length", type=int, default=10,
                            help="Minimum sequence length")
    text_parser.add_argument("--class_names", type=str, nargs="+", default=None,
                            help="Names of classes")
    text_parser.add_argument("--format", type=str, default="csv",
                            choices=["csv", "parquet"],
                            help="Output file format")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.data_type == "time_series":
        # Generate time series data
        print(f"Generating synthetic time series data with {args.n_samples} samples and {args.n_features} features")
        df = generate_time_series(
            n_samples=args.n_samples,
            n_features=args.n_features,
            seq_length=args.seq_length,
            freq=args.freq,
            seasonality=not args.no_seasonality,
            trend=not args.no_trend,
            noise_level=args.noise_level,
            missing_values=args.missing_values,
            n_anomalies=args.n_anomalies,
            start_date=args.start_date
        )
        
        # Save data
        output_path = os.path.join(args.output_dir, f"time_series.{args.format}")
        save_time_series(df, output_path, format=args.format)
    
    elif args.data_type == "text":
        # Generate text classification data
        print(f"Generating synthetic text classification data with {args.n_samples} samples and {args.n_classes} classes")
        texts, labels, class_names = generate_text_classification(
            n_samples=args.n_samples,
            n_classes=args.n_classes,
            vocab_size=args.vocab_size,
            max_length=args.max_length,
            min_length=args.min_length,
            class_names=args.class_names
        )
        
        # Save data
        output_path = os.path.join(args.output_dir, f"text_classification.{args.format}")
        save_text_classification(texts, labels, class_names, output_path, format=args.format)
    
    else:
        print("Please specify a data type to generate (time_series or text)")
        sys.exit(1)


if __name__ == "__main__":
    main()
