"""
Fine-tuning script for sequence modeling tasks.

This script demonstrates how to fine-tune trittention models on sequence-based
tasks like time series prediction, natural language processing, or other 
sequential data problems.
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split

# Add parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.cfgs import TrittentionConfig
from models.lightning_module import TrittentionLightningModule, TrittentionDataModule


class SequenceDataset(Dataset):
    """
    Dataset for sequence modeling tasks.
    
    This dataset handles loading sequence data from various sources and preparing
    it for training sequence models.
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 128,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        stride: int = 1
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (CSV, TSV, or JSON)
            seq_length: Length of sequences to generate
            target_cols: Column names to use as targets (if None, last column used)
            feature_cols: Column names to use as features (if None, all non-target columns used)
            normalize: Whether to normalize features
            stride: Stride for sequence generation
        """
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.normalize = normalize
        self.stride = stride
        
        # Load data based on file extension
        self.data = self._load_data()
        
        # Determine target and feature columns
        if self.target_cols is None:
            # Use last column as target by default
            self.target_cols = [self.data.columns[-1]]
        
        if self.feature_cols is None:
            # Use all columns that are not targets as features
            self.feature_cols = [col for col in self.data.columns if col not in self.target_cols]
        
        # Extract features and targets
        self.features = self.data[self.feature_cols].values
        self.targets = self.data[self.target_cols].values
        
        # Normalize if requested
        if self.normalize:
            self._normalize_data()
        
        # Create sequences
        self._create_sequences()
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from the specified file.
        
        Returns:
            DataFrame containing the loaded data
        """
        # Get file extension
        ext = self.data_path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(self.data_path)
        elif ext == '.tsv':
            return pd.read_csv(self.data_path, sep='\t')
        elif ext == '.json':
            return pd.read_json(self.data_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _normalize_data(self):
        """Normalize features and targets to zero mean and unit variance."""
        # Calculate mean and std for features
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero
        
        # Normalize features
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        # Calculate mean and std for targets
        self.target_mean = np.mean(self.targets, axis=0)
        self.target_std = np.std(self.targets, axis=0)
        self.target_std[self.target_std == 0] = 1.0  # Avoid division by zero
        
        # Normalize targets
        self.targets = (self.targets - self.target_mean) / self.target_std
    
    def _create_sequences(self):
        """Create sequences for training."""
        self.sequence_indices = []
        
        # Generate sequence indices
        for i in range(0, len(self.features) - self.seq_length + 1, self.stride):
            self.sequence_indices.append(i)
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence by index.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (features, targets) for the sequence
        """
        # Get starting index for this sequence
        start_idx = self.sequence_indices[idx]
        
        # Extract sequence
        feature_seq = self.features[start_idx:start_idx + self.seq_length]
        target_seq = self.targets[start_idx:start_idx + self.seq_length]
        
        # Convert to tensors
        feature_tensor = torch.tensor(feature_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32)
        
        return feature_tensor, target_tensor
    
    def denormalize_targets(self, normalized_targets: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Denormalize target values.
        
        Args:
            normalized_targets: Normalized target values
            
        Returns:
            Denormalized target values
        """
        if self.normalize:
            # Convert to numpy if tensor
            if isinstance(normalized_targets, torch.Tensor):
                normalized_targets = normalized_targets.detach().cpu().numpy()
                
            # Denormalize
            return normalized_targets * self.target_std + self.target_mean
        else:
            # No normalization was applied
            if isinstance(normalized_targets, torch.Tensor):
                return normalized_targets.detach().cpu().numpy()
            return normalized_targets
    
    def get_feature_dim(self) -> int:
        """Get the dimension of features."""
        return len(self.feature_cols)
    
    def get_target_dim(self) -> int:
        """Get the dimension of targets."""
        return len(self.target_cols)
    
    def get_column_names(self) -> Tuple[List[str], List[str]]:
        """Get feature and target column names."""
        return self.feature_cols, self.target_cols


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune trittention models on sequence tasks")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data file (CSV, TSV, or JSON)")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Length of sequences")
    parser.add_argument("--target_cols", type=str, nargs="+", default=None,
                        help="Target column names")
    parser.add_argument("--feature_cols", type=str, nargs="+", default=None,
                        help="Feature column names")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize data")
    
    # Model parameters
    parser.add_argument("--attention_type", type=str, default="trittention",
                        choices=["standard", "trittention", "trittention_cube", 
                                "local", "mixed", "sparse", "windowed"],
                        help="Type of attention mechanism to use")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=64,
                        help="Window size for local attention")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "step", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    
    # Experiment parameters
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (default: attention_type)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.attention_type}_{args.hidden_size}"
    
    # Convert "none" to None for scheduler
    if args.scheduler == "none":
        args.scheduler = None
    
    return args


def create_config(args):
    """Create configuration object from arguments."""
    config = TrittentionConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        window_size=args.window_size,
        # Set use flags based on attention type
        use_trittention=args.attention_type in ["trittention", "trittention_cube"],
        use_local_trittention=args.attention_type == "local",
        use_mixed_attention=args.attention_type == "mixed"
    )
    
    return config


def main():
    """Main function to run the fine-tuning."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Create result directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = SequenceDataset(
        data_path=args.data_path,
        seq_length=args.seq_length,
        target_cols=args.target_cols,
        feature_cols=args.feature_cols,
        normalize=args.normalize
    )
    
    # Print dataset info
    feature_cols, target_cols = dataset.get_column_names()
    print(f"Dataset loaded with {len(dataset)} sequences")
    print(f"Feature columns: {feature_cols}")
    print(f"Target columns: {target_cols}")
    
    # Split dataset into train and validation
    train_size = int(len(dataset) * (1 - args.val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data module
    data_module = TrittentionDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    config = create_config(args)
    model = TrittentionLightningModule(
        attention_type=args.attention_type,
        config=config,
        input_size=dataset.get_feature_dim(),
        hidden_size=args.hidden_size,
        output_size=dataset.get_target_dim(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "checkpoints" / args.experiment_name,
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=save_dir / "logs",
        name=args.experiment_name
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu" if args.gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Evaluate model on validation set
    val_result = trainer.test(model, data_module.val_dataloader())[0]
    
    # Save model parameters
    model_path = save_dir / f"{args.experiment_name}_weights.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save configuration
    config_path = save_dir / f"{args.experiment_name}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "args": vars(args),
            "validation_result": val_result,
            "feature_columns": feature_cols,
            "target_columns": target_cols,
            "num_parameters": sum(p.numel() for p in model.parameters())
        }, f, indent=2)
    
    print(f"Training completed. Results saved to {save_dir}")
    print(f"Validation results: {val_result}")


if __name__ == "__main__":
    main()
