"""
Example script for training Trittention models with PyTorch Lightning.

This script demonstrates how to use the modernized codebase to train and evaluate
different attention mechanisms efficiently.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config.cfgs import TrittentionConfig
from utils.data_utils import load_toy_dataset, create_dataloaders
from models.lightning_module import TrittentionLightningModule, TrittentionDataModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate Trittention models")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="arithmetic_operations",
                        choices=["arithmetic_operations", "longest_increasing_subsequence"],
                        help="Dataset to use for training and evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Model parameters
    parser.add_argument("--attention_type", type=str, default="trittention",
                        choices=["standard", "trittention", "trittention_cube", 
                                "local", "mixed", "sparse", "windowed"],
                        help="Type of attention mechanism to use")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=64, help="Window size for local attention")
    parser.add_argument("--use_low_rank", action="store_true", help="Use low-rank approximation for sparse attention")
    parser.add_argument("--sparsity_threshold", type=float, default=0.01, 
                        help="Sparsity threshold for sparse attention")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs")
    parser.add_argument("--optimizer", type=str, default="adamw", 
                        choices=["adam", "adamw", "sgd"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["cosine", "linear", "step", "none"], 
                        help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    
    # Experiment parameters
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--experiment_name", type=str, default=None, 
                        help="Experiment name (default: attention_type)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_attention", action="store_true", 
                        help="Save attention pattern visualizations")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.attention_type}_{args.hidden_size}_{args.dataset}"
    
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
    
    # Add sparse attention parameters if needed
    if args.attention_type in ["sparse", "windowed"]:
        config.sparsity_threshold = args.sparsity_threshold
        config.use_low_rank = args.use_low_rank
        if args.use_low_rank:
            config.rank = args.hidden_size // 4
    
    return config


def main():
    """Main function to run the training and evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Create result directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_toy_dataset(args.dataset)
    
    # Determine input and output sizes based on dataset
    sample_input, sample_target = dataset[0]
    input_size = sample_input.size(-1) if sample_input.dim() > 1 else 1
    output_size = sample_target.size(-1) if sample_target.dim() > 1 else 1
    
    # Create LightningModule
    config = create_config(args)
    model = TrittentionLightningModule(
        attention_type=args.attention_type,
        config=config,
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=output_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        save_attention_patterns=args.save_attention
    )
    
    # Create DataModule
    data_module = TrittentionDataModule(
        train_dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed
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
    
    # Test model using the validation set as test set (since we don't have a separate test set)
    trainer.test(model, data_module.val_dataloader())
    
    # Save model parameters separately
    torch.save(model.state_dict(), save_dir / f"{args.experiment_name}_weights.pt")
    
    print(f"Training completed. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
