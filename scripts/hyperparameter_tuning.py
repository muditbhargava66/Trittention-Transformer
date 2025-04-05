"""
Hyperparameter tuning script for Trittention models.

This script performs hyperparameter optimization for different attention mechanisms
using PyTorch Lightning and optuna for efficient hyperparameter search.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from datetime import datetime

import torch
import optuna
from optuna.trial import Trial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent directory to Python path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.append(str(project_dir))

from config.cfgs import TrittentionConfig
from models.lightning_module import TrittentionLightningModule, TrittentionDataModule
from utils.data_utils import load_toy_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Trittention models")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="arithmetic_operations",
                        choices=["arithmetic_operations", "longest_increasing_subsequence"],
                        help="Dataset to use for training and evaluation")
    parser.add_argument("--batch_size_min", type=int, default=8,
                        help="Minimum batch size to try")
    parser.add_argument("--batch_size_max", type=int, default=64,
                        help="Maximum batch size to try")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # Model parameters
    parser.add_argument("--attention_type", type=str, default="trittention",
                        choices=["standard", "trittention", "trittention_cube", 
                                "local", "mixed", "sparse", "windowed"],
                        help="Type of attention mechanism to tune")
    parser.add_argument("--hidden_size_min", type=int, default=32,
                        help="Minimum hidden size to try")
    parser.add_argument("--hidden_size_max", type=int, default=256,
                        help="Maximum hidden size to try")
    parser.add_argument("--num_heads_min", type=int, default=1,
                        help="Minimum number of attention heads to try")
    parser.add_argument("--num_heads_max", type=int, default=8,
                        help="Maximum number of attention heads to try")
    
    # Training parameters
    parser.add_argument("--learning_rate_min", type=float, default=1e-5,
                        help="Minimum learning rate to try")
    parser.add_argument("--learning_rate_max", type=float, default=1e-2,
                        help="Maximum learning rate to try")
    parser.add_argument("--weight_decay_min", type=float, default=1e-6,
                        help="Minimum weight decay to try")
    parser.add_argument("--weight_decay_max", type=float, default=0.1,
                        help="Maximum weight decay to try")
    parser.add_argument("--dropout_min", type=float, default=0.0,
                        help="Minimum dropout probability to try")
    parser.add_argument("--dropout_max", type=float, default=0.5,
                        help="Maximum dropout probability to try")
    
    # Optimization parameters
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of hyperparameter optimization trials")
    parser.add_argument("--epochs_per_trial", type=int, default=20,
                        help="Maximum number of epochs per trial")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./results/hyperparameter_tuning",
                        help="Directory to save results")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Name for the optimization study")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Set study name if not provided
    if args.study_name is None:
        args.study_name = f"{args.attention_type}_{args.dataset}_tuning"
    
    return args


def define_hyperparameter_space(trial: Trial, args) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for optuna.
    
    Args:
        trial: Optuna trial object
        args: Command line arguments
        
    Returns:
        Dictionary of hyperparameters
    """
    # Model hyperparameters
    params = {
        # Model architecture
        "hidden_size": trial.suggest_int("hidden_size", args.hidden_size_min, args.hidden_size_max, log=True),
        "num_attention_heads": trial.suggest_int("num_attention_heads", args.num_heads_min, args.num_heads_max),
        
        # Training hyperparameters
        "learning_rate": trial.suggest_float("learning_rate", args.learning_rate_min, args.learning_rate_max, log=True),
        "batch_size": trial.suggest_int("batch_size", args.batch_size_min, args.batch_size_max, log=True),
        
        # Regularization
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
    }
    
    # Handle weight decay - use log=True only if min value is > 0
    # Ensure weight_decay_min is positive for log scaling
    weight_decay_min = max(1e-6, args.weight_decay_min)
    weight_decay_max = args.weight_decay_max
    
    # Add weight decay parameter
    params["weight_decay"] = trial.suggest_float("weight_decay", weight_decay_min, weight_decay_max, log=True)
    
    # Attention-specific hyperparameters
    if args.attention_type in ["local", "mixed", "sparse", "windowed"]:
        params["window_size"] = trial.suggest_int("window_size", 4, min(128, args.hidden_size_max), log=True)
    
    if args.attention_type == "sparse":
        params["sparsity_threshold"] = trial.suggest_float("sparsity_threshold", 0.01, 0.5, log=True)
    
    # Make sure hidden_size is divisible by num_attention_heads
    params["hidden_size"] = params["hidden_size"] - (params["hidden_size"] % params["num_attention_heads"])
    if params["hidden_size"] < params["num_attention_heads"]:
        params["hidden_size"] = params["num_attention_heads"]
    
    return params


def create_model_and_datamodule(params: Dict[str, Any], args) -> Tuple[TrittentionLightningModule, TrittentionDataModule]:
    """
    Create model and data module with given hyperparameters.
    
    Args:
        params: Hyperparameters dictionary
        args: Command line arguments
        
    Returns:
        Tuple of (model, data_module)
    """
    # Create configuration
    config = TrittentionConfig(
        hidden_size=params["hidden_size"],
        num_attention_heads=params["num_attention_heads"],
        attention_probs_dropout_prob=params["dropout"],
        hidden_dropout_prob=params["dropout"]
    )
    
    # Set attention-specific parameters
    if "window_size" in params:
        config.window_size = params["window_size"]
    
    if "sparsity_threshold" in params:
        config.sparsity_threshold = params["sparsity_threshold"]
    
    # Set use flags based on attention type
    config.use_trittention = args.attention_type in ["trittention", "trittention_cube"]
    config.use_local_trittention = args.attention_type == "local"
    config.use_mixed_attention = args.attention_type == "mixed"
    
    # Load dataset
    dataset = load_toy_dataset(args.dataset)
    
    # Create an embedding for arithmetic operations if needed
    embedding_dim = None
    if args.dataset == 'arithmetic_operations':
        # Configure token embedding if using token-based dataset
        vocab_size = 32  # Assuming the tokenizer creates ids 0-31
        embedding_dim = params["hidden_size"] // 2  # Use half the hidden size for embedding dimension
        # Only print in debug mode
        # print(f"Setting up token embedding with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
        # We'll use the embedding dimension as input_size
        input_size = embedding_dim
        
        # For arithmetic operations, output is a scalar value
        # Get a sample target to verify
        _, sample_target = dataset[0]
        # Ensure output_size is set based on actual data
        if sample_target.dim() > 0:
            output_size = sample_target.size(-1) if sample_target.dim() > 1 else 1
        else:
            output_size = 1
    else:
        # Determine input and output sizes based on dataset
        sample_input, sample_target = dataset[0]
        
        # For debug purposes only
        # print(f"\nInput tensor shape: {sample_input.shape}, type: {sample_input.dtype}")
        # print(f"Target tensor shape: {sample_target.shape}, type: {sample_target.dtype}\n")
        
        # Handle different input tensor dimensions
        if sample_input.dim() > 1:
            # For multi-dimensional inputs, use the last dimension
            input_size = sample_input.size(-1)
        else:
            # For 1D inputs (e.g., token sequences), we'll use an embedding
            # For the SequenceDataset, this means using the tensor as-is
            input_size = 1
        
        # Handle different output tensor dimensions
        if sample_target.dim() > 1:
            output_size = sample_target.size(-1)
        else:
            output_size = 1  # For scalar outputs
    
    # Print debug info when debugging
    print(f"Model configuration - input_size: {input_size}, hidden_size: {params['hidden_size']}, output_size: {output_size}")
    
    # Ensure input_size is at least 1
    if input_size <= 0:
        print(f"Warning: Invalid input size {input_size}, setting to 1")
        input_size = 1
        
    # Ensure output_size is at least 1
    if output_size <= 0:
        print(f"Warning: Invalid output size {output_size}, setting to 1")
        output_size = 1
    
    # Create model
    model = TrittentionLightningModule(
        attention_type=args.attention_type,
        config=config,
        input_size=input_size,
        hidden_size=params["hidden_size"],
        output_size=output_size,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        optimizer_type="adamw",
        scheduler_type="cosine",
        vocab_size=32 if args.dataset == 'arithmetic_operations' else None,
        embedding_dim=embedding_dim
    )
    
    # Ensure the dataset is preloaded and initialized
    data_module = TrittentionDataModule(
        train_dataset=dataset,
        batch_size=params["batch_size"],
        num_workers=4,
        val_split=args.val_split
    )
    
    # Debug batch information
    # try:
    #     # Set up data module to access sample batches
    #     data_module.setup()
    #     
    #     # Get a sample batch to check dimensions
    #     sample_batch = next(iter(data_module.train_dataloader()))
    #     print(f"\nSample batch shapes: inputs={sample_batch[0].shape}, targets={sample_batch[1].shape}")
    # except Exception as e:
    #     print(f"Could not print sample batch info: {e}")
    
    return model, data_module


def objective(trial: Trial, args) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        args: Command line arguments
        
    Returns:
        Validation loss
    """
    # Sample hyperparameters
    params = define_hyperparameter_space(trial, args)
    
    # Create model and data module
    model, data_module = create_model_and_datamodule(params, args)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min"
        )
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=Path(args.output_dir) / "logs",
        name=f"{args.study_name}_trial_{trial.number}"
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs_per_trial,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu" if args.gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Return best validation loss
    return trainer.callback_metrics["val_loss"].item()


def tune_hyperparameters(args):
    """
    Run hyperparameter tuning with optuna.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Define partial objective function with args
    objective_with_args = lambda trial: objective(trial, args)
    
    # Run optimization
    study.optimize(objective_with_args, n_trials=args.n_trials)
    
    # Print and save results
    print("\nHyperparameter Tuning Results:")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (validation loss): {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_trial.params.items():
        print(f"{param}: {value}")
    
    # Save study results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = output_dir / f"{args.study_name}_results_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump({
            "study_name": args.study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "all_trials": [
                {
                    "trial": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in study.trials
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        # Create plots
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        
        # Save plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        history_path = plots_dir / f"{args.study_name}_history_{timestamp}.png"
        importance_path = plots_dir / f"{args.study_name}_importance_{timestamp}.png"
        
        fig1.write_image(str(history_path))
        fig2.write_image(str(importance_path))
        
        print(f"Plots saved to {plots_dir}")
    except ImportError:
        print("Optuna visualization requires plotly. Install with 'pip install plotly'.")
    
    # Train a final model with the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    best_params = study.best_trial.params
    
    model, data_module = create_model_and_datamodule(best_params, args)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs_per_trial * 2,  # Train for longer
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=f"{args.study_name}_best_{timestamp}",
                monitor="val_loss",
                mode="min"
            )
        ],
        logger=TensorBoardLogger(
            save_dir=output_dir / "logs",
            name=f"{args.study_name}_final"
        ),
        accelerator="gpu" if args.gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True
    )
    
    trainer.fit(model, data_module)
    
    # Save model parameters separately
    model_path = output_dir / f"{args.study_name}_final_weights_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    
    print(f"Final model saved to {model_path}")


def main():
    """Main function."""
    args = parse_args()
    
    print(f"\nHyperparameter tuning for {args.attention_type} on {args.dataset}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    
    # Run hyperparameter tuning
    tune_hyperparameters(args)


if __name__ == "__main__":
    main()
